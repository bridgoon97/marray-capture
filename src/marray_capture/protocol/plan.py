"""采集方案生成。

设计原则 (对应「不做精确角度标定」的要求):

1. **不报绝对角度, 只报相对动作。** 方位维度全部用「向右转一格」完成, 佩戴者坐在
   转椅上原地转即可。角度标签只是名义值, 用于分组统计, 不参与任何计算。
2. **每转四分之一圈给一个粗方向校验句** ("音箱现在大致在你左侧"), 防止累计误差跑飞。
3. **音箱朝向是「圈级」参数不是「位置级」参数。** 音箱固定指向椅子, 佩戴者原地旋转时
   音箱始终正对他; 想要「干扰人没正对你」的情形, 只能整圈重来一次并把音箱转个方向。
   这样每种朝向只需要走过去调一次。
4. **循环顺序按操作成本排**: 高度 (要走到支架旁) > 距离 (挪椅子) > 方位 (转椅子)。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from ..settings import ProtocolConfig

# 粗方向标签, 方位角以逆时针为正 (0=正前, 90=正左, 180=正后, 270=正右)
_COARSE = [
    (0, "正前"), (45, "左前"), (90, "正左"), (135, "左后"),
    (180, "正后"), (225, "右后"), (270, "正右"), (315, "右前"),
]


def coarse_label(az_deg: float) -> str:
    a = az_deg % 360.0
    best = min(_COARSE, key=lambda kv: min(abs(a - kv[0]), 360 - abs(a - kv[0])))
    return best[1]


def _dist_words(cm: int) -> str:
    if cm % 100 == 0:
        return f"{cm // 100} 米"
    if cm < 100:
        return f"{cm} 厘米"
    return f"{cm / 100:.1f} 米"


def _height_words(cm: int) -> str:
    if cm >= 150:
        return f"{_dist_words(cm)}，站立说话的高度"
    if cm >= 110:
        return f"{_dist_words(cm)}，坐着说话的高度"
    return f"{_dist_words(cm)}，比坐姿再低一些"


def _orient_words(deg: int) -> str:
    return {0: "正对着你", 90: "侧对着你", 180: "背对着你"}.get(deg % 360, f"转过 {deg} 度")


@dataclass
class Step:
    """方案里的一步。kind='setup' 只播指令不录音; kind='measure' 要录。"""

    idx: int
    kind: str                       # setup | measure
    instruction: str
    settle_s: float
    take_id: str = ""
    tag: str = ""                   # grid | orient | rewear | random
    subject_id: str = ""
    wearing_id: str = ""
    side: str = ""
    distance_cm: int | None = None
    height_cm: int | None = None
    speaker_deg: int = 0
    az_index: int | None = None
    az_nominal_deg: float | None = None
    az_label: str = ""
    stance: str = "坐"
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    @property
    def measures(self) -> list[Step]:
        return [s for s in self.steps if s.kind == "measure"]

    def estimated_seconds(self, per_take_s: float) -> float:
        """per_take_s = 激励波形时长 (preroll+扫频+gap+tail)。"""
        total = 0.0
        for s in self.steps:
            total += len(s.instruction) / 5.0 + s.settle_s          # 语音约 5 字/秒
            if s.kind == "measure":
                total += per_take_s + 1.0                            # +1s 处理与落盘
        return total

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": self.config, "steps": [s.to_dict() for s in self.steps]}
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Plan":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(steps=[Step(**s) for s in d["steps"]], config=d.get("config", {}))

    def all_instructions(self) -> list[str]:
        seen, out = set(), []
        for s in self.steps:
            if s.instruction and s.instruction not in seen:
                seen.add(s.instruction)
                out.append(s.instruction)
        return out


def _ring(
    steps_out: list[Step],
    cfg: ProtocolConfig,
    distance_cm: int,
    height_cm: int,
    speaker_deg: int,
    n_steps: int,
    tag: str,
    wearing_id: str,
    need_height_setup: bool,
    counter: list[int],
) -> None:
    """生成一圈: 一个 setup 步 + n_steps 个测量步。"""
    setup_bits = []
    if need_height_setup:
        setup_bits.append(f"请把音箱支架调到大约 {_height_words(height_cm)}")
    if speaker_deg % 360 != 0:
        setup_bits.append(f"并把音箱转成{_orient_words(speaker_deg)}")
    setup_bits.append(f"然后坐回椅子，把椅子挪到距离音箱大约 {_dist_words(distance_cm)} 的位置，面向音箱")
    instruction = "，".join(setup_bits) + "。坐好以后保持不动，等提示音。"

    counter[0] += 1
    steps_out.append(Step(
        idx=counter[0], kind="setup", instruction=instruction,
        settle_s=cfg.setup_settle_s, tag=tag,
        subject_id=cfg.subject_id, wearing_id=wearing_id, side=cfg.side,
        distance_cm=distance_cm, height_cm=height_cm, speaker_deg=speaker_deg,
    ))

    per = 360.0 / n_steps
    for k in range(n_steps):
        az = (k * per) % 360.0
        label = coarse_label(az)
        if k == 0:
            instr = "面向音箱，保持不动。"
        else:
            instr = f"向右转一格，大约 {int(round(per))} 度。"
            # 每四分之一圈给一次粗方向校验, 防止累计误差
            if abs((az % 90.0)) < 1e-6 and az > 0:
                instr += f"音箱现在应该大致在你的{label}方向。"
        counter[0] += 1
        take = (f"{cfg.subject_id}_{wearing_id}_{cfg.side}"
                f"_D{distance_cm:03d}_H{height_cm:03d}_O{speaker_deg:03d}_A{k:02d}")
        steps_out.append(Step(
            idx=counter[0], kind="measure", instruction=instr, settle_s=cfg.settle_s,
            take_id=take, tag=tag,
            subject_id=cfg.subject_id, wearing_id=wearing_id, side=cfg.side,
            distance_cm=distance_cm, height_cm=height_cm, speaker_deg=speaker_deg,
            az_index=k, az_nominal_deg=round(az, 1), az_label=label, stance="坐",
        ))


def build_plan(cfg: ProtocolConfig, seed: int = 2026) -> Plan:
    """按配置生成完整方案。"""
    rng = np.random.default_rng(seed)
    steps: list[Step] = []
    counter = [0]

    steps.append(Step(
        idx=0, kind="setup",
        instruction=(
            f"开始采集。被试 {cfg.subject_id}，第 {cfg.wearing_id} 次佩戴，"
            f"耳机戴在{'右' if cfg.side.upper().startswith('R') else '左'}耳。"
            "请戴好耳机，检查佩戴位置，然后坐到椅子上。"
        ),
        settle_s=cfg.setup_settle_s, tag="grid",
        subject_id=cfg.subject_id, wearing_id=cfg.wearing_id, side=cfg.side,
    ))

    # ---- 主网格: 外层高度 (要走到支架旁), 中层距离 (挪椅子), 内层方位 (转椅子)
    last_height: int | None = None
    for height in cfg.heights_cm:
        for dist in cfg.distances_cm:
            dense = (dist == cfg.dense_distance_cm and height == cfg.dense_height_cm)
            n = cfg.dense_steps if dense else cfg.sparse_steps
            _ring(steps, cfg, dist, height, 0, n, "grid", cfg.wearing_id,
                  need_height_setup=(height != last_height), counter=counter)
            last_height = height

    # ---- 音箱朝向变体: 只在基准圈上, 每种朝向走一次稀疏圈
    for deg in cfg.speaker_orientations:
        if deg % 360 == 0:
            continue
        _ring(steps, cfg, cfg.dense_distance_cm, cfg.dense_height_cm, deg,
              cfg.orientation_subset_steps, "orient", cfg.wearing_id,
              need_height_setup=(cfg.dense_height_cm != last_height), counter=counter)
        last_height = cfg.dense_height_cm

    # ---- 重戴: 每次摘下重戴后走一个稀疏圈; 高度跟着音箱高度列表轮转,
    # 不要固定在一个值 —— 主网格里扫过的每个高度, 重戴时也该再覆盖到, 否则
    # 重戴后所有圈都停在同一个高度, 高度维度的重戴一致性就测不到了。
    heights = cfg.heights_cm or [cfg.rewearing_height_cm]
    for w in range(cfg.rewearing_rings):
        wid = f"{cfg.wearing_id}r{w + 1}"
        height = heights[w % len(heights)]
        counter[0] += 1
        steps.append(Step(
            idx=counter[0], kind="setup",
            instruction=(
                f"这是第 {w + 1} 次重新佩戴。请把耳机完全摘下来，随手放一下，"
                "再重新戴回同一只耳朵，不要刻意复原到原来的位置。戴好后坐回椅子。"
            ),
            settle_s=cfg.setup_settle_s, tag="rewear",
            subject_id=cfg.subject_id, wearing_id=wid, side=cfg.side,
        ))
        _ring(steps, cfg, cfg.rewearing_distance_cm, height, 0,
              cfg.rewearing_steps, "rewear", wid,
              need_height_setup=(height != last_height), counter=counter)
        last_height = height

    # ---- 随机抖动位: 给粗方向 + 粗距离 + 姿势, 打破整齐栅格
    if cfg.random_positions > 0:
        counter[0] += 1
        steps.append(Step(
            idx=counter[0], kind="setup",
            instruction=(
                "接下来是随机位置。每次会告诉你一个大概的方向、距离和姿势，"
                "不用量，走到差不多的位置站定或坐下就行。"
            ),
            settle_s=8.0, tag="random",
            subject_id=cfg.subject_id, wearing_id=cfg.wearing_id, side=cfg.side,
        ))
        labels = [lb for _, lb in _COARSE]
        for i in range(cfg.random_positions):
            az_lb = labels[int(rng.integers(len(labels)))]
            dist = int(rng.integers(40, 250))
            stance = ["坐着", "站着", "站着稍微弯腰"][int(rng.integers(3))]
            az_nom = float(next(a for a, lb in _COARSE if lb == az_lb))
            counter[0] += 1
            steps.append(Step(
                idx=counter[0], kind="measure",
                instruction=(
                    f"随机位置 {i + 1}。请让音箱大致落在你的{az_lb}方向，"
                    f"距离大约 {_dist_words(dist)}，{stance}。站定后保持不动。"
                ),
                settle_s=max(cfg.settle_s, 8.0),
                take_id=f"{cfg.subject_id}_{cfg.wearing_id}_{cfg.side}_R{i:02d}",
                tag="random",
                subject_id=cfg.subject_id, wearing_id=cfg.wearing_id, side=cfg.side,
                distance_cm=dist, height_cm=None, speaker_deg=0,
                az_index=None, az_nominal_deg=az_nom, az_label=az_lb, stance=stance,
                note="随机位, 标签为粗略值",
            ))

    counter[0] += 1
    steps.append(Step(
        idx=counter[0], kind="setup",
        instruction="本轮采集结束，谢谢。",
        settle_s=2.0, tag="grid",
        subject_id=cfg.subject_id, wearing_id=cfg.wearing_id, side=cfg.side,
    ))

    return Plan(steps=steps, config=asdict(cfg))
