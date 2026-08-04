# AGENTS.md

给在这个仓库里干活的编码 agent 看的。人也可以看。

---

## 这是什么

开放式耳机麦克风阵列的**干扰人扫频实录**采集工具。一个 PySide6 GUI 串起五步：
设备配置 → 采集方案 → 语音导播采集 → IR 可靠性验证 → 混响随机增强。

**使用场景决定了很多设计**：操作者戴着待测耳机、坐在房间另一头的转椅上，
手够不到电脑。所以采集流程是全自动语音导播的，采集页要三米开外能读懂。

不是产品，是给一两个人用的实验室仪器。别按 SaaS 的标准加东西。

---

## 常用命令

```bash
uv sync --extra tts                 # 装依赖（tts 是可选组：piper + edge-tts）
uv run marray-capture               # 启动 GUI
uv run marray-capture --check       # 环境自检：列声卡设备 + TTS 后端可用性
uv run marray-capture --dump-params # 输出参数说明 markdown（见下方「单一真源」）
uv run pytest -q                    # 全部测试，不需要声卡、不联网
uv run pyflakes src tests           # 静态检查
```

测试里 `tests/test_gui.py` 会自己设 `QT_QPA_PLATFORM=offscreen`，不用手动导。

改完 GUI **一定要看渲染结果**，别只跑测试。离屏截图：

```python
os.environ["QT_QPA_PLATFORM"] = "offscreen"
app = QApplication([]); theme.apply(app)
w = MainWindow(settings); w.show(); app.processEvents()
w.grab().save("/tmp/ui.png")
```

这套流程抓到过三个测试抓不到的 bug（字号被 QSS 盖掉、电平表启动时是空的、
注意事项重复三遍）。

---

## 代码地图

| 路径 | 职责 |
|---|---|
| `settings.py` | 所有配置的 dataclass + JSON 持久化。**通道映射的语义在这里定义** |
| `audio/sweep.py` | ESS 生成 / 逆滤波器 / 反卷积 / 激励拼装 |
| `audio/engine.py` | 播放录音引擎。全双工单流 + 分离双流两种模式 |
| `audio/devices.py` | 声卡枚举。**分「不打开设备的快查」和「会阻塞的探测」两套 API** |
| `audio/prompts.py` | 语音导播的渲染与缓存；提示音生成 |
| `audio/tts.py` | 四个 TTS 后端（piper / edge / sapi / say）与自动选择 |
| `protocol/plan.py` | 采集方案生成（圈、格、随机位） |
| `protocol/runner.py` | 会话执行器。不依赖 Qt，GUI 用回调接 |
| `qc/metrics.py` | 逐位置的 IR 可靠性验证 |
| `qc/requantify.py` | 用最新 QC 逻辑对**已落盘的 raw** 重跑, 改判旧会话 |
| `scripts/analyze_session.py` | 离线诊断某个会话的脚本 |
| `rir/extract.py` | 反卷积域定位 + 裁剪 + 平均 + 重采样 |
| `rir/speaker_eq.py` | 去音箱响应（最小相位反滤波器） |
| `rir/augment_runner.py` | 混响随机增强的批处理 |
| `rir/rir_augment/` | **vendored**，来自作者的 `rir-augment` 项目，见下 |
| `store.py` | 会话目录与 manifest |
| `gui/theme.py` | 设计 token + QSS。改配色/字号只改这里 |
| `gui/widgets.py` | 通用控件（电平表、IR 图、指令卡、注意事项按钮…） |
| `gui/notes.py` | **注意事项与参数说明的内容**，纯数据 |

---

## 会静默出错的地方（改动前必读）

这些是踩过的坑。**共同点是出错不抛异常，只是数据悄悄变错。**
分三组：数据语义 / 信号处理 / 实时性与界面。

---

## A. 数据语义

### A1. 通道顺序：麦序号决定，物理通道号不决定

麦克风挂在声卡的哪几个通道上是任意的。`ChannelMap` 里三件事分开：
`enabled`（录不录）/ `role`（mic/vpu/ref）/ `order`（第几个麦克风）。

`AudioConfig.ordered_channels()` 的落盘顺序固定为
**麦克风（按 order）→ VPU → 参考麦**，于是：

- `channel_layout()` 返回的 `mic_cols` **恒为 `[0, 1, ... n-1]`**
- IR 文件第 k 列 = 第 k 个麦克风 = 阵列几何坐标的第 k 行

**改动这套顺序等于改动 IR 的通道语义**，会让后处理的 `sinc(2πf·d_ij/c)`
按错误的麦间距算，下游波束形成器得到过于乐观的结果。真要改，
`rir/augment_runner.py` 的几何校验和 `gui/device_page.py` 的摘要行要一起改。

### A2. 电平回调是物理通道顺序的

`AudioEngine` 的 `level_cb` 拿到的是原始 `indata` 的逐通道 RMS，
**列序是声卡物理顺序**，不是落盘顺序。任何新的电平显示路径都必须过
`engine.map_levels(rms, indices)`，否则标签和读数会对不上
（mic1 那一行显示的是 ch1 的电平）。

### A3. VPU 是非声学通道，不能套声学判据

VPU 拾的是骨导/接触振动，中低频本来就没有声学信号，SNR / DDR / 可靠带宽
偏低是**物理必然**，不是故障。`evaluate_take(..., vpu_cols=...)` 会把这些
判据对 VPU 关掉，只保留削顶和"几乎没信号"。

而且**不能用绝对电平门限判 VPU 死没死**：耳机被人头一挡，VPU 峰值能掉到
−50 dBFS 以下，但信号仍高出本底十几 dB。判据是 `rec_snr < 6 dB`
（真断线时信号≈噪底，rec_snr≈0）。

麦克风的死通道判据同理用**相对量**：比同一次采集里最响的麦低 30 dB 以上。
同源采集里所有麦录的是同一个声源，绝对电平会随距离整体起落，相对关系才稳。

### A4. IR 不做**逐条**峰值归一

通道间与位置间的**绝对电平关系**是后面算 ILD / DRR 的依据。
`store.save_ir()` 存 32-bit float 原值。`rir_augment/io_utils.save_rir()`
默认会归一化——vendored 代码里那个函数**本工具没用它**，别顺手换过去。

**这条禁的是「每条 IR 各自除以自己的峰值」**，那会毁掉位置间的相对电平。
用**整批共享的单一标量**缩放是允许的（后处理页的「全局归一化」选项就是：
取整批全局峰值，全部乘同一个 `0.98/peak`，scale 记进 manifest）。
判断标准很简单：缩放因子是不是所有 IR 共用一个。

同理，IR 波形图画**绝对 dB**（0 dB = 全刻度直通），不做显示归一化 ——
之前时域图归一到 0 dB、频响图用绝对值，两张图自相矛盾，看着像削顶。

---

## B. 信号处理

### B1. 播放缓冲区布局在两种收发模式下必须一致

都是 `[guard 静音][play][tail 静音]`，录音从 0 开始。上层靠这条关系定位：

```
录音域到达位置 = guard + starts[k] + 设备延迟
```

`extract.locate_peaks()` 的搜索窗是 `guard_s + max_latency_s`。
动了任何一头，另一头要跟着动，否则峰值定位会落在窗外。

### B2. 直达峰定位有三道防线，别简化成 argmax

这块是整个管线里最容易出**看似合理实则垃圾**的地方，实测踩过三次：

1. **prominence 守卫**（`_PROMINENCE_DB = 30`）。搜索窗里的能量峰若不明显高出
   噪底就抛 `PeakNotFoundError`，**不要静默抓 argmax**。真实直达峰经匹配滤波
   相干积分有 42+ dB prominence（即使 SNR≈0）；峰落窗外时窗内只有 18~24 dB，
   纯噪底 ~10 dB。抓错了下游会算出 −4400 ppm / NCC 0.02 / DDR 0~10 这种
   "看着像佩戴者动了"的假指标。
2. **窄窗搜不到要回退全局窗**。标定值是上次测的，WASAPI 共享模式的启动延迟
   每次抖几十毫秒，死守窄窗会误杀正常 take。回退后仍无峰才真抛错。
3. **后续扫频的峰位用互相关模板匹配**（`_locate_subsequent`），不是 argmax、
   也不是 onset。直达被头遮挡时，1~3 ms 处的反射幅度**可能比直达峰还高**，
   argmax 锁反射；而"最早的显著峰"也会被直达之前的反射骗到。
   拿第一次扫频直达峰 ±1 ms 当模板做跨通道互相关：直达对齐时两次扫频
   **共享的全部早反射同时对上**，总相关最高，单个强反射拼不过。

`PeakNotFoundError` 在采集路径上要 catch 成这一条 FAIL 并继续；
在标定/参考 IR 路径上让它上抛到 Worker，UI 直接显示。

### B3. 多次扫频求平均/算 NCC 前要做亚样本对齐

按整样本直达峰对齐后仍有亚样本错位，直接平均会梳状衰减高频，把 NCC 压到 0.8x。
`extract_take` 从第 2 条起按互相关峰（抛物线插值）做 **FFT 相位平移**对齐 ——
全通，不动幅度谱，绝对电平关系不受影响。

### B4. 质检的噪声窗取在直达峰**之前**，峰值取在信号窗内

ESS 的谐波失真产物落在 `-L·ln(k)`（`L = T/ln(f2/f1)`，几百 ms）。二次谐波之后
到直达峰之间那段既没有失真产物、也没有混响，是干净的反卷积域噪底。
用峰后的尾巴估会把真实混响算成噪声，偏悲观。别"顺手改成峰后"。

`peak_dbfs` 要在**扫频信号窗**里取，不能对整段 `rec` 取 max ——
开流瞬态（guard 区约 5 ms 处近 0 dBFS 的脉冲）会被误判成削顶。
噪声窗下界也要抬过前 10 ms，免得噪底被那个脉冲估高、连累录音 SNR。

### B5. WASAPI 只接受完整通道数

WASAPI 共享模式不接受端点混音格式的子集，8 通道卡上请求 5 个会报
`Invalid number of channels`。`AudioEngine._open_input()` 会自动退到全通道数
再切片，并把结果记进 `input_channels_override`。新增开流路径记得也走它。

这个协商结果**不算异常**，不要因为它把 take 判成 WARN（字符串仍存进 qc 供查看）。

---

## C. 实时性与界面

### C1. 实时音频回调路径上不能有高频 Qt 重绘

**这条会毁掉录音，不只是卡界面。** PortAudio 的回调是 native 线程，跑 Python
回调要拿 GIL；GUI 线程若卡在 QSS 重绘里抱死 GIL，音频回调拿不到 GIL 就错过
截止时间 → 输出 underrun → 扫频有缺口 → 录音的时间轴作废。

实际踩过：`LevelMeter` 每帧对每个 bar `setStyleSheet`（整条 QSS 重新解析 + 重绘），
采集时扬声器扫频肉眼可闻地卡顿。三条对策，改这块时别退回去：

- 缓存每个 bar 的颜色档，**只在跨档（OK/WARN/BAD）时** `setStyleSheet`，
  同档之下只 `setValue`
- 电平刷新**节流到 ~25 Hz**（分离双流模式的回调是每块都来的，上百 Hz）
- 音频回调里只做 `indata[:n].copy()`（float32 memcpy），
  RMS 的算术挪到消费线程 `_drain_levels`

### C2. pyqtgraph 的性能要显式打开

IR 默认 1 秒 × 48 kHz = 4.8 万点/通道，默认设置下平移/缩放每帧重绘全部点。
三件事缺一不可（做完 IRView 两图重绘 55 ms → 1.4 ms/帧）：

- `setDownsampling` 只作用于**已存在的曲线** —— 必须在 `show_ir` 里
  逐条曲线设 `setDownsampling(method='peak')`，在 `__init__` 里设无效
- 关 `antialias`（点密包络看不出锯齿）
- 坐标轴（含 grid）和每条曲线开 `DeviceCoordinateCache`，
  否则 grid 每帧重画约 30 ms

图上的鼠标平移/缩放/右键菜单是**故意关掉的**：这套界面三米外看，
不在图上拖拽探查，关掉杜绝交互卡顿。要细看就拿原始 IR 文件。

### C3. Qt 的 `clicked` 会带一个 bool

`btn.clicked.connect(self.foo)` 会调用 `foo(False)`。
如果 `foo` 的第一个位置参数不是 bool，就会在运行时炸。
**统一用 `lambda: self.foo()` 转接。**（踩过：`only_take_ids=False` →
`x in False` → "argument of type bool is not iterable"。）

### C4. QSS 的优先级高于 `setFont()`

`theme.stylesheet()` 里有 `* { font-size: ...pt }` 这样的通配规则，
它会盖掉用 `QFont` 设的字号。**字号一律走 QSS + objectName**
（例如指令卡的 `QLabel#display`）。

### C5. 不要在 GUI 线程上打开音频设备

`sd.check_input_settings` / `check_output_settings` 会**真的去打开设备**，
碰上虚拟声卡或正在重连的蓝牙设备会阻塞几十秒，界面直接假死。

`audio/devices.py` 已经分好了：`list_devices` / `supported_rates` / `validate`
只读描述信息；`probe_rates` / `probe_settings` 会阻塞，**必须放 `Worker` 线程**。

`engine.play()` 用自己的 `OutputStream` 而不是 `sd.play`，因为后者完全不检查
中止 —— 采集页按「停止」时 20 秒的倒计时会毫无反应。回调里查 `_abort` 和
`pause_event`，停止能立刻打断引导语/倒计时/反馈音，暂停能把播放在原位挂住。

### C6. 页面控件回填要挡住写回

`DevicePage.pull()` 有 `_loading` 守卫：回填时每个 `setValue` 都会触发
`valueChanged`，而那时其余控件还停在 Qt 默认值上，`_push()` 会把默认值写回配置
（输出增益 −6 dB 变 0 dB、输出通道数 2 变 1）。

守卫解除后**要同时补调 `_push_channels()` 和 `_push()`** ——
只补一个的话电平表和通道摘要会一直是空的。

设备或 `output_latency` 一变，要**清掉 `measured_latency_samples`** ——
留着旧标定值会让窄窗搜不到真实峰。

---

## 旧数据可以救回来

改了 QC 判据之后，不必重采：`qc/requantify.py` 对已落盘的 **raw 录音**用最新
逻辑重跑 deconv → 定位 → 切 IR → evaluate，改判 manifest、重存 IR、重出 qc.csv。
质检页有对应按钮。

它**不依赖录制时的 `starts`/leadin**：两次扫频的直达峰按"间隔≈名义间隔"配对找
（取最高的那对），第二个峰再走互相关。所以哪怕当初的元数据不全也能重跑。

**这也是「保存原始录音」默认开着的理由** —— 关掉就没有退路了。

## 约定

**语言**：所有面向用户的文本、文档、代码注释一律**中文**。提交信息也是中文。

**注释写「为什么」**，不写「做了什么」。这个仓库里绝大多数注释都在解释
某个取舍背后的物理原因或踩过的坑——保持这个密度和风格。

**参数说明是单一真源**：`gui/notes.py` 的 `PARAMS` 同时供
① 控件 tooltip ② 界面「参数说明」小窗 ③ `docs/protocol-params.md`。
**不要手改那个 md**，改 `PARAMS` 然后重新生成：

```bash
uv run marray-capture --dump-params > docs/protocol-params.md
```

**注意事项**同理写在 `gui/notes.py`。两条编写原则：
只写做错了会静默出错或事后无法补救的；尽量挂 `when=` 谓词做成跟配置联动的，
常驻的通用清单会被当成墙纸。

**vendored `rir_augment`**：来自作者本地的 `~/Projects/rir-augment`（未托管）。
本工具对它有两点**有意的**差异，改动时别丢：
1. 只对麦克风通道合成扩散尾，VPU 等非声学通道保留实测原样
2. 按质检结论筛输入，FAIL 的位置不进增强

---

## 测试

`tests/` 下 69 个用例，**不需要声卡、不联网**。手法：

- **仿真信号**：已知 IR → 卷积 → 加噪 → 走真实的反卷积/提取/质检链路，比对还原度
- **替身声卡**：`FakeStream` / `FakeEngine` 驱动真实的回调逻辑
- **替身 TTS**：`FakeBackend` 合成固定长度的正弦
- **离屏 GUI**：`QT_QPA_PLATFORM=offscreen`，验证配置不被冲掉、控件状态、注意事项联动

新增功能时，优先补**能抓住静默错误**的用例（顺序、对齐、映射），
而不是「函数被调用了」这种。上面「会静默出错的地方」每一条都应该有对应用例。

**构造用例时要还原真实的失效条件，不要只喂理想信号。** 这一轮几条 bug 都是
理想信号测不出来的，对应的回归用例长这样：

- 直达峰定位：IR 要带**多个早反射和衰减尾**，第二次扫频只把**直达压低 22 dB**
  模拟头遮挡（反射不动）—— argmax 会锁反射并报几百 ppm 假漂移
- 亚样本对齐：给 0.5 个样本的错位，断言对齐后 NCC > 0.99
- 削顶判据：在 guard 区放一个近 0 dBFS 的启动脉冲
- VPU：造「峰值 < −50 dBFS 但 SNR > 6 dB」的遮挡场景，和真·死通道各一条
- 电平表：断言同档不重复 `setStyleSheet`、跨档才设（能抓住 C1 的旧写法）

---

## 已经做过的决定，不要推翻

- **不换 Web/Electron 前端。** 音频设备枚举和多通道全双工采集都在 Python 里，
  换栈要加 IPC 层且零功能收益；pyqtgraph 在实时波形上比 JS 图表快；
  部署从一条 `uv run` 变成打包浏览器运行时。Qt 观感的问题用 QSS 解决。
- **Windows 上用 WASAPI，不要假设有 ASIO。** PyPI 的 `sounddevice` 轮子自带的
  PortAudio 没编译 ASIO 支持，装了 ASIO4ALL 也不会出现 ASIO 条目。
  代码里的 ASIO 全双工分支保留着，但不是主路径。
- **采集协议不做角度标定。** 方位全用「向右转一格」，角度标签只是名义值，
  只用于分组统计，不参与任何计算。
- **每个位置扫 2 次**是硬要求，两次之间的一致性（NCC）是唯一的质检依据。
- **「保存原始录音」默认开着。** 判据改了可以用 `qc/requantify.py` 重跑救回旧数据，
  关掉就没退路了。
- **IR 图不做交互**（平移/缩放/右键菜单全关）。三米外看的界面，交互只会带来卡顿。

---

## 排查现场问题的顺序

真实采集里出问题时，按这个顺序看，能少走弯路：

1. **先看 QC 详情里的 DDR 分量** `ir_peak_db` / `ir_noise_db`，而不是只看 DDR 比值 ——
   能分清是"峰太小（定位错/扫频没录好）"还是"噪底太高（漂移失配/失真/爆音）"
2. **NCC 低 + 漂移几百 ppm** 先怀疑峰位锁错（B2），不是真的佩戴者动了。
   验证方法：按名义位置切 IR 重算 NCC，若接近 1 就是定位问题
3. **扫频听着卡顿** = 输出 underrun，查 GUI 线程有没有在高频重绘（C1）
4. **"找不到峰"** 查 `output_latency`（实测有声卡在 high 下启动延迟 8~19 秒）、
   设备页重做「延迟标定」、必要时调大「最大延迟余量」
5. `scripts/analyze_session.py` 可以离线把一个会话的指标拉出来对比

---

## 相关文档

- [README.md](README.md) —— 设计要点与原理
- [docs/deploy-windows.md](docs/deploy-windows.md) —— 正式采集环境
- [docs/deploy-macos.md](docs/deploy-macos.md) —— 开发排练环境
- [docs/protocol-params.md](docs/protocol-params.md) —— 采集参数逐个说明（自动生成）
