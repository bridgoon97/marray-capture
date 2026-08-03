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

这些是踩过的坑。**它们的共同点是出错不报异常，只是数据悄悄变错。**

### 1. 通道顺序：麦序号决定，物理通道号不决定

麦克风挂在声卡的哪几个通道上是任意的。`ChannelMap` 里三件事分开：
`enabled`（录不录）/ `role`（mic/vpu/ref）/ `order`（第几个麦克风）。

`AudioConfig.ordered_channels()` 的落盘顺序固定为
**麦克风（按 order）→ VPU → 参考麦**，于是：

- `channel_layout()` 返回的 `mic_cols` **恒为 `[0, 1, ... n-1]`**
- IR 文件第 k 列 = 第 k 个麦克风 = 阵列几何坐标的第 k 行

**改动这套顺序等于改动 IR 的通道语义**，会让后处理的
`sinc(2πf·d_ij/c)` 按错误的麦间距算，下游波束形成器得到过于乐观的结果。
真要改，`rir/augment_runner.py` 的几何校验和 `gui/device_page.py`
的摘要行要一起改。

### 2. 电平回调是物理通道顺序的

`AudioEngine` 的 `level_cb` 拿到的是原始 `indata` 的逐通道 RMS，
**列序是声卡物理顺序**，不是落盘顺序。任何新的电平显示路径都必须过
`engine.map_levels(rms, indices)`，否则标签和读数会对不上
（mic1 那一行显示的是 ch1 的电平）。

### 3. 播放缓冲区布局在两种收发模式下必须一致

都是 `[guard 静音][play][tail 静音]`，录音从 0 开始。上层靠这条关系定位：

```
录音域到达位置 = guard + starts[k] + 设备延迟
```

`extract.locate_peaks()` 的搜索窗是 `guard_s + max_latency_s`。
动了任何一头，另一头要跟着动，否则峰值定位会落在窗外。

### 4. 质检的噪声窗取在直达峰**之前**

ESS 的谐波失真产物落在 `-L·ln(k)`（`L = T/ln(f2/f1)`）。二次谐波之后到直达峰
之间那段既没有失真产物、也没有混响，是干净的反卷积域噪底。
用峰后的尾巴估会把真实混响算成噪声，偏悲观。别"顺手改成峰后"。

### 5. IR 一律不做峰值归一

通道间与位置间的**绝对电平关系**是后面算 ILD / DRR 的依据。
`store.save_ir()` 存 32-bit float 原值。`rir_augment/io_utils.save_rir()`
默认会归一化——vendored 代码里那个函数**本工具没用它**，别顺手换过去。

### 6. Qt 的 `clicked` 会带一个 bool

`btn.clicked.connect(self.foo)` 会调用 `foo(False)`。
如果 `foo` 的第一个位置参数不是 bool，就会在运行时炸。
**统一用 `lambda: self.foo()` 转接。**（已经踩过一次：
`only_take_ids=False` → `x in False` → "argument of type bool is not iterable"。）

### 7. QSS 的优先级高于 `setFont()`

`theme.stylesheet()` 里有 `* { font-size: ...pt }` 这样的通配规则，
它会盖掉用 `QFont` 设的字号。**字号一律走 QSS + objectName**
（例如指令卡的 `QLabel#display`）。

### 8. 不要在 GUI 线程上打开音频设备

`sd.check_input_settings` / `check_output_settings` 会**真的去打开设备**，
碰上虚拟声卡或正在重连的蓝牙设备会阻塞几十秒，界面直接假死。

`audio/devices.py` 已经分好了：`list_devices` / `supported_rates` / `validate`
只读描述信息；`probe_rates` / `probe_settings` 会阻塞，**必须放 `Worker` 线程**。

### 9. 页面控件回填要挡住写回

`DevicePage.pull()` 有 `_loading` 守卫：回填时每个 `setValue` 都会触发
`valueChanged`，而那时其余控件还停在 Qt 默认值上，`_push()` 会把默认值写回配置
（输出增益 −6 dB 变 0 dB、输出通道数 2 变 1）。

守卫解除后**要同时补调 `_push_channels()` 和 `_push()`** ——
只补一个的话电平表和通道摘要会一直是空的。

### 10. WASAPI 只接受完整通道数

WASAPI 共享模式不接受端点混音格式的子集，8 通道卡上请求 5 个会报
`Invalid number of channels`。`AudioEngine._open_input()` 会自动退到全通道数
再切片，并把结果记进 `input_channels_override`。新增开流路径记得也走它。

---

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

`tests/` 下 56 个用例，**不需要声卡、不联网**。手法：

- **仿真信号**：已知 IR → 卷积 → 加噪 → 走真实的反卷积/提取/质检链路，比对还原度
- **替身声卡**：`FakeStream` / `FakeEngine` 驱动真实的回调逻辑
- **替身 TTS**：`FakeBackend` 合成固定长度的正弦
- **离屏 GUI**：`QT_QPA_PLATFORM=offscreen`，验证配置不被冲掉、控件状态、注意事项联动

新增功能时，优先补**能抓住静默错误**的用例（顺序、对齐、映射），
而不是「函数被调用了」这种。上面「会静默出错的地方」每一条都应该有对应用例。

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

---

## 相关文档

- [README.md](README.md) —— 设计要点与原理
- [docs/deploy-windows.md](docs/deploy-windows.md) —— 正式采集环境
- [docs/deploy-macos.md](docs/deploy-macos.md) —— 开发排练环境
- [docs/protocol-params.md](docs/protocol-params.md) —— 采集参数逐个说明（自动生成）
