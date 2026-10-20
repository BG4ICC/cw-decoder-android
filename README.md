# CW 解码器（CW AI Decoder for Android）

由 **BG4ICC** 开发的安卓摩尔斯电码（CW）解码器，解码核心使用 **DeepCW** 开源神经网络内核
（[e04/deepcw-engine](https://github.com/e04/deepcw-engine)），在弱信号、QSB、噪声环境下
显著优于传统 DSP 解码。**完全免费、完全开源**。

## 功能特性

- **三通道解码**（b 版）：推荐通道（DeepCW 神经网络 + 语言模型纠错）/ NN 直读 / 传统通道
- **n 版**：双通道精简版（推荐 + NN 直读），可与 b 版同机共存
- 实时瀑布图与频率锁定，12–45+ WPM 自适应速度判定
- 通联排版：呼号识别、段落切分、RST/报告语义吸附
- **发送页**：文本拍发、队列顺序发、智能词板（按发送上下文概率排序的 Q 简语/缩略语/自定义词）
- 后台持续收听（前台服务 + 唤醒锁），纯本地运行，零联网
- 单文件 WebView 架构：全部解码逻辑在 `index.html` 内，DeepCW 引擎经 ONNX Runtime Web 本地推理

## 目录结构

```
index.html        全部 UI 与解码/发送逻辑（单文件，含内嵌 CWNET/KEYNET 权重）
deepcw/           DeepCW 引擎资产：model.onnx / ednet.onnx / onnxruntime-web wasm / worker
android/          原生壳：MainActivity.java（WebView 容器）、CwRxService.java（后台收听前台服务）
build.sh          免 Android Studio 命令行构建脚本（SDK 自举 + ecj 编译 + aapt2 打包签名）
version.txt       versionCode（b 版基线，n 版自动 -100）
versionname.txt   versionName 基线
ic_launcher.png   应用图标
```

## 构建

依赖：Linux 环境 + curl + JDK（仅用 ecj.jar 编译，构建脚本自动下载 Android SDK 与 ecj）。

```bash
bash build.sh b    # b 版（三通道全功能）：包名 com.cwdecoder.app，VC=version.txt
bash build.sh n    # n 版（双通道精简）：包名 com.cwdecoder.app.n，VC=version.txt-100，可同机共存
```

产物为已签名 APK（调试签名请自行替换 `build.sh` 中的 keystore 路径；仓库不含签名私钥）。

## 开源许可与署名

- 本项目整体以 **AGPL-3.0-only** 发布（见 [LICENSE](LICENSE)），以满足 DeepCW 内核的许可证要求。
- DeepCW 内核（`deepcw/model.onnx`、`deepcw/ednet.onnx`、`deepcw/model.onnx.json`）来自
  [e04/deepcw-engine](https://github.com/e04/deepcw-engine)，License: AGPL-3.0-only；
  `deepcw/dcwpre.js`、`dcwlm.js`、`dcwworker.js` 为基于其参考实现的移植与扩展。
- `deepcw/ort*` 为 [ONNX Runtime](https://github.com/microsoft/onnxruntime)（MIT）。
- 详见 [NOTICE](NOTICE)。

## 致谢

感谢 DeepCW 作者（e04）开源高精度 CW 神经网络解码模型，让移动端弱信号抄收成为现实。73！
