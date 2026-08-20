# 篮球标注审核工具（独立版）

从 `D:\bball_training_pack\scripts\annotator.py` 抽离出的**纯标注审核工具**，已移除「模型重识别/补标」功能，可独立运行、独立打包，不依赖原训练项目。

## 文件

| 文件 | 说明 |
|---|---|
| `annotator.py` | 源码（纯审核版，仅依赖 tkinter + Pillow） |
| `dist/篮球标注审核工具.exe` | 打包好的独立可执行文件（约 32MB，双击即用，无需装 Python） |

## 使用

- **双击** `dist/篮球标注审核工具.exe`，弹窗选择审核目录。
- 或命令行：`py annotator.py [审核目录]`

## 支持的目录格式（自动检测）

1. **LabelImg 平铺格式**：图片 + 同名 `.txt` + `classes.txt`（`review/` 下的目录）
2. **YOLO 数据集格式**：`images/` + `labels/` + `data.yaml`（或 `train/`、`val/` 分集）

## 快捷键

| 按键 | 功能 |
|---|---|
| ← / → | 上一张 / 下一张 |
| D / F | 通过 / 不通过 并下一张 |
| W / Esc | 进入 / 退出绘制模式 |
| Del | 删除选中框 |
| 0-9 | 设定绘制类别 |
| 右键框 | 改该框类别 |
| [ / ] | 上一处 / 下一处缺失 |
| Ctrl+S | 保存当前帧 |
| Ctrl+0 | 适应窗口 |
| 空格+拖动 | 平移画面 |

## 审核进度保存

审核结论写入审核目录下：
- `.review_state.json` — 主状态（pass/fail）
- `.reviewed_frames.txt` / `.failed_frames.txt` / `.not_reviewed.txt` — 兼容 `review_tracker.py`

## 与原项目的关系

- 原文件 `D:\bball_training_pack\scripts\annotator.py` **未被改动**。
- 本独立版已删除「重识别 / 批量补标」功能（原依赖 `best.pt` + `ultralytics` + `torch`）。
- 其余审核流程（看框 / 改框 / 打标记 / 保存）与原版完全一致。

## 重新打包

```powershell
cd C:\Users\12479\bball_annotator_tool
py -m PyInstaller --onefile --windowed --name BasketballAnnotator annotator.py
# 产物在 dist\BasketballAnnotator.exe，可自行重命名
```
