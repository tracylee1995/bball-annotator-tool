#!/usr/bin/env python3
"""
篮球数据标注审核工具 (tkinter 桌面 GUI)

用途:
  逐张查看已预标注的帧，检查/修正标注框，并把审核进度存下来。

用法:
  py annotator.py                # 弹窗选择目录
  py annotator.py <审核目录>      # 直接打开审核目录

支持的目录格式 (自动检测):
  1. LabelImg 平铺格式: 图片 + 同名 txt + classes.txt
     (review/ 下的目录都是这种)
  2. YOLO 数据集格式: images/ + labels/ + data.yaml
     (datasets/ 下的目录)

快捷键:
  ← / →      上一张/下一张
  D          通过并下一张     F          不通过并下一张
  W          进入绘制模式     Esc        退出绘制 / 取消选中
  Del        删除选中框       0-9        设定绘制类别
  右键框     改该框类别
  [ / ]      上一处/下一处缺失   Ctrl+S    保存当前帧
  Ctrl+0     适应窗口         空格+拖动  平移画面

进度保存:
  审核结论(通过/不通过)写入目录下的 .review_state.json (每次标记时自动更新)
  同时导出 .reviewed_frames.txt(通过) / .failed_frames.txt(不通过) / .not_reviewed.txt(未审)。
"""
import argparse
import ast
import json
import re
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

DEFAULT_CLASSES = ["player", "basketball", "hoop", "backboard"]

# 类别配色 (高对比度, 循环使用)
PALETTE = [
    "#1E88E5",  # 蓝
    "#FB8C00",  # 橙
    "#43A047",  # 绿
    "#8E24AA",  # 紫
    "#E53935",  # 红
    "#00ACC1",  # 青
    "#FDD835",  # 黄
    "#6D4C41",  # 棕
    "#7E57C2",  # 深紫
    "#00897B",  # 青绿
    "#F06292",  # 粉
    "#5C6BC0",  # 靛蓝
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
STATE_FILE = ".review_state.json"
HANDLE_HIT = 8   # 角点命中半径(像素)
MIN_BOX_PX = 3   # 画新框的最小边长(像素)

# 快速过滤规则: (显示名, 匹配规则)
#   匹配规则 = None | ("missing", (类别名...)) | ("verdict", "pass"/"fail"/None)
#   "missing" 表示缺失该类别的帧; "verdict" 表示审核结论为 pass/fail/未审核(None) 的帧
FILTER_RULES = [
    ("无过滤", None),
    # 缺失检查
    ("缺球员", ("missing", ("player",))),
    ("缺篮球", ("missing", ("basketball",))),
    ("缺篮筐", ("missing", ("hoop",))),
    ("缺篮板", ("missing", ("backboard",))),
    ("缺核心类(球员/篮筐/篮板)", ("missing", ("player", "hoop", "backboard"))),
    # 审核结果
    ("未通过(F)", ("verdict", "fail")),
    ("已通过(D)", ("verdict", "pass")),
    ("未审核", ("verdict", None)),
]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def color_for(cls):
    return PALETTE[cls % len(PALETTE)]


class Box:
    """一个标注框 (YOLO 归一化坐标: cx cy w h, 均 0~1)"""
    __slots__ = ("cls", "cx", "cy", "w", "h")

    def __init__(self, cls, cx, cy, w, h):
        self.cls = int(cls)
        self.cx = float(cx)
        self.cy = float(cy)
        self.w = float(w)
        self.h = float(h)

    def to_line(self):
        return f"{self.cls} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"

    def pixel_rect(self, W, H):
        """返回图像像素坐标 (x1, y1, x2, y2), 保证 x1<=x2"""
        x1 = (self.cx - self.w / 2.0) * W
        y1 = (self.cy - self.h / 2.0) * H
        x2 = (self.cx + self.w / 2.0) * W
        y2 = (self.cy + self.h / 2.0) * H
        return x1, y1, x2, y2

    def copy(self):
        return Box(self.cls, self.cx, self.cy, self.w, self.h)


class Session:
    """数据集会话: 目录格式检测、文件清单、标签读写、审核状态持久化"""

    def __init__(self, root_dir):
        self.root = Path(root_dir).resolve()
        self.fmt, self.classes = self._detect()
        self.pairs = self._collect()          # [(img_path, lbl_path, stem), ...]
        self.verdicts = {}                    # stem -> "pass" | "fail"
        self._load_state()

    # ---- 目录格式检测 ----
    def _detect(self):
        classes_file = self.root / "classes.txt"
        if classes_file.exists():
            names = [ln.strip() for ln in classes_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            return ("flat", names if names else list(DEFAULT_CLASSES))

        data_file = self.root / "data.yaml"
        if data_file.exists():
            txt = data_file.read_text(encoding="utf-8")
            m = re.search(r"names\s*:\s*(\[.*\])", txt)
            if m:
                try:
                    names = ast.literal_eval(m.group(1))
                    if isinstance(names, list) and names:
                        return ("dataset", names)
                except (ValueError, SyntaxError):
                    pass

        if (self.root / "images").is_dir():
            return ("dataset", list(DEFAULT_CLASSES))
        return ("flat", list(DEFAULT_CLASSES))

    def _collect(self):
        # 可能的 (图片目录, 标签目录) 组合
        sources = []
        if self.fmt == "flat":
            sources.append((self.root, self.root))
        else:
            if (self.root / "images").is_dir():
                sources.append((self.root / "images", self.root / "labels"))
            else:
                # YOLO 分集布局: train/images + val/images
                for split in ("train", "val"):
                    if (self.root / split / "images").is_dir():
                        sources.append((self.root / split / "images", self.root / split / "labels"))
        pairs = []
        for img_dir, lbl_dir in sources:
            if img_dir.is_dir():
                for img in sorted(img_dir.iterdir()):
                    if img.suffix.lower() in IMAGE_EXTS:
                        stem = img.stem
                        pairs.append((img, lbl_dir / f"{stem}.txt", stem))
        return pairs

    # ---- 标签读写 ----
    def load_labels(self, lbl_path):
        boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        boxes.append(Box(parts[0], parts[1], parts[2], parts[3], parts[4]))
                    except (ValueError, IndexError):
                        continue
        return boxes

    def save_labels(self, lbl_path, boxes):
        if boxes:
            lbl_path.write_text("\n".join(b.to_line() for b in boxes) + "\n", encoding="utf-8")
        else:
            lbl_path.write_text("", encoding="utf-8")

    # ---- 审核状态持久化 ----
    def _load_state(self):
        p = self.root / STATE_FILE
        self.verdicts = {}
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data.get("verdicts"), dict):
                    # 新版: {"verdicts": {stem: "pass"/"fail"}}
                    self.verdicts = {k: v for k, v in data["verdicts"].items() if v in ("pass", "fail")}
                else:
                    # 旧版: {"reviewed": [stems]} → 全部视为"通过"
                    for stem in data.get("reviewed", []):
                        self.verdicts[stem] = "pass"
            except (ValueError, OSError):
                self.verdicts = {}

    def save_state(self):
        # 主状态文件
        data = {"verdicts": dict(self.verdicts)}
        (self.root / STATE_FILE).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        # 导出标记文件 (兼容 review_tracker.py)
        all_stems = {stem for _, _, stem in self.pairs}
        passed = sorted(s for s in all_stems if self.verdicts.get(s) == "pass")
        failed = sorted(s for s in all_stems if self.verdicts.get(s) == "fail")
        not_reviewed = sorted(s for s in all_stems if s not in self.verdicts)
        (self.root / ".reviewed_frames.txt").write_text("\n".join(passed), encoding="utf-8")
        (self.root / ".failed_frames.txt").write_text("\n".join(failed), encoding="utf-8")
        (self.root / ".not_reviewed.txt").write_text("\n".join(not_reviewed), encoding="utf-8")


class AnnotatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("篮球数据标注审核工具")
        self.root.geometry("1400x900")

        self.session = None
        self.index = 0
        self.total = 0

        self.img = None            # PIL 原图
        self._photo = None         # 当前显示的 PhotoImage (防 GC)
        self.boxes = []            # 当前帧标注框
        self.dirty = False         # 当前帧是否有未保存修改
        self.selected = None       # 选中的框下标

        self.zoom = 1.0
        self.ox = 0.0
        self.oy = 0.0
        self.fit_mode = True

        self.current_class = 0
        self.draw_mode = False
        self._draw_start = None    # 绘制起点 (图像像素)
        self._draw_cur = None
        self._drag_state = None    # move/resize 拖拽状态
        self.space_held = False    # 空格是否按住 (用于平移)
        self._panning = False      # 是否正在平移

        self._present = {}         # stem -> 该帧实际出现的类别 id 集合
        self._match_stems = []     # 当前过滤命中的帧 (缺失目标类)

        self._last_canvas_size = (0, 0)

        self._build_ui()
        self._show_placeholder("请点击「打开目录」选择审核目录，\n或运行 py annotator.py <目录>")

    # ---------- UI 构建 ----------
    def _build_ui(self):
        # 顶部工具栏
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="打开目录", command=self.ask_open).pack(side=tk.LEFT)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(bar, text="绘制类别:").pack(side=tk.LEFT)
        self.class_var = tk.StringVar()
        self.class_combo = ttk.Combobox(bar, textvariable=self.class_var,
                                        state="readonly", width=14)
        self.class_combo.pack(side=tk.LEFT, padx=4)
        self.class_combo.bind("<<ComboboxSelected>>", self._on_class_combo)

        self.draw_btn = ttk.Checkbutton(bar, text="绘制(w)", command=self.toggle_draw)
        self.draw_btn.pack(side=tk.LEFT, padx=6)
        ttk.Button(bar, text="删除(Del)", command=self.delete_selected).pack(side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(bar, text="◀ 上一张(←)", command=lambda: self.goto(self.index - 1)).pack(side=tk.LEFT)
        ttk.Button(bar, text="下一张(→) ▶", command=lambda: self.goto(self.index + 1)).pack(side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(bar, text="保存(Ctrl+S)", command=self.save_current).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="适应窗口(Ctrl+0)", command=self.fit_and_render).pack(side=tk.LEFT, padx=2)

        self.info_var = tk.StringVar(value="未打开目录")
        ttk.Label(bar, textvariable=self.info_var, foreground="#555").pack(side=tk.RIGHT)

        # 主体: 左画布 + 右侧栏
        body = ttk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side = ttk.Frame(body, width=280)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        # 进度区
        prog = ttk.LabelFrame(side, text="审核进度", padding=8)
        prog.pack(fill=tk.X, padx=6, pady=(6, 4))
        self.progress_var = tk.StringVar(value="—")
        ttk.Label(prog, textvariable=self.progress_var).pack(anchor=tk.W)
        self.progress_bar = ttk.Progressbar(prog, maximum=1.0, value=0.0)
        self.progress_bar.pack(fill=tk.X, pady=4)
        ttk.Button(prog, text="◀ 上一张未审核", command=self.prev_unreviewed).pack(fill=tk.X, pady=1)
        ttk.Button(prog, text="下一张未审核 ▶", command=self.next_unreviewed).pack(fill=tk.X, pady=1)

        # 跳转
        jump = ttk.LabelFrame(side, text="跳转", padding=8)
        jump.pack(fill=tk.X, padx=6, pady=4)
        self.jump_var = tk.StringVar()
        jentry = ttk.Entry(jump, textvariable=self.jump_var, width=10)
        jentry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        jentry.bind("<Return>", self._on_jump)
        ttk.Button(jump, text="跳转", command=self._on_jump).pack(side=tk.LEFT, padx=4)

        # 快速过滤 (缺失异常)
        filt = ttk.LabelFrame(side, text="快速过滤 (缺失检查)", padding=8)
        filt.pack(fill=tk.X, padx=6, pady=4)
        self.filter_var = tk.StringVar(value="无过滤")
        self.filter_combo = ttk.Combobox(filt, textvariable=self.filter_var, state="readonly")
        self.filter_combo["values"] = [label for label, _ in FILTER_RULES]
        self.filter_combo.current(0)
        self.filter_combo.pack(fill=tk.X)
        self.filter_combo.bind("<<ComboboxSelected>>", self._on_filter_change)
        self.filter_count_var = tk.StringVar(value="匹配: —")
        ttk.Label(filt, textvariable=self.filter_count_var, foreground="#555").pack(anchor=tk.W, pady=(4, 0))
        self.filter_progress_bar = ttk.Progressbar(filt, maximum=1.0, value=0.0)
        self.filter_progress_bar.pack(fill=tk.X, pady=(2, 0))
        self.filter_progress_var = tk.StringVar(value="")
        ttk.Label(filt, textvariable=self.filter_progress_var, foreground="#555").pack(anchor=tk.W)
        frow = ttk.Frame(filt)
        frow.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(frow, text="◀ 上一处([)", command=self.prev_match).pack(side=tk.LEFT)
        ttk.Button(frow, text="下一处(]) ▶", command=self.next_match).pack(side=tk.LEFT, padx=2)

        # 图例
        legend = ttk.LabelFrame(side, text="类别图例", padding=8)
        legend.pack(fill=tk.X, padx=6, pady=4)
        self.legend_frame = legend

        # 帧列表 (审核结果 + 四要素有无)
        ulist = ttk.LabelFrame(side, text="帧列表 (点击跳转)", padding=8)
        ulist.pack(fill=tk.BOTH, expand=True, padx=6, pady=(4, 6))
        cols = ("frame", "verdict", "player", "ball", "hoop", "board")
        self.frame_list = ttk.Treeview(ulist, columns=cols, show="headings", selectmode="browse")
        self.frame_list.heading("frame", text="帧")
        self.frame_list.heading("verdict", text="审")
        self.frame_list.heading("player", text="员")
        self.frame_list.heading("ball", text="球")
        self.frame_list.heading("hoop", text="筐")
        self.frame_list.heading("board", text="板")
        self.frame_list.column("frame", width=86, anchor=tk.W, stretch=False)
        self.frame_list.column("verdict", width=26, anchor=tk.CENTER, stretch=False)
        for c in ("player", "ball", "hoop", "board"):
            self.frame_list.column(c, width=22, anchor=tk.CENTER, stretch=False)
        self.frame_list.tag_configure("pass", foreground="#2e7d32")
        self.frame_list.tag_configure("fail", foreground="#c62828")
        self.frame_list.tag_configure("unreviewed", foreground="#999999")
        self.frame_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(ulist, orient=tk.VERTICAL, command=self.frame_list.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.frame_list.config(yscrollcommand=sb.set)
        self.frame_list.bind("<<TreeviewSelect>>", self._on_list_select)

        # 底部状态栏
        self.status_var = tk.StringVar(value="就绪")
        status = ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W,
                           relief=tk.SUNKEN, padding=(8, 3))
        status.pack(side=tk.BOTTOM, fill=tk.X)

        # 事件绑定
        self.canvas.bind("<ButtonPress-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Configure>", self._on_resize)
        self.root.bind("<KeyPress-space>", lambda e: self._set_space(True))
        self.root.bind("<KeyRelease-space>", lambda e: self._set_space(False))
        self.root.bind("<Key>", self._on_key)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- 目录 / 会话管理 ----------
    def ask_open(self):
        path = filedialog.askdirectory(title="选择审核目录")
        if path:
            self.open_directory(path)

    def open_directory(self, path):
        try:
            session = Session(path)
        except Exception as e:
            messagebox.showerror("打开失败", f"无法读取目录:\n{e}")
            return

        if not session.pairs:
            messagebox.showwarning("无图片", f"目录里没有找到图片:\n{session.root}")
            return

        self.session = session
        self.total = len(session.pairs)
        self.index = 0
        self.selected = None
        self.draw_mode = False
        self.draw_btn.state(["!selected"])

        # 更新类别下拉与图例
        self.class_combo["values"] = session.classes
        self.current_class = 0
        self.class_var.set(session.classes[0] if session.classes else "")
        self._rebuild_legend()

        self.root.title(f"篮球数据标注审核工具 — {session.root.name}")
        self.info_var.set(f"{session.root.name}  ·  {self.total} 帧  ·  {len(session.classes)} 类")

        # 预计算每帧实际出现的类别 (供缺失过滤/状态栏使用)
        self._present = {}
        for _, lbl, stem in session.pairs:
            self._present[stem] = {b.cls for b in session.load_labels(lbl)}
        self._match_stems = []
        self.filter_var.set("无过滤")
        self.filter_count_var.set("匹配: —")
        self.filter_progress_var.set("")
        self.filter_progress_bar["value"] = 0

        self._rebuild_list()

        self.fit_mode = True
        self.load_frame()
        self.canvas.focus_set()   # 焦点给画布, 空格/方向键直达

    def _rebuild_legend(self):
        for w in self.legend_frame.winfo_children():
            w.destroy()
        for i, name in enumerate(self.session.classes):
            row = ttk.Frame(self.legend_frame)
            row.pack(fill=tk.X, pady=1)
            tk.Canvas(row, width=16, height=16, bg=color_for(i),
                      highlightthickness=0).pack(side=tk.LEFT)
            ttk.Label(row, text=f" {i}  {name}").pack(side=tk.LEFT)

    # ---------- 帧加载 / 渲染 ----------
    def goto(self, idx):
        if self.session is None:
            return
        if idx < 0 or idx >= self.total:
            return
        if idx == self.index:
            return
        # 离开前自动保存
        if self.dirty:
            self.save_current()
        self.index = idx
        self.selected = None
        self.load_frame()

    def load_frame(self):
        img_path, lbl_path, stem = self.session.pairs[self.index]
        self.boxes = self.session.load_labels(lbl_path)
        self.dirty = False
        self.selected = None
        self._drag_state = None
        self._draw_start = self._draw_cur = None
        try:
            self.img = Image.open(img_path).convert("RGB")
        except OSError as e:
            self._show_placeholder(f"图片读取失败:\n{img_path}\n{e}")
            return
        self.fit_mode = True
        self.fit_to_window()
        self._sync_list_selection()
        self.render()

    def fit_to_window(self):
        if self.img is None:
            return
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        W, H = self.img.size
        self.zoom = clamp(min(cw / W, ch / H) * 0.98, 0.01, 20.0)
        self.ox = (cw - W * self.zoom) / 2.0
        self.oy = (ch - H * self.zoom) / 2.0

    def fit_and_render(self):
        self.fit_mode = True
        self.fit_to_window()
        self.render()

    def render(self):
        if self.img is None:
            return
        self.canvas.delete("all")
        W, H = self.img.size
        dw, dh = max(1, int(W * self.zoom)), max(1, int(H * self.zoom))
        disp = self.img if (dw == W and dh == H) else self.img.resize((dw, dh))
        self._photo = ImageTk.PhotoImage(disp)
        self.canvas.create_image(self.ox, self.oy, anchor=tk.NW, image=self._photo)
        self._draw_boxes(W, H)
        self.update_status()

    def _draw_boxes(self, W, H):
        for i, b in enumerate(self.boxes):
            x1, y1, x2, y2 = b.pixel_rect(W, H)
            X1, Y1 = self.to_canvas(x1, y1)
            X2, Y2 = self.to_canvas(x2, y2)
            col = color_for(b.cls)
            selected = (i == self.selected)
            if selected:
                # 选中框: 内部阴影填充(50%点阵) + 加粗边框 + 角点手柄
                self.canvas.create_rectangle(X1, Y1, X2, Y2, fill=col, stipple="gray50", outline="")
                self.canvas.create_rectangle(X1, Y1, X2, Y2, outline=col, width=4)
                for cx, cy in ((X1, Y1), (X2, Y1), (X1, Y2), (X2, Y2)):
                    self.canvas.create_rectangle(cx - 5, cy - 5, cx + 5, cy + 5,
                                                 fill=col, outline="#fff", width=1)
            else:
                self.canvas.create_rectangle(X1, Y1, X2, Y2, outline=col, width=2)
            # 标签
            name = self.session.classes[b.cls] if self.session and b.cls < len(self.session.classes) else str(b.cls)
            txt = self.canvas.create_text(X1 + 3, Y1 + 3, anchor=tk.NW, text=name,
                                          fill="#fff", font=("TkDefaultFont", 10, "bold"))
            bbox = self.canvas.bbox(txt)
            if bbox:
                self.canvas.create_rectangle(bbox, fill=col, outline="")
                self.canvas.tag_raise(txt)

        # 绘制中的橡皮筋框
        if self._draw_start is not None and self._draw_cur is not None:
            x1, y1 = self.to_canvas(*self._draw_start)
            x2, y2 = self.to_canvas(*self._draw_cur)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#FFD600",
                                         width=2, dash=(5, 3))

    # ---------- 坐标变换 ----------
    def to_image(self, cx, cy):
        return ((cx - self.ox) / self.zoom, (cy - self.oy) / self.zoom)

    def to_canvas(self, px, py):
        return (self.ox + px * self.zoom, self.oy + py * self.zoom)

    # ---------- 鼠标交互 ----------
    def _on_click(self, event):
        if self.img is None:
            return
        cx, cy = event.x, event.y

        if self.space_held:
            self._panning = True
            self._pan_last = (cx, cy)
            return

        # 无论是否绘制模式, 都先检测是否点中已有框: 点中则选中(可移动/缩放/删除)
        idx, mode, corner = self._hit_test(cx, cy)
        if idx is not None:
            self.selected = idx
            if mode == "resize":
                # 记录固定对角点 (图像像素)
                W, H = self.img.size
                x1, y1, x2, y2 = self.boxes[idx].pixel_rect(W, H)
                opposite = {"tl": (x2, y2), "tr": (x1, y2), "bl": (x2, y1), "br": (x1, y1)}
                self._drag_state = ("resize", idx, opposite[corner])
            else:
                W, H = self.img.size
                px, py = self.to_image(cx, cy)
                bx = self.boxes[idx].cx * W
                by = self.boxes[idx].cy * H
                self._drag_state = ("move", idx, (bx - px, by - py))
            self.render()
            return

        # 未点中框
        self.selected = None
        self._drag_state = None
        if self.draw_mode:
            self._draw_start = self._clamp_image(self.to_image(cx, cy))
            self._draw_cur = self._draw_start
        self.render()

    def _on_drag(self, event):
        if self.img is None:
            return

        if self._panning:
            dx = event.x - self._pan_last[0]
            dy = event.y - self._pan_last[1]
            self._pan_last = (event.x, event.y)
            self.ox += dx
            self.oy += dy
            self.fit_mode = False
            self.render()
            return

        W, H = self.img.size

        if self.draw_mode and self._draw_start is not None:
            self._draw_cur = self._clamp_image(self.to_image(event.x, event.y))
            self.render()
            return

        if self._drag_state is None:
            return
        kind, idx, arg = self._drag_state
        px, py = self._clamp_image(self.to_image(event.x, event.y))
        b = self.boxes[idx]

        if kind == "move":
            dx, dy = arg
            b.cx = clamp((px + dx) / W, 0.0, 1.0)
            b.cy = clamp((py + dy) / H, 0.0, 1.0)
        elif kind == "resize":
            fx, fy = arg  # 固定对角点
            x1, y1 = min(fx, px), min(fy, py)
            x2, y2 = max(fx, px), max(fy, py)
            b.cx = (x1 + x2) / 2.0 / W
            b.cy = (y1 + y2) / 2.0 / H
            b.w = (x2 - x1) / W
            b.h = (y2 - y1) / H

        self._changed()
        self.render()

    def _on_release(self, event):
        if self.img is None:
            return
        self._panning = False
        if self.draw_mode and self._draw_start is not None and self._draw_cur is not None:
            x1, y1 = self._draw_start
            x2, y2 = self._draw_cur
            W, H = self.img.size
            bx1, by1 = min(x1, x2), min(y1, y2)
            bx2, by2 = max(x1, x2), max(y1, y2)
            if (bx2 - bx1) >= MIN_BOX_PX and (by2 - by1) >= MIN_BOX_PX:
                box = Box(self.current_class,
                          (bx1 + bx2) / 2.0 / W, (by1 + by2) / 2.0 / H,
                          (bx2 - bx1) / W, (by2 - by1) / H)
                self.boxes.append(box)
                self.selected = len(self.boxes) - 1
                self._changed()
        self._draw_start = self._draw_cur = None
        self._drag_state = None
        self.render()

    def _on_motion(self, event):
        if self.img is not None:
            px, py = self.to_image(event.x, event.y)
            self.update_status(f"鼠标: ({int(px)}, {int(py)})")

    def _on_wheel(self, event):
        if self.img is None:
            return
        factor = 1.1 if event.delta > 0 else 0.9
        self.zoom_at(factor, event.x, event.y)

    def _set_space(self, held):
        self.space_held = held
        if not held:
            self._panning = False
        return "break"   # 阻止空格触发焦点控件的默认行为(翻页/激活按钮)

    def _on_resize(self, event):
        size = (event.width, event.height)
        if size != self._last_canvas_size:
            self._last_canvas_size = size
            if self.img is not None and self.fit_mode:
                self.fit_to_window()
                self.render()

    def zoom_at(self, factor, cx, cy):
        ix, iy = self.to_image(cx, cy)
        self.zoom = clamp(self.zoom * factor, 0.02, 20.0)
        self.ox = cx - ix * self.zoom
        self.oy = cy - iy * self.zoom
        self.fit_mode = False
        self.render()

    def _clamp_image(self, pt):
        W, H = self.img.size
        return (clamp(pt[0], 0, W), clamp(pt[1], 0, H))

    def _hit_test(self, cx, cy):
        W, H = self.img.size
        for idx in range(len(self.boxes) - 1, -1, -1):
            b = self.boxes[idx]
            x1, y1, x2, y2 = b.pixel_rect(W, H)
            X1, Y1 = self.to_canvas(x1, y1)
            X2, Y2 = self.to_canvas(x2, y2)
            corners = {"tl": (X1, Y1), "tr": (X2, Y1), "bl": (X1, Y2), "br": (X2, Y2)}
            for name, (px, py) in corners.items():
                if abs(cx - px) <= HANDLE_HIT and abs(cy - py) <= HANDLE_HIT:
                    return idx, "resize", name
            if X1 <= cx <= X2 and Y1 <= cy <= Y2:
                return idx, "move", None
        return None, None, None

    # ---------- 键盘 ----------
    def _on_key(self, event):
        key = event.keysym.lower()
        ctrl = (event.state & 0x0004) != 0

        # 审核核心键 D/F：无论焦点在哪个控件都响应（焦点落在类别下拉/
        # 跳转框上时也不能打断打标记，否则无法完成审核）
        if key == "d":
            self.mark_pass()
            return "break"
        if key == "f":
            self.mark_fail()
            return "break"

        # 输入框内不响应其余全局快捷键（避免翻帧/绘制/删除干扰文字输入）
        if isinstance(event.widget, (tk.Entry, ttk.Combobox, tk.Spinbox)):
            return

        if key == "right":
            self.goto(self.index + 1)
        elif key == "left":
            self.goto(self.index - 1)
        elif key == "w":
            self.toggle_draw()
        elif key == "delete" or key == "backspace":
            self.delete_selected()
        elif key == "escape":
            self.draw_mode = False
            self.draw_btn.state(["!selected"])
            self.selected = None
            self._draw_start = self._draw_cur = None
            self.render()
        elif key == "s" and ctrl:
            self.save_current()
        elif key == "0" and ctrl:
            self.fit_and_render()
        elif key in ("plus", "equal"):
            self.zoom_at(1.1, self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)
        elif key == "minus":
            self.zoom_at(0.9, self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)
        elif key == "bracketright":
            self.next_match()
        elif key == "bracketleft":
            self.prev_match()
        elif event.char.isdigit():
            cls = int(event.char)
            if self.session and cls < len(self.session.classes):
                self.set_current_class(cls)

    # ---------- 编辑操作 ----------
    def toggle_draw(self):
        if self.img is None:
            return
        self.draw_mode = not self.draw_mode
        if self.draw_mode:
            self.draw_btn.state(["selected"])
            self.selected = None
        else:
            self.draw_btn.state(["!selected"])
            self._draw_start = self._draw_cur = None
        self.render()

    def delete_selected(self):
        if self.selected is not None and 0 <= self.selected < len(self.boxes):
            del self.boxes[self.selected]
            self.selected = None
            self._changed()
            self.render()
        else:
            self.update_status("未选中框：请先用鼠标点选要删除的框")

    def set_current_class(self, cls):
        self.current_class = cls
        if self.session and cls < len(self.session.classes):
            self.class_var.set(self.session.classes[cls])

    def _on_class_combo(self, event=None):
        idx = self.class_combo.current()
        if idx >= 0:
            self.set_current_class(idx)

    def _on_right_click(self, event):
        """右键框: 弹出菜单改该框类别 (不影响绘制类别)"""
        if self.img is None or self.session is None:
            return
        idx, _, _ = self._hit_test(event.x, event.y)
        if idx is None:
            return
        menu = tk.Menu(self.root, tearoff=0)
        for i, name in enumerate(self.session.classes):
            menu.add_command(label=f"{i} · {name}", command=lambda i=i: self._relabel_box(idx, i))
        menu.tk_popup(event.x_root, event.y_root)

    def _relabel_box(self, idx, cls):
        if 0 <= idx < len(self.boxes):
            self.boxes[idx].cls = cls
            self.selected = idx
            self._changed()
            self.render()

    def _changed(self):
        """任意编辑后调用: 仅置脏标记, 审核结论由 D/F 显式标记"""
        self.dirty = True
        self.update_status()

    # ---------- 保存 ----------
    def save_current(self):
        if self.session is None:
            return
        if not self.dirty:
            self.update_status()
            return
        _, lbl_path, stem = self.session.pairs[self.index]
        self.session.save_labels(lbl_path, self.boxes)
        self._present[stem] = {b.cls for b in self.boxes}
        self.dirty = False
        self._update_current_row()
        # 若缺失过滤已激活, 刷新匹配计数
        if self.filter_var.get() != "无过滤":
            self._on_filter_change()
        self.update_status("已保存 ✓")

    # ---------- 审核进度 ----------
    def mark_pass(self):
        """D 键: 标记当前帧为「通过」"""
        self._mark("pass")

    def mark_fail(self):
        """F 键: 标记当前帧为「不通过」"""
        self._mark("fail")

    def _mark(self, verdict):
        if self.session is None:
            return
        self.session.verdicts[self.session.pairs[self.index][2]] = verdict
        self.session.save_state()
        self._update_current_row()
        self._update_progress()
        self._update_filter_progress()
        # 根据当前过滤类型决定标记后跳转位置
        rule = next((r for lbl, r in FILTER_RULES if lbl == self.filter_var.get()), None)
        if rule is None:
            self.goto(self.index + 1)
        elif rule[0] == "verdict":
            # 审核结果过滤: 标记会改变匹配集, 重算后跳到当前之后的下一处匹配
            self._on_filter_change()
            self._advance_to_next_match_forward()
        elif self._match_stems:
            self.next_match()
        else:
            self.goto(self.index + 1)

    def _unreviewed_stems(self):
        if self.session is None:
            return []
        return [s for _, _, s in self.session.pairs if s not in self.session.verdicts]

    def _rebuild_list(self):
        """全量重建帧列表: 每行 = 帧名 + 审核结果 + 四要素有无"""
        if not hasattr(self, "frame_list"):
            return
        self.frame_list.delete(*self.frame_list.get_children())
        if self.session is None:
            return
        nc = len(self.session.classes)
        for _, _, stem in self.session.pairs:
            v = self.session.verdicts.get(stem)
            verdict, vtag = ("✓", "pass") if v == "pass" else (("✗", "fail") if v == "fail" else ("·", "unreviewed"))
            present = self._present.get(stem, set())
            dots = ["●" if c in present else "○" for c in range(nc)]
            self.frame_list.insert("", tk.END, values=(stem, verdict) + tuple(dots), tags=(vtag,))
        self._sync_list_selection()

    def _update_current_row(self):
        """更新当前帧那一行的审核标记与四要素"""
        if self.session is None:
            return
        stem = self.session.pairs[self.index][2]
        v = self.session.verdicts.get(stem)
        verdict, vtag = ("✓", "pass") if v == "pass" else (("✗", "fail") if v == "fail" else ("·", "unreviewed"))
        present = self._present.get(stem, set())
        dots = ["●" if c in present else "○" for c in range(len(self.session.classes))]
        for iid in self.frame_list.get_children():
            if self.frame_list.set(iid, "frame") == stem:
                self.frame_list.item(iid, values=(stem, verdict) + tuple(dots), tags=(vtag,))
                break

    def _sync_list_selection(self):
        if self.session is None:
            return
        stem = self.session.pairs[self.index][2]
        for iid in self.frame_list.get_children():
            if self.frame_list.set(iid, "frame") == stem:
                self.frame_list.selection_set(iid)
                self.frame_list.see(iid)
                break

    def _on_list_select(self, event=None):
        sel = self.frame_list.selection()
        if not sel or self.session is None:
            return
        stem = self.frame_list.set(sel[0], "frame")
        for i, (_, _, s) in enumerate(self.session.pairs):
            if s == stem:
                self.goto(i)
                break

    def next_unreviewed(self):
        self._step_unreviewed(1)

    def prev_unreviewed(self):
        self._step_unreviewed(-1)

    def _step_unreviewed(self, direction):
        self._step_in(self._unreviewed_stems(), direction)

    def _step_in(self, stems, direction):
        """在给定 stem 列表中向前/向后跳转 (direction=±1)"""
        if self.session is None or not stems:
            return
        current = self.session.pairs[self.index][2]
        cur = stems.index(current) if current in stems else -1
        target = stems[(cur + direction) % len(stems)]
        for i, (_, _, s) in enumerate(self.session.pairs):
            if s == target:
                self.goto(i)
                break

    def _update_progress(self):
        if self.session is None:
            self.progress_var.set("—")
            self.progress_bar["value"] = 0
            return
        total = self.total
        passed = sum(1 for v in self.session.verdicts.values() if v == "pass")
        failed = sum(1 for v in self.session.verdicts.values() if v == "fail")
        done = passed + failed
        pct = done / total * 100 if total else 0
        self.progress_var.set(f"通过 {passed}  ·  不通过 {failed}  ·  剩余 {total - done}  ({pct:.1f}%)")
        self.progress_bar["value"] = done / total if total else 0

    # ---------- 快速过滤 (缺失异常) ----------
    def _on_filter_change(self, event=None):
        if not self.session:
            return
        label = self.filter_var.get()
        rule = next((r for lbl, r in FILTER_RULES if lbl == label), None)
        if rule is None:
            self._match_stems = []
            self.filter_count_var.set("匹配: —")
            self.filter_progress_var.set("")
            self.filter_progress_bar["value"] = 0
            return
        kind, arg = rule
        if kind == "missing":
            name_to_id = {name: i for i, name in enumerate(self.session.classes)}
            required = {name_to_id[n] for n in arg if n in name_to_id}
            self._match_stems = [s for _, _, s in self.session.pairs
                                 if not required <= self._present.get(s, set())]
        else:  # verdict
            if arg is None:
                self._match_stems = [s for _, _, s in self.session.pairs
                                     if s not in self.session.verdicts]
            else:
                self._match_stems = [s for _, _, s in self.session.pairs
                                     if self.session.verdicts.get(s) == arg]
        self.filter_count_var.set(f"匹配: {len(self._match_stems)} 帧")
        self._update_filter_progress()

    def next_match(self):
        self._step_in(self._match_stems, 1)

    def prev_match(self):
        self._step_in(self._match_stems, -1)

    def _advance_to_next_match_forward(self):
        """跳到当前帧之后(按帧顺序)的第一处匹配; 无则回到开头"""
        if not self._match_stems:
            self.goto(self.index + 1)
            return
        match_set = set(self._match_stems)
        for i in range(self.index + 1, self.total):
            if self.session.pairs[i][2] in match_set:
                self.goto(i)
                return
        for i in range(self.total):
            if self.session.pairs[i][2] in match_set:
                self.goto(i)
                return

    def _update_filter_progress(self):
        label = self.filter_var.get()
        rule = next((r for lbl, r in FILTER_RULES if lbl == label), None)
        # 进度条仅对「缺失检查」类过滤有意义; 审核结果类过滤只显示数量
        if not self._match_stems or (rule and rule[0] != "missing"):
            self.filter_progress_var.set("")
            self.filter_progress_bar["value"] = 0
            return
        total = len(self._match_stems)
        done = sum(1 for s in self._match_stems if s in self.session.verdicts)
        pct = done / total * 100 if total else 0
        self.filter_progress_var.set(f"已审 {done}/{total}  ({pct:.0f}%)")
        self.filter_progress_bar["value"] = done / total if total else 0

    def _on_jump(self, event=None):
        try:
            idx = int(self.jump_var.get().strip())
        except ValueError:
            return
        self.goto(idx)

    # ---------- 状态 / 杂项 ----------
    def update_status(self, extra=None):
        if self.session is None:
            self.status_var.set(extra or "就绪")
            return
        _, _, stem = self.session.pairs[self.index]
        v = self.session.verdicts.get(stem)
        mark = {"pass": "● 通过", "fail": "● 不通过"}.get(v, "○ 未审核")
        present = {b.cls for b in self.boxes}
        missing = [self.session.classes[c] for c in range(len(self.session.classes)) if c not in present]
        miss = f"  ⚠缺:{'/'.join(missing)}" if missing else ""
        dirty = " ●未保存" if self.dirty else ""
        cls = ""
        if self.selected is not None:
            c = self.boxes[self.selected].cls
            name = self.session.classes[c] if c < len(self.session.classes) else c
            cls = f"  选中:{name}"
        zoom = f"  缩放:{int(self.zoom * 100)}%"
        base = f"第 {self.index + 1}/{self.total} 张  {stem}  框:{len(self.boxes)}  {mark}{miss}{dirty}{cls}{zoom}"
        self.status_var.set(base + (f"  |  {extra}" if extra else ""))
        self._update_progress()

    def _show_placeholder(self, text):
        self.canvas.delete("all")
        self.canvas.create_text(400, 300, text=text, fill="#bbb",
                                font=("TkDefaultFont", 14), justify=tk.CENTER)

    def _on_close(self):
        if self.session is not None and self.dirty:
            self.save_current()
        if self.session is not None:
            self.session.save_state()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="篮球数据标注审核工具")
    parser.add_argument("directory", nargs="?", default=None, help="审核目录")
    args = parser.parse_args()

    root = tk.Tk()
    app = AnnotatorApp(root)
    if args.directory:
        app.open_directory(args.directory)
    else:
        app.ask_open()
    root.mainloop()


if __name__ == "__main__":
    main()
