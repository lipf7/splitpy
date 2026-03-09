#!/home/lpf/anaconda3/envs/split/bin/python3.12
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import shlex
import sys

SPLIT_COMMAND = "split_calc_auto"


class Param:
    def __init__(self, label, flag, default="", ptype="str",
                 choices=None, browse=None):
        self.label = label
        self.flag = flag
        self.default = default
        self.ptype = ptype
        self.choices = choices
        self.browse = browse
        self.var = None


class SplitGUI(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Shear-wave Splitting Calculator (剪切波分裂计算器)")
        self.geometry("1050x720")

        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        self.style.configure("Title.TLabel",
                             font=("Helvetica", 14, "bold"))
        self.style.configure("Section.TLabelframe",
                             padding=(10, 6))
        self.style.configure("Section.TLabelframe.Label",
                             font=("Helvetica", 11, "bold"))

        self.define_params()
        self.build_layout()

    # ------------------------------------------------
    def define_params(self):

        self.groups = {

            "Station & Data": [
                Param("Station .pkl", None, browse="file"),  # 台站数据库文件（.pkl格式）
                Param("SDS data dir", "--event-datadir", browse="dir"),  # 事件数据目录（SDS格式）
                Param("Local event list (.CSV)", "--local-event", browse="file"),  # 本地事件列表（CSV格式）
                Param("Station keys", "--keys"),  # 台站键值（逗号分隔）
                Param("Data format", "--data-format", "SAC",
                      choices=["SAC", "MSEED"]),  # 数据格式（SAC或MSEED）
            ],

            "Event Settings": [
                Param("Start time (UTC)", "--start"),  # 事件搜索开始时间（UTC）
                Param("End time (UTC)", "--end"),  # 事件搜索结束时间（UTC）
                Param("Min magnitude", "--min-mag", "6.0"),  # 最小震级
                Param("Max magnitude", "--max-mag"),  # 最大震级
                Param("Reverse order", "--reverse", ptype="bool"),  # 反转事件顺序
            ],

            "Geometry": [
                Param("Phase", "--phase", "SKS",
                      choices=["SKS", "SKKS", "PKS"]),  # 相位名称（SKS、SKKS或PKS）
                Param("Min distance (deg)", "--min-dist", "85"),  # 最小距离（度）
                Param("Max distance (deg)", "--max-dist", "120"),  # 最大距离（度）
            ],

            "Signal Processing": [
                Param("Filter bands (Hz) (fmin-fmax,...)", "--filter-bands", "0.02-0.2,0.05-0.5,0.1-1.0"),  # 滤波频带（Hz），格式：fmin-fmax,fmin-fmax,...
                Param("Sampling rate (Hz)", "--sampling-rate", "10"),  # 采样率（Hz）
                Param("Window (s)", "--window", "120"),  # 窗口长度（秒）
                Param("Min SNRQ", "--min-snr", "4"),  # 最小径向分量SNR（用于参考阈值）
                Param("SNRT threshold", "--snrT", "1"),  # 横向分量SNR阈值（用于空解判断）和Q值计算控制
                Param("Max delay (s)", "--max-delay", "4"),  # 最大延迟时间（秒）
                Param("DT delay (s)", "--dt-delay", "0.1"),  # 延迟时间增量（秒）
                Param("Dphi (deg)", "--dphi", "1"),  # 快速轴角度增量（度）
            ],

            "Control": [
                Param("Calc", "--calc", ptype="bool"),  # 执行分裂分析
                Param("Recalc", "--recalc", ptype="bool"),  # 重新计算（不重新下载数据）
                Param("Overwrite", "--overwrite", ptype="bool"),  # 覆盖现有结果
                Param("Skip existing", "--skip-existing", ptype="bool"),  # 跳过已有结果的事件
                Param("Verbose", "--verbose", ptype="bool"),  # 详细输出
                Param("Diagnostic plot dir",
                      "--plot-diagnostic", browse="dir"),  # 诊断图保存目录
            ],

            "Advanced Settings": [
                Param("Shift sec for window search", "--shift-sec", "5.0"),  # 窗口搜索偏移秒数
                Param("Step size for window search", "--search-step", "1.0"),  # 窗口搜索步长
            ],
        }

    # ------------------------------------------------
    def build_layout(self):

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        right = ttk.Frame(main, width=350)
        right.pack(side="right", fill="y", padx=10, pady=10)

        ttk.Label(left, text="Parameters (参数)",
                  style="Title.TLabel").pack(anchor="w", pady=5)

        canvas = tk.Canvas(left, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left, orient="vertical",
                                  command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for gname, params in self.groups.items():
            box = ttk.Labelframe(scroll_frame,
                                 text=gname,
                                 style="Section.TLabelframe")
            box.pack(fill="x", padx=5, pady=6)

            for p in params:
                row = ttk.Frame(box)
                row.pack(fill="x", pady=2)

                ttk.Label(row, text=p.label,
                          width=24).pack(side="left")

                if p.ptype == "bool":
                    p.var = tk.BooleanVar()
                    ttk.Checkbutton(row, variable=p.var).pack(side="left")
                else:
                    p.var = tk.StringVar(value=p.default)
                    if p.choices:
                        ttk.OptionMenu(row, p.var,
                                       p.default, *p.choices).pack(
                                           side="left", fill="x", expand=True)
                    else:
                        ttk.Entry(row, textvariable=p.var,
                                  width=40).pack(side="left", fill="x", expand=True)

                    if p.browse:
                        ttk.Button(row, text="Browse",
                                   command=lambda v=p.var, b=p.browse:
                                   self.browse(v, b)).pack(side="left", padx=4)

        ttk.Label(right, text="Command Preview (命令预览)",
                  style="Title.TLabel").pack(anchor="w", pady=5)

        self.cmd = tk.Text(right, height=18,
                           font=("Courier", 10))
        self.cmd.pack(fill="both", expand=True)

        btns = ttk.Frame(right)
        btns.pack(pady=10)

        ttk.Button(btns, text="Build Command (构建命令)",
                   command=self.build_cmd).pack(fill="x", pady=4)
        ttk.Button(btns, text="Run (运行)",
                   command=self.run).pack(fill="x", pady=4)
        ttk.Button(btns, text="Quit (退出)",
                   command=self.quit).pack(fill="x", pady=4)

    # ------------------------------------------------
    def browse(self, var, mode):
        if mode == "file":
            path = filedialog.askopenfilename()
        else:
            path = filedialog.askdirectory()
        if path:
            var.set(path)

    # ------------------------------------------------
    def build_cmd(self):

        cmd = [SPLIT_COMMAND]

        indb = self.groups["Station & Data"][0].var.get()
        if not indb:
            messagebox.showerror("Error (错误)", "Station DB is required (需要台站数据库)")
            return
        cmd.append(indb)

        for params in self.groups.values():
            for p in params:
                if p.flag is None:
                    continue
                if p.ptype == "bool":
                    if p.var.get():
                        cmd.append(p.flag)
                else:
                    val = p.var.get().strip()
                    if val:
                        cmd += [p.flag, val]

        self.cmd.delete("1.0", tk.END)
        self.cmd.insert(tk.END,
                        " ".join(shlex.quote(c) for c in cmd))

    # ------------------------------------------------
    def run(self):
        command = self.cmd.get("1.0", tk.END).strip()
        if not command:
            return
        subprocess.Popen(command, shell=True)
        messagebox.showinfo("Running (运行中)", "Calculation started. (计算已开始)")


if __name__ == "__main__":
    SplitGUI().mainloop()

