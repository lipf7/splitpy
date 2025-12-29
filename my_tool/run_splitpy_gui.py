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

        self.title("Shear-wave Splitting Calculator")
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
                Param("Station DB", None, browse="file"),
                Param("Event data dir", "--event-datadir", browse="dir"),
                Param("Local event CSV", "--local-event", browse="file"),
                Param("Station keys", "--keys"),
                Param("Data format", "--data-format", "SAC",
                      choices=["SAC", "MSEED"]),
            ],

            "Event Settings": [
                Param("Start time (UTC)", "--start"),
                Param("End time (UTC)", "--end"),
                Param("Min magnitude", "--min-mag", "6.0"),
                Param("Max magnitude", "--max-mag"),
                Param("Reverse order", "--reverse", ptype="bool"),
            ],

            "Geometry": [
                Param("Phase", "--phase", "SKS",
                      choices=["SKS", "SKKS", "PKS"]),
                Param("Min distance (deg)", "--min-dist", "85"),
                Param("Max distance (deg)", "--max-dist", "120"),
            ],

            "Signal Processing": [
                Param("Filter bands (Hz) fmin-fmax,...", "--filter-bands", "0.02-0.2,0.05-0.5,0.1-1.0"),
                Param("Sampling rate (Hz)", "--sampling-rate", "10"),
                Param("Window (s)", "--window", "120"),
                Param("Min SNRQ", "--min-snr", "5"),
                Param("SNRT threshold", "--snrT", "1"),
                Param("Max delay (s)", "--max-delay", "4"),
                Param("DT delay (s)", "--dt-delay", "0.1"),
                Param("Dphi (deg)", "--dphi", "1"),
            ],

            "Control": [
                Param("Calc", "--calc", ptype="bool"),
                Param("Recalc", "--recalc", ptype="bool"),
                Param("Overwrite", "--overwrite", ptype="bool"),
                Param("Skip existing", "--skip-existing", ptype="bool"),
                Param("Verbose", "--verbose", ptype="bool"),
                Param("Diagnostic plot dir",
                      "--plot-diagnostic", browse="dir"),
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

        ttk.Label(left, text="Parameters",
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

        ttk.Label(right, text="Command Preview",
                  style="Title.TLabel").pack(anchor="w", pady=5)

        self.cmd = tk.Text(right, height=18,
                           font=("Courier", 10))
        self.cmd.pack(fill="both", expand=True)

        btns = ttk.Frame(right)
        btns.pack(pady=10)

        ttk.Button(btns, text="Build Command",
                   command=self.build_cmd).pack(fill="x", pady=4)
        ttk.Button(btns, text="Run",
                   command=self.run).pack(fill="x", pady=4)
        ttk.Button(btns, text="Quit",
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
            messagebox.showerror("Error", "Station DB is required")
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
        messagebox.showinfo("Running", "Calculation started.")


if __name__ == "__main__":
    SplitGUI().mainloop()

