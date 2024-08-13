# -*- coding: utf-8 -*-
"""
本文件功能：
将指定文件夹的witec的mapping光谱去基线，保存到另一个文件夹。
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import Pretreatment as pre

class BaselineCorrectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Baseline Correction")

        self.input_dir = ""
        self.output_dir = ""

        # 输入目录选择按钮
        self.input_button = tk.Button(root, text="Select Input Directory", command=self.select_input_directory)
        self.input_button.pack(pady=5)

        # 输出目录选择按钮
        self.output_button = tk.Button(root, text="Select Output Directory", command=self.select_output_directory)
        self.output_button.pack(pady=5)

        # 输入平滑参数 lam
        self.lam_label = tk.Label(root, text="Enter lam (smoothing parameter 1000):")
        self.lam_label.pack(pady=5)
        self.lam_entry = tk.Entry(root)
        self.lam_entry.pack(pady=5)

        # 输入权重参数 p
        self.p_label = tk.Label(root, text="Enter p (weight parameter 0.01):")
        self.p_label.pack(pady=5)
        self.p_entry = tk.Entry(root)
        self.p_entry.pack(pady=5)

        # 输入循环次数 niter
        self.niter_label = tk.Label(root, text="Enter niter (number of iterations 3):")
        self.niter_label.pack(pady=5)
        self.niter_entry = tk.Entry(root)
        self.niter_entry.pack(pady=5)

        # 处理按钮
        self.process_button = tk.Button(root, text="Process", command=self.process)
        self.process_button.pack(pady=20)

    def select_input_directory(self):
        self.input_dir = filedialog.askdirectory()
        if self.input_dir:
            print(f"Selected input directory: {self.input_dir}")

    def select_output_directory(self):
        self.output_dir = filedialog.askdirectory()
        if self.output_dir:
            print(f"Selected output directory: {self.output_dir}")

    def process(self):
        if not self.input_dir or not self.output_dir:
            messagebox.showerror("Error", "Please select both input and output directories.")
            return

        try:
            lam = float(self.lam_entry.get())
            p = float(self.p_entry.get())
            niter = int(self.niter_entry.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid values for lam, p, and niter.")
            return

        processor = pre.WitecRamanProcessor(self.input_dir, self.output_dir)
        processor.lam = lam
        processor.p = p
        processor.niter = niter
        processor.Denoise = False

        # 开始处理
        processor.process_directory_reBaseLine()
        messagebox.showinfo("Success", "Processing complete!")

# 创建主窗口
root = tk.Tk()
app = BaselineCorrectionApp(root)
root.mainloop()
