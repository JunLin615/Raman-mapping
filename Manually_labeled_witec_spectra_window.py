# -*- coding: utf-8 -*-
"""
本文件功能：
打开一个witec采集的mapping拉曼光谱，然后去基线，然后人工一张张判断打标签。
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import os
import Pretreatment as pre

class SpectralLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectral Labeling App")

        self.input_file = ""
        self.output_dir = ""
        self.template_file = 'template_file.txt'

        # 输入文件选择按钮
        self.input_button = tk.Button(root, text="Select Input File", command=self.select_input_file)
        self.input_button.pack(pady=5)

        # 输出目录选择按钮
        self.output_button = tk.Button(root, text="Select Output Directory", command=self.select_output_directory)
        self.output_button.pack(pady=5)

        # 标记列表输入框
        self.mark_list_labels = []
        self.mark_list_entries = []
        for i in range(5):
            label = tk.Label(root, text=f"Enter mark {i+1}:")
            label.pack(pady=5)
            entry = tk.Entry(root)
            entry.pack(pady=5)
            self.mark_list_labels.append(label)
            self.mark_list_entries.append(entry)

        # 处理按钮
        self.process_button = tk.Button(root, text="Process", command=self.process)
        self.process_button.pack(pady=20)

    def select_input_file(self):
        self.input_file = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if self.input_file:
            print(f"Selected input file: {self.input_file}")

    def select_output_directory(self):
        self.output_dir = filedialog.askdirectory()
        if self.output_dir:
            print(f"Selected output directory: {self.output_dir}")

    def process(self):
        if not self.input_file or not self.output_dir:
            messagebox.showerror("Error", "Please select both input file and output directory.")
            return

        try:
            mark_list = [float(entry.get()) for entry in self.mark_list_entries]
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid values for all marks.")
            return

        processor = pre.WitecRamanProcessor(self.input_file, self.output_dir)
        wavelengths, spectral_data = processor.read_data(self.input_file, delimiter='\t')
        file_name0 = os.path.basename(self.input_file)
        processor.Denoise = False

        # 打开光谱标记应用
        root = tk.Tk()
        app = pre.SpectralLabelingApp(root, spectral_data, self.template_file, self.output_dir, wavelengths, mark_list, file_name0, start_wavelength=500, end_wavelength=1800)
        root.mainloop()

# 创建主窗口
if __name__ == "__main__":
    root = tk.Tk()
    app = SpectralLabelingApp(root)
    root.mainloop()
