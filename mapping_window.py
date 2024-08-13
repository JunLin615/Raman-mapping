# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:55:04 2024

@author: ljjjun
"""

import Pretreatment

import pandas as pd
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox


class SpectrumAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectrum Analyzer")

        self.label = tk.Label(root, text="Select a file and plot the spectrum and heatmap")
        self.label.pack(pady=10)

        self.select_button = tk.Button(root, text="Select File", command=self.load_file)
        self.select_button.pack(pady=5)

        self.wavelength_label = tk.Label(root, text="Enter target wavelength:")
        self.wavelength_label.pack(pady=5)

        self.wavelength_entry = tk.Entry(root)
        self.wavelength_entry.pack(pady=5)

        self.plot_button = tk.Button(root, text="Plot", command=self.plot_spectrum_and_heatmap)
        self.plot_button.pack(pady=5)

        self.file_path = ""

    def load_file(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            self.label.config(text=f"Selected file: {self.file_path}")

    def plot_spectrum_and_heatmap(self):
        if not self.file_path:
            self.label.config(text="No file selected!")
            return

        try:
            target_wavelength = float(self.wavelength_entry.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a valid number for the target wavelength.")
            return

        data = pd.read_csv(self.file_path, sep='\t')

        # 提取波长数据和光谱强度数据
        wavelengths = data.iloc[:, 0]
        spectral_data = data.iloc[:, 1:]

        # 提取列标签中的位置信息
        positions = []
        for col in tqdm(spectral_data.columns, desc="提取列标签信息"):
            label = col.split('(')[-1].split(')')[0]
            x, y = map(int, label.split('/'))
            positions.append((x, y))

        # 创建位置到列的映射
        position_to_col = dict(zip(positions, spectral_data.columns))

        # 查找特定波数或最近的波数
        closest_wavelength_idx = (np.abs(wavelengths - target_wavelength)).idxmin()

        # 获取唯一的x和y值
        x_vals = sorted(set([pos[0] for pos in positions]))
        y_vals = sorted(set([pos[1] for pos in positions]))

        # 创建映射字典
        x_map = {val: i for i, val in enumerate(x_vals)}
        y_map = {val: i for i, val in enumerate(y_vals)}

        # 创建空的热图
        hotmap = np.zeros((len(x_vals), len(y_vals)))

        # 填充热图
        for pos, col in tqdm(position_to_col.items(), desc="重整为热图"):
            x, y = pos
            hotmap[x_map[x], y_map[y]] = spectral_data[col].iloc[closest_wavelength_idx]

        # 找到在该波长处光谱强度最强的列
        max_intensity_col = spectral_data.iloc[closest_wavelength_idx].idxmax()
        print(max_intensity_col)

        # 绘制热图
        plt.figure()
        plt.imshow(hotmap.T, cmap='hot', interpolation='nearest', origin='lower')
        plt.colorbar(label='Intensity')
        plt.title(f'Hotmap for Wavelength {wavelengths.iloc[closest_wavelength_idx]} nm')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.show()

        # 绘制在特定波长处光谱强度最强的那列的光谱曲线图
        plt.figure()
        plt.plot(wavelengths, spectral_data[max_intensity_col])
        plt.title(f'Spectrum of the Column with Maximum Intensity at {wavelengths.iloc[closest_wavelength_idx]} cm-1')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.xlim(500, 1800)
        # 获取在 x 轴范围内的 y 数据
        x_limits = (500, 1800)
        mask = (wavelengths >= x_limits[0]) & (wavelengths <= x_limits[1])
        y_data_in_range = spectral_data[max_intensity_col][mask]

        # 自动缩放 y 轴
        plt.ylim(y_data_in_range.min(), y_data_in_range.max()*1.1)
        plt.show()


# 创建主窗口
root = tk.Tk()
app = SpectrumAnalyzer(root)
root.mainloop()
