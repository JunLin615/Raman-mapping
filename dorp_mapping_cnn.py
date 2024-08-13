# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:55:04 2024

@author: ljjjun
"""

import Pretreatment as pr
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from CNN_classification_train import Net,  MyDataset
import CNN_recognition_func as CNNR
class SpectrumAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectrum Analyzer")

        self.label = tk.Label(root, text="Select a file and plot the spectrum and heatmap")
        self.label.pack(pady=10)

        self.select_button = tk.Button(root, text="Select data folder", command=self.load_file)
        self.select_button.pack(pady=5)

        self.select_CNN_button = tk.Button(root, text="Select CNN model", command=self.load_cnn)
        self.select_CNN_button.pack(pady=5)

        self.select_output_directory_button = tk.Button(root, text="Select output_directory", command=self.load_output_directory)
        self.select_output_directory_button.pack(pady=5)

        self.wavelength_label = tk.Label(root, text="Enter target wavelength:")
        self.wavelength_label.pack(pady=5)

        self.wavelength_entry = tk.Entry(root)
        self.wavelength_entry.pack(pady=5)

        self.plot_button = tk.Button(root, text="Plot", command=self.process_directory)
        self.plot_button.pack(pady=5)

        self.file_path = ""
    def load_file(self):
        self.file_path = filedialog.askdirectory()
        if self.file_path:
            self.label.config(text=f"Selected folder: {self.file_path}")
    def load_output_directory(self):
        self.output_directory = filedialog.askdirectory()
        if self.output_directory:
            self.label.config(text=f"Selected output directory: {self.output_directory}")
    def load_cnn(self):
        self.cnn_filename = filedialog.askopenfilename()
        if self.cnn_filename:
            self.label.config(text=f"Selected CNN model: {self.cnn_filename}")
        pass

    def hotmaps_func(self,hotmaps,m_name,concentration):
        total_maps = 100
        if len(hotmaps)<=100:
            filled_hotmaps = hotmaps + [np.zeros((8, 8)) for _ in range(total_maps - len(hotmaps))]
        else:
            filled_hotmaps = hotmaps[0:100]
        # 创建图像和子图
        fig, axs = plt.subplots(10, 10, figsize=(15, 15))

        # 设定所有子图的范围
        vmin = np.min([np.min(hm) for hm in hotmaps])
        vmax = np.max([np.max(hm) for hm in hotmaps])

        # 绘制每个hotmap到子图中
        for i in range(10):
            for j in range(10):
                im = axs[i, j].imshow(filled_hotmaps[i * 10 + j], vmin=vmin, vmax=vmax, cmap='hot',origin='lower')
                axs[i, j].axis('off')  # 隐藏坐标轴

        # 添加一个共享的colorbar
        cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.025, pad=0.02)
        cbar.set_label('Color Scale')

        # 添加标题
        plt.suptitle(f'{m_name} Hotmaps', fontsize=16)

        out_subname = self.output_directory + f'/mapping/{concentration[2]}'

        os.makedirs(out_subname, exist_ok=True)
        plt.savefig(out_subname + f'/mapping_{m_name}_{self.p_info}.png')

        # 显示图像
        #plt.show()
    def process_directory(self):
        """
        批量处理文件夹中的所有文件。
        """
        self.processor = pr.WitecRamanProcessor('data/', 'data/')
        self.model = CNNR.load_model(self.cnn_filename)
        files = [file for file in os.listdir(self.file_path) if file.endswith('.txt')]
        try:
            self.target_wavelength = float(self.wavelength_entry.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a valid number for the target wavelength.")
            return

        hotmaps = [[],[],[],[]]

        m_name = [f'raw_{self.target_wavelength}','p','1','0']
        #hotmaps_raw, hotmaps_p, hotmaps_0, hotmaps_1 = [],[],[],[]
        file_names = []
        for file_name in tqdm(files, desc="Processing directory", position=0):
            file_path = os.path.join(self.file_path, file_name)
            file_names.append(file_name)
            hotmap = self.plot_spectrum_and_heatmap(file_path)
            for i in range(4):
                hotmaps[i].append(hotmap[i])

        concentration = self.processor.parse_filename(os.path.basename(file_name))

        # 统计列表中一共有多少子元素（每个二维矩阵元素数相加）
        total_elements = sum(matrix.size for matrix in hotmaps[1])

        # 统计列表中子元素等于1的总数（所有二维矩阵中等于1的元素个数相加）
        count_of_ones = sum(np.sum(matrix == 1) for matrix in hotmaps[1])
        p = count_of_ones/total_elements*100

        self.p_info = f'Y{count_of_ones}N{total_elements}P{p:.2f} percent'


        for i in range(4):
            self.hotmaps_func(hotmaps[i], m_name[i], concentration)

        #print(file_names[5])



    def plot_spectrum_and_heatmap(self,file_path):
        if not self.file_path:
            self.label.config(text="No file selected!")
            return

        data_run = MyDataset(file_path, 'Gramian_angular', [500, 1800])

        wavelengths, spectral_data = self.processor.read_data(file_path, delimiter='\t')

        # 新建一个空的 DataFrame 用于存储计算结果，只有一行
        #results_df = pd.DataFrame(index=['predicted', '0', '1'], columns=spectral_data.columns)

        results_df, good_list, good_list_xy = CNNR.CNN_reg(wavelengths, spectral_data, data_run, self.model)
        # 创建mapping图像
        hotmap_raw, hotmap_p, hotmap_0, hotmap_1 = CNNR.make_hotmap(wavelengths, spectral_data, self.target_wavelength, results_df)



        return hotmap_raw, hotmap_p, hotmap_1, hotmap_0



# 创建主窗口
root = tk.Tk()
app = SpectrumAnalyzer(root)
root.mainloop()
