import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import torch
import os
import Pretreatment as pr
from CNN_classification_train import Net, MyDataset
import CNN_recognition_func as CNNR
class SpectralProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN Spectral Processing Tool")
        self.model_path = ""
        self.output_directory = ""
        self.data_path = ""
        self.target_wavelength = 0

        self.setup_ui()

    def setup_ui(self):
        # 创建并放置标签和输入框
        tk.Label(self.root, text="Model Path:").grid(row=0, column=0, padx=10, pady=10)
        self.model_entry = tk.Entry(self.root, width=50)
        self.model_entry.grid(row=0, column=1, padx=10, pady=10)
        tk.Button(self.root, text="Browse", command=self.select_model).grid(row=0, column=2, padx=10, pady=10)

        tk.Label(self.root, text="Output Directory:").grid(row=1, column=0, padx=10, pady=10)
        self.output_entry = tk.Entry(self.root, width=50)
        self.output_entry.grid(row=1, column=1, padx=10, pady=10)
        tk.Button(self.root, text="Browse", command=self.select_output_directory).grid(row=1, column=2, padx=10, pady=10)

        tk.Label(self.root, text="Data Path:").grid(row=2, column=0, padx=10, pady=10)
        self.data_entry = tk.Entry(self.root, width=50)
        self.data_entry.grid(row=2, column=1, padx=10, pady=10)
        tk.Button(self.root, text="Browse", command=self.select_data_file).grid(row=2, column=2, padx=10, pady=10)

        tk.Label(self.root, text="Target Wavelength:").grid(row=3, column=0, padx=10, pady=10)
        self.target_wavelength_entry = tk.Entry(self.root, width=50)
        self.target_wavelength_entry.grid(row=3, column=1, padx=10, pady=10)

        # 创建并放置处理数据的按钮
        tk.Button(self.root, text="Process Data", command=self.process_data).grid(row=4, column=1, padx=10, pady=20)

    def select_model(self):
        self.model_path = filedialog.askopenfilename(title="Select Model File")
        self.model_entry.delete(0, tk.END)
        self.model_entry.insert(0, self.model_path)

    def select_output_directory(self):
        self.output_directory = filedialog.askdirectory(title="Select Output Directory")
        self.output_entry.delete(0, tk.END)
        self.output_entry.insert(0, self.output_directory)

    def select_data_file(self):
        self.data_path = filedialog.askopenfilename(title="Select Data File")
        self.data_entry.delete(0, tk.END)
        self.data_entry.insert(0, self.data_path)

    def process_data(self):
        self.target_wavelength = float(self.target_wavelength_entry.get())
        model = CNNR.load_model(self.model_path)
        processor = pr.WitecRamanProcessor(self.data_entry.get(), self.output_entry.get())
        wavelengths, spectral_data = processor.read_data(self.data_path, delimiter='\t')
        data_run = MyDataset(self.model_entry.get(), 'Gramian_angular', [500, 1800])

        results_df, good_list, good_list_xy = CNNR.CNN_reg(wavelengths, spectral_data, data_run, model)
        #print(wavelengths)
        #print(self.target_wavelength)

        hotmap_raw, hotmap_p, hotmap_0, hotmap_1 = CNNR.make_hotmap(wavelengths, spectral_data, self.target_wavelength, results_df)
        total_elements = hotmap_p.size

        # 统计列表中子元素等于1的总数（所有二维矩阵中等于1的元素个数相加）
        count_of_ones = np.sum(hotmap_p == 1)
        p = count_of_ones/total_elements*100

        self.p_info = f'Y{count_of_ones}N{total_elements}P{p:.2f} percent'

        self.save_and_plot_results(hotmap_raw, hotmap_p, hotmap_0, hotmap_1, wavelengths, good_list, spectral_data, processor)
    def save_and_plot_results(self, hotmap_raw, hotmap_p, hotmap_0, hotmap_1, wavelengths, good_list, spectral_data, processor):
        sum_good = np.sum(good_list, axis=0)
        ave_good_good = sum_good / len(good_list)
        ave_good_all = sum_good / spectral_data.shape[1]

        save_ave_good_good = np.column_stack((wavelengths.to_numpy(), ave_good_good))
        save_ave_good_all = np.column_stack((wavelengths.to_numpy(), ave_good_all))
        concentration = processor.parse_filename(os.path.basename(self.data_path))
        raw_rb = spectral_data.mean(axis=1)
        save_raw_rb = np.column_stack((wavelengths.to_numpy(), raw_rb.to_numpy()))

        os.makedirs(self.output_directory + f'/mapping/{concentration[2]}', exist_ok=True)
        plt.figure()
        plt.imshow(hotmap_raw.T, cmap='hot', interpolation='nearest', origin='lower')
        plt.colorbar(label='Intensity')
        plt.title(f'raw{self.target_wavelength}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig(self.output_directory + f'/mapping/{concentration[2]}/' + f'mapping_raw_{self.p_info}.png')

        plt.figure()
        plt.imshow(hotmap_p.T, cmap='hot', interpolation='nearest', origin='lower', vmin=0, vmax=1)
        plt.colorbar(label='Intensity')
        plt.title(f'predicted')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig(self.output_directory + f'/mapping/{concentration[2]}/' + f'mapping_p_{self.p_info}.png')

        plt.figure()
        plt.imshow(hotmap_0.T, cmap='hot', interpolation='nearest', origin='lower', vmin=0, vmax=np.max(hotmap_0))
        plt.colorbar(label='Intensity')
        plt.title(f'0')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig(self.output_directory + f'/mapping/{concentration[2]}/' + f'mapping_0_{self.p_info}.png')

        plt.figure()
        plt.imshow(hotmap_1.T, cmap='hot', interpolation='nearest', origin='lower', vmin=0, vmax=np.max(hotmap_1))
        plt.colorbar(label='Intensity')
        plt.title(f'1')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig(self.output_directory + f'/mapping/{concentration[2]}/' + f'mapping_1_{self.p_info}.png')

        os.makedirs(self.output_directory + f'/raw_rb_allmean/', exist_ok=True)
        np.savetxt(self.output_directory + f'/raw_rb_allmean/{concentration[2]}.txt', save_raw_rb, fmt='%.10f', delimiter=',')
        os.makedirs(self.output_directory + f'/ave_good_good/', exist_ok=True)
        np.savetxt(self.output_directory + f'/ave_good_good/{concentration[2]}.txt', save_ave_good_good, fmt='%.10f', delimiter=',')
        os.makedirs(self.output_directory + f'/ave_good_all/', exist_ok=True)
        np.savetxt(self.output_directory + f'/ave_good_all/{concentration[2]}.txt', save_ave_good_all, fmt='%.10f', delimiter=',')


if __name__ == '__main__':
    root = tk.Tk()
    app = SpectralProcessingApp(root)
    root.mainloop()
