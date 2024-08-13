import pandas as pd
import os
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog

class DataProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Processor")
        self.file_path = ""
        self.output_dir = ""

        self.setup_ui()

    def setup_ui(self):
        # 创建并放置标签和输入框
        tk.Label(self.root, text="Input Directory:").grid(row=0, column=0, padx=10, pady=10)
        self.input_entry = tk.Entry(self.root, width=50)
        self.input_entry.grid(row=0, column=1, padx=10, pady=10)
        tk.Button(self.root, text="Browse", command=self.select_input_directory).grid(row=0, column=2, padx=10, pady=10)

        tk.Label(self.root, text="Output Directory:").grid(row=1, column=0, padx=10, pady=10)
        self.output_entry = tk.Entry(self.root, width=50)
        self.output_entry.grid(row=1, column=1, padx=10, pady=10)
        tk.Button(self.root, text="Browse", command=self.select_output_directory).grid(row=1, column=2, padx=10, pady=10)

        # 创建并放置处理数据的按钮
        tk.Button(self.root, text="Process Data", command=self.process_data).grid(row=2, column=1, padx=10, pady=20)

    def select_input_directory(self):
        self.file_path = filedialog.askdirectory(title="Select Input Directory")
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, self.file_path)

    def select_output_directory(self):
        self.output_dir = filedialog.askdirectory(title="Select Output Directory")
        self.output_entry.delete(0, tk.END)
        self.output_entry.insert(0, self.output_dir)

    def process_data(self):
        input_dir = self.input_entry.get()
        output_dir = self.output_entry.get()
        self.process_directory_reBaseLine(input_dir, output_dir)

    def read_data(self, file_path, delimiter='\t'):
        """
        读取高光谱数据。
        """
        data = pd.read_csv(file_path, delimiter=delimiter)
        wavelengths = data.iloc[:, 0]
        spectral_data = data.iloc[:, 1:]
        return wavelengths, spectral_data

    def smart2witec(self, file_path, output_dir, delimiter='\t'):
        df_b = pd.read_csv(file_path, delimiter=delimiter)
        df_b.rename(columns={'Unnamed: 0': 'x', 'Unnamed: 1': 'y'}, inplace=True)

        # 将B格式转换为A格式
        wavelengths = df_b.columns[2:]

        # 初始化存储列的字典
        columns_dict = {'X-Axis': wavelengths}
        for i in range(df_b.shape[0]):
            columns_dict[f'a-5({df_b["x"][i]}/{df_b["y"][i]})'] = df_b.iloc[i, 2:].values
        df_a = pd.DataFrame(columns_dict)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0])

        # 保存为CSV文件
        df_a.to_csv(output_path + '.txt', sep='\t', index=False)

    def process_directory_reBaseLine(self, input_dir, output_dir):
        """
        批量处理文件夹中的所有文件，目的是去基线并保存。
        """
        files = [file for file in os.listdir(input_dir) if file.endswith('.txt')]
        for file_name in tqdm(files, desc="Processing directory", position=0):
            file_path = os.path.join(input_dir, file_name)
            self.smart2witec(file_path, output_dir, delimiter='\t')

if __name__ == '__main__':
    root = tk.Tk()
    app = DataProcessorApp(root)
    root.mainloop()
