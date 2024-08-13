import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from tkinter import Tk, filedialog
import os
import Pretreatment as pre

"""
功能：

测试去基线用什么参数合适
也可以用来看mapping中指定xy光谱的形状
有交互窗口
"""

def update(val):
    global data_column, baseline, r_baseline, x1, y1, lam, p, niter

    x1 = int(text_x.text)
    y1 = int(text_y.text)
    lam = float(text_lam.text)
    p = float(text_p.text)
    niter = int(text_niter.text)

    processor = pre.WitecRamanProcessor(input_directory, output_directory)
    processor.lam = lam
    processor.p = p
    processor.niter = niter
    processor.Denoise = False

    data_column = processor.find_closest_point_index(spectral_data, x1, y1)
    baseline, r_baseline = processor.baseline_als(data_column)

    ax.clear()
    ax.plot(wavelengths, data_column, label='Original Data')
    ax.plot(wavelengths, baseline, label='Baseline')
    ax.plot(wavelengths, r_baseline, label='Corrected Data')
    ax.set_xlabel('Raman shift(cm-1)')
    ax.set_ylabel('I(a.u)')
    ax.set_title(f'lam{lam}, p{p}, niter{niter}, x{x1}, y{y1}')
    ax.set_xlim(500, 1800)
    ax.legend()
    fig.canvas.draw_idle()


def save(event):
    root = Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    root.update()
    root.destroy()

    if file_path:
        fig2, ax2 = plt.subplots()
        ax2.plot(wavelengths, data_column, label='Original Data')
        ax2.plot(wavelengths, baseline, label='Baseline')
        ax2.plot(wavelengths, r_baseline, label='Corrected Data')
        ax2.set_xlabel('Raman shift(cm-1)')
        ax2.set_ylabel('I(a.u)')
        ax2.set_title(f'lam{lam}, p{p}, niter{niter}, x{x1}, y{y1}')
        ax2.set_xlim(500, 1800)
        ax2.legend()

        fig2.savefig(file_path, bbox_inches='tight')
        plt.close(fig2)


def load_file(event):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    root.update()
    root.destroy()

    if file_path:
        global data_path, wavelengths, spectral_data, processor
        data_path = file_path
        processor = pre.WitecRamanProcessor(input_directory, output_directory)
        wavelengths, spectral_data = processor.read_data(data_path, delimiter='\t')
        update(None)


if __name__ == "__main__":
    input_directory = 'data'
    output_directory = 'data'
    data_path = 'data'

    x1, y1 = 0, 0
    lam, p, niter = 1e3, 0.01, 3

    #processor = pre.WitecRamanProcessor(input_directory, output_directory)
    #processor.Denoise = False
    #wavelengths, spectral_data = processor.read_data(data_path, delimiter='\t')

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.45)
    axcolor = 'lightgoldenrodyellow'

    # 添加文本框
    axbox_x = plt.axes([0.25, 0.35, 0.1, 0.04])
    axbox_y = plt.axes([0.25, 0.3, 0.1, 0.04])
    axbox_lam = plt.axes([0.25, 0.25, 0.1, 0.04])
    axbox_p = plt.axes([0.25, 0.2, 0.1, 0.04])
    axbox_niter = plt.axes([0.25, 0.15, 0.1, 0.04])
    ax_save = plt.axes([0.8, 0.025, 0.1, 0.04])
    ax_load = plt.axes([0.25, 0.025, 0.1, 0.04])

    text_x = TextBox(axbox_x, 'X', initial=str(x1))
    text_y = TextBox(axbox_y, 'Y', initial=str(y1))
    text_lam = TextBox(axbox_lam, 'lam', initial=str(lam))
    text_p = TextBox(axbox_p, 'p', initial=str(p))
    text_niter = TextBox(axbox_niter, 'niter', initial=str(niter))
    btn_save = Button(ax_save, 'Save', color=axcolor, hovercolor='0.975')
    btn_load = Button(ax_load, 'Load File', color=axcolor, hovercolor='0.975')

    text_x.on_submit(update)
    text_y.on_submit(update)
    text_lam.on_submit(update)
    text_p.on_submit(update)
    text_niter.on_submit(update)
    btn_save.on_clicked(save)
    btn_load.on_clicked(load_file)

    plt.show()
