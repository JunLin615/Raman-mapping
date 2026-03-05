import os
import tkinter as tk
from tkinter import filedialog, messagebox
import Pretreatment as pre


class BaselineRemovalApp:
    """
    GUI for batch baseline removal of Witec exported mapping / spectra data.

    Backward compatible by default:
    - If "Compatibility mode" is enabled, it will:
        * Process files even when filename does not match the strict standard,
          and record issues to a non-.txt log file (CSV).
        * Auto-detect header (avoid dropping first data row in no-header files).
        * Auto-sniff common delimiters (\t, comma, semicolon, whitespace).
    - If disabled, it keeps the original strict behavior.
    """

    def __init__(self, master):
        self.master = master
        master.title("Witec Mapping Baseline Removal")

        self.input_dir = ""
        self.output_dir = ""

        # Input directory selection
        self.input_label = tk.Label(master, text="Input Directory:")
        self.input_label.grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.input_entry = tk.Entry(master, width=55)
        self.input_entry.grid(row=0, column=1, padx=5, pady=5)
        self.input_button = tk.Button(master, text="Browse", command=self.browse_input)
        self.input_button.grid(row=0, column=2, padx=5, pady=5)

        # Output directory selection
        self.output_label = tk.Label(master, text="Output Directory:")
        self.output_label.grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.output_entry = tk.Entry(master, width=55)
        self.output_entry.grid(row=1, column=1, padx=5, pady=5)
        self.output_button = tk.Button(master, text="Browse", command=self.browse_output)
        self.output_button.grid(row=1, column=2, padx=5, pady=5)

        # Parameters (pre-fill defaults)
        self.lam_label = tk.Label(master, text="lam (default 1000):")
        self.lam_label.grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.lam_entry = tk.Entry(master)
        self.lam_entry.insert(0, "1000")
        self.lam_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        self.p_label = tk.Label(master, text="p (default 0.01):")
        self.p_label.grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.p_entry = tk.Entry(master)
        self.p_entry.insert(0, "0.01")
        self.p_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)

        self.niter_label = tk.Label(master, text="niter (default 3):")
        self.niter_label.grid(row=4, column=0, sticky="e", padx=5, pady=5)
        self.niter_entry = tk.Entry(master)
        self.niter_entry.insert(0, "3")
        self.niter_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)

        # Compatibility mode toggle
        self.compat_var = tk.BooleanVar(value=True)
        self.compat_check = tk.Checkbutton(
            master,
            text="Compatibility mode (process non-standard files + write issues log)",
            variable=self.compat_var,
        )
        self.compat_check.grid(row=5, column=1, sticky="w", padx=5, pady=2)

        # Process button
        self.process_button = tk.Button(master, text="Process", command=self.process_files)
        self.process_button.grid(row=6, column=1, pady=15)

        # Tip
        self.tip_label = tk.Label(
            master,
            text="Tip: When compatibility mode is enabled, a CSV log will be saved in the output directory.",
            fg="gray"
        )
        self.tip_label.grid(row=7, column=0, columnspan=3, sticky="w", padx=5, pady=2)

    def browse_input(self):
        self.input_dir = filedialog.askdirectory()
        if self.input_dir:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, self.input_dir)

    def browse_output(self):
        self.output_dir = filedialog.askdirectory()
        if self.output_dir:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, self.output_dir)

    def process_files(self):
        self.input_dir = self.input_entry.get().strip()
        self.output_dir = self.output_entry.get().strip()

        if not self.input_dir or not self.output_dir:
            messagebox.showerror("Error", "Please select both input and output directories.")
            return

        # Read parameters (allow empty -> fallback defaults)
        try:
            lam_text = self.lam_entry.get().strip()
            p_text = self.p_entry.get().strip()
            niter_text = self.niter_entry.get().strip()

            lam = float(lam_text) if lam_text else 1000.0
            p = float(p_text) if p_text else 0.01
            niter = int(float(niter_text)) if niter_text else 3
        except ValueError:
            messagebox.showerror("Error", "Invalid input for lam, p, or niter.")
            return

        compat_mode = bool(self.compat_var.get())

        # Prepare log path (non-.txt) only when compatibility mode is enabled
        log_path = None
        if compat_mode:
            log_path = os.path.join(self.output_dir, "rebaseline_issues.csv")

        try:
            processor = pre.WitecRamanProcessor(
                self.input_dir,
                self.output_dir,
                compat_mode=compat_mode,
                log_path=log_path,
                header_mode=("auto" if compat_mode else "legacy"),
            )
            processor.lam = lam
            processor.p = p
            processor.niter = niter
            processor.Denoise = False  # Only baseline removal in this GUI

            processor.process_directory_reBaseLine()

            if compat_mode and log_path:
                messagebox.showinfo(
                    "Success",
                    f"Processing completed.\nIssues log saved to:\n{log_path}"
                )
            else:
                messagebox.showinfo("Success", "Processing completed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BaselineRemovalApp(root)
    root.mainloop()
