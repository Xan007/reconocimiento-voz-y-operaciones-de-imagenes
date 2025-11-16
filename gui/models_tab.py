import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from recognition import recognizer

class ModelsTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.models = recognizer.load_models()
        self.setup_gui()
        
    def setup_gui(self):
        """Configurar la interfaz gráfica"""
        # Frame para selección de modelo
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Seleccionar modelo:").pack(side=tk.LEFT, padx=5)
        
        self.model_var = tk.StringVar(value="uno")
        model_combo = ttk.Combobox(control_frame, 
                                 textvariable=self.model_var,
                                 values=["uno", "dos"],
                                 state="readonly")
        model_combo.pack(side=tk.LEFT, padx=5)
        model_combo.bind('<<ComboboxSelected>>', self.update_plots)
        
        # Checkbox para normalización
        self.normalize_var = tk.BooleanVar(value=False)
        normalize_check = ttk.Checkbutton(control_frame, 
                                        text="Normalizar energías", 
                                        variable=self.normalize_var,
                                        command=self.update_plots)
        normalize_check.pack(side=tk.LEFT, padx=15)
        
        # Frame para gráficas
        plot_frame = ttk.Frame(self)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Crear figura con subplots
        self.fig = plt.figure(figsize=(12, 8))
        self.gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 1])
        
        # Subplot para distribución de energía
        self.ax_energy = self.fig.add_subplot(self.gs[0])
        self.ax_std = self.fig.add_subplot(self.gs[1])
        
        # Ajustar layout
        self.fig.tight_layout(pad=3.0)
        
        # Canvas de matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Información del modelo
        info_frame = ttk.LabelFrame(self, text="Información del Modelo")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.info_label = ttk.Label(info_frame, text="")
        self.info_label.pack(padx=5, pady=5)
        
        # Mostrar datos iniciales
        self.update_plots()
        
    def update_plots(self, event=None):
        """Actualizar las gráficas con el modelo seleccionado"""
        model_name = self.model_var.get()
        model = self.models[model_name]
        
        # Limpiar gráficas
        self.ax_energy.clear()
        self.ax_std.clear()
        
        # Graficar distribución de energía
        normalize = self.normalize_var.get()
        model.plot_energy_distribution(self.ax_energy, show_std=True, normalize=normalize)
        model.plot_std_comparison(self.ax_std, normalize=normalize)
        
        # Actualizar información del modelo
        try:
            info_text = (f"Modelo: {model_name.upper()}\n"
                        f"Número de muestras: {model.num_samples}\n"
                        f"Energía promedio total: {np.sum(model.mean_energy):.2f}\n"
                        f"Variabilidad promedio: {np.mean(model.std_energy):.2f}")
        except:
            info_text = "Error al cargar la información del modelo"
        self.info_label.config(text=info_text)
        
        # Refrescar canvas
        self.fig.tight_layout()
        self.canvas.draw()