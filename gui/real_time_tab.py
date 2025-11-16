import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
from queue import Queue
import threading

from config import FS, TARGET_N
from processing.audio_utils import normalize_audio, pad_or_trim
from processing.fft_utils import analyze_signal
from recognition import recognizer

class RealTimeTab(ttk.Frame):
    def __init__(self, parent, audio_queue):
        super().__init__(parent)
        self.audio_queue = audio_queue
        self.recording = False
        self.models = recognizer.load_models()
        
        self.setup_gui()
        self.setup_audio()
        
    def setup_gui(self):
        """Configurar la interfaz gráfica"""
        # Frame principal con 3 subplots
        self.fig = plt.figure(figsize=(12, 8))
        self.gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5],
                                      width_ratios=[1, 1])
        
        # Subplot para forma de onda (arriba, todo el ancho)
        self.ax_wave = self.fig.add_subplot(self.gs[0, :])
        self.ax_wave.set_title("Forma de Onda")
        self.ax_wave.set_xlabel("Tiempo (s)")
        self.ax_wave.set_ylabel("Amplitud")
        
        # Subplot para espectro (medio, todo el ancho)
        self.ax_spec = self.fig.add_subplot(self.gs[1, :])
        self.ax_spec.set_title("Espectro de Frecuencia")
        self.ax_spec.set_xlabel("Frecuencia (Hz)")
        self.ax_spec.set_ylabel("Magnitud (dB)")
        
        # Subplots para comparación de modelos (abajo, dividido)
        self.ax_model1 = self.fig.add_subplot(self.gs[2, 0])
        self.ax_model2 = self.fig.add_subplot(self.gs[2, 1])
        
        # Ajustar layout
        self.fig.tight_layout(pad=3.0)
        
        # Canvas de matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Frame para controles
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Botón de inicio/parada
        self.record_button = ttk.Button(control_frame, text="Iniciar",
                                      command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=5)
        
        # Etiqueta de estado
        self.status_label = ttk.Label(control_frame, text="Estado: Detenido")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
    def setup_audio(self):
        """Configurar el sistema de audio"""
        self.buffer = np.zeros(TARGET_N, dtype=np.float32)
        self.processing = False
        
    def toggle_recording(self):
        """Alternar entre grabar y detener"""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        """Iniciar la grabación"""
        self.recording = True
        self.record_button.config(text="Detener")
        self.status_label.config(text="Estado: Grabando")
        
        # Iniciar stream de audio
        self.stream = sd.InputStream(
            channels=1,
            samplerate=FS,
            blocksize=int(FS * 0.1),  # Procesar cada 100ms
            callback=self.audio_callback
        )
        self.stream.start()
        
        # Iniciar actualización de gráficos
        self.update_plots()
        
    def stop_recording(self):
        """Detener la grabación"""
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self.recording = False
        self.record_button.config(text="Iniciar")
        self.status_label.config(text="Estado: Detenido")
        
    def audio_callback(self, indata, frames, time, status):
        """Callback para procesar el audio en tiempo real"""
        if status:
            print('Error:', status)
        
        # Convertir a mono si es estéreo
        if indata.shape[1] > 1:
            data = np.mean(indata, axis=1)
        else:
            data = indata[:, 0]
            
        # Actualizar buffer circular
        self.buffer[:-frames] = self.buffer[frames:]
        self.buffer[-frames:] = data
        
        if not self.processing:
            self.processing = True
            try:
                # Normalizar y preparar audio
                x = normalize_audio(self.buffer.copy())
                x = pad_or_trim(x)
                # Analizar señal
                spectrum, freqs, energies = analyze_signal(x)
                # Enviar datos para actualización
                self.audio_queue.put({
                    'waveform': x,
                    'spectrum': spectrum,
                    'freqs': freqs,
                    'energies': energies
                })
            finally:
                self.processing = False
                
    def update_plots(self):
        """Actualizar las gráficas con nuevos datos"""
        if not self.recording:
            return
            
        try:
            data = self.audio_queue.get_nowait()
            
            # Actualizar forma de onda
            self.ax_wave.clear()
            tiempo = np.arange(len(data['waveform'])) / FS
            self.ax_wave.plot(tiempo, data['waveform'])
            self.ax_wave.set_title("Forma de Onda")
            self.ax_wave.set_xlabel("Tiempo (s)")
            self.ax_wave.set_ylabel("Amplitud")
            
            # Actualizar espectro
            self.ax_spec.clear()
            self.ax_spec.plot(data['freqs'], 20 * np.log10(np.abs(data['spectrum'])))
            self.ax_spec.set_title("Espectro de Frecuencia")
            self.ax_spec.set_xlabel("Frecuencia (Hz)")
            self.ax_spec.set_ylabel("Magnitud (dB)")
            self.ax_spec.grid(True, alpha=0.3)
            
            # Actualizar comparaciones de modelo
            self.ax_model1.clear()
            self.ax_model2.clear()
            
            # Configurar más divisiones en el eje x
            xticks = np.arange(16)  # 16 subbandas
            
            # Configurar el ax_model1
            self.models['uno'].plot_energy_distribution(
                self.ax_model1, normalize=True, compare_energies=data['energies'])
            self.ax_model1.set_title("Comparación con Modelo UNO")
            self.ax_model1.set_xticks(xticks)
            self.ax_model1.set_xticklabels([f'B{i+1}' for i in xticks])
            self.ax_model1.grid(True, axis='y', alpha=0.3)
            
            # Configurar el ax_model2
            self.models['dos'].plot_energy_distribution(
                self.ax_model2, normalize=True, compare_energies=data['energies'])
            self.ax_model2.set_title("Comparación con Modelo DOS")
            self.ax_model2.set_xticks(xticks)
            self.ax_model2.set_xticklabels([f'B{i+1}' for i in xticks])
            self.ax_model2.grid(True, axis='y', alpha=0.3)
            
            # Refrescar canvas
            self.canvas.draw()
            
        except Queue.Empty:
            pass
            
        # Programar siguiente actualización
        self.after(50, self.update_plots)