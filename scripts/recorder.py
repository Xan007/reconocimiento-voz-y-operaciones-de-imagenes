import os
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from processing.audio_utils import save_wav
from config import COMANDOS_DIR, FS, DURATION
import time 
class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Grabador de comandos (UNO / DOS)")
        self.root.geometry("700x500")
        self.recording = False
        self.audio_data = []
        self.stream = None

        # ===== UI =====
        self.cmd_var = tk.StringVar(value="uno")

        ttk.Label(root, text="Comando a grabar:").pack(pady=5)
        ttk.Radiobutton(root, text="UNO", variable=self.cmd_var, value="uno").pack()
        ttk.Radiobutton(root, text="DOS", variable=self.cmd_var, value="dos").pack()

        self.record_btn = ttk.Button(root, text="🎙️ Grabar", command=self.toggle_recording)
        self.record_btn.pack(pady=15)

        self.status_label = ttk.Label(root, text="Listo para grabar.")
        self.status_label.pack(pady=5)

        # ===== Matplotlib =====
        fig = Figure(figsize=(6, 3), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_title("Forma de onda (tiempo real)")
        self.ax.set_xlim(0, int(FS * DURATION))
        self.ax.set_ylim(-1, 1)
        self.line, = self.ax.plot([], [], lw=1.2, color="blue")
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def toggle_recording(self):
        if not self.recording:
            
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.audio_data = []
        self.recording = True
        self.record_btn.config(text="⏹️ Detener")
        self.status_label.config(text="Grabando...")

        time.sleep(0.5) 

        # Crear stream con callback
        self.stream = sd.InputStream(
            samplerate=FS,
            channels=1,
            dtype='float32',
            callback=self.audio_callback
        )
        self.stream.start()

        # Actualizar gráfico cada 100ms
        self.update_plot()

        # Detener automáticamente después de DURATION segundos
        self.root.after(int(DURATION * 1010), self.stop_recording)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print("⚠️", status)
        if self.recording:
            self.audio_data.extend(indata[:, 0])

    def update_plot(self):
        if not self.recording:
            return
        if len(self.audio_data) > 0:
            y = np.array(self.audio_data[-int(FS * DURATION):])
            x = np.arange(len(y))
            self.line.set_data(x, y)
            self.ax.set_xlim(0, len(y))
            self.canvas.draw_idle()
        self.root.after(100, self.update_plot)

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.record_btn.config(text="🎙️ Grabar")
        self.status_label.config(text="Grabación terminada.")
        self.save_recording()

    def save_recording(self):
        if not self.audio_data:
            self.status_label.config(text="⚠️ No se grabó nada.")
            return

        audio = np.array(self.audio_data, dtype=np.float32)
        cmd = self.cmd_var.get()
        folder = os.path.join(COMANDOS_DIR, cmd)
        os.makedirs(folder, exist_ok=True)

        # Numerar archivo automáticamente
        existing = [f for f in os.listdir(folder) if f.endswith(".wav")]
        idx = len(existing) + 1
        filename = os.path.join(folder, f"{cmd}_{idx:02d}.wav")

        save_wav(filename, audio, FS)
        self.status_label.config(text=f"✅ Guardado: {filename}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RecorderApp(root)
    root.mainloop()
