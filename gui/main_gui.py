import tkinter as tk
from tkinter import ttk
import queue
from gui.real_time_tab import RealTimeTab
from gui.models_tab import ModelsTab

class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Reconocimiento de Comandos por FFT")
        self.geometry("1200x800")
        
        # Cola para comunicación entre hilos
        self.audio_queue = queue.Queue()
        
        # Crear notebook para pestañas
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Pestaña de reconocimiento en tiempo real
        self.real_time_tab = RealTimeTab(self.notebook, self.audio_queue)
        self.notebook.add(self.real_time_tab, text='Reconocimiento en Tiempo Real')
        
        # Pestaña de visualización de modelos
        self.models_tab = ModelsTab(self.notebook)
        self.notebook.add(self.models_tab, text='Análisis de Modelos')
        
        # Configurar estilo
        self.style = ttk.Style()
        self.style.configure('TNotebook.Tab', padding=[12, 4])
        
    def on_closing(self):
        """Manejo del cierre de la aplicación"""
        self.destroy()

def main():
    app = MainApplication()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

if __name__ == "__main__":
    main()