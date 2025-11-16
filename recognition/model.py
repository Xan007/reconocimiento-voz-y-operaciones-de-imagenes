import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self, data):
        self.command = data["command"]
        self.mean_energy = np.array(data["mean_energy"])
        self.std_energy = np.array(data["std_energy"])
        self.num_samples = data["num_samples"]
        self.bands = len(self.mean_energy)
        
    def plot_energy_distribution(self, ax, show_std=True, normalize=False, compare_energies=None):
        """
        Graficar la distribución de energía por banda
        
        Parameters:
        -----------
        compare_energies : array-like, optional
            Si se proporciona, graficará estas energías junto al modelo para comparación
        """
        bands = np.arange(self.bands)
        energies = self.mean_energy
        std = self.std_energy
        width = 0.35 if compare_energies is not None else 0.8
        
        # Normalización para que todas las barras sean visibles
        if normalize:
            # Para el modelo
            min_e = np.min(energies)
            max_e = np.max(energies)
            energies = (energies - min_e) / (max_e - min_e) if max_e > min_e else energies
            std = std / (max_e - min_e) if max_e > min_e else std
            
            # Para la señal de comparación si existe
            if compare_energies is not None:
                min_c = np.min(compare_energies)
                max_c = np.max(compare_energies)
                if max_c > min_c:
                    compare_energies = (compare_energies - min_c) / (max_c - min_c)
        
        # Graficar barras del modelo
        ax.bar(bands - width/2 if compare_energies is not None else bands, 
               energies, width if compare_energies is not None else 0.8,
               alpha=0.6, label=f"Modelo {self.command.upper()}")
        
        # Graficar energías de comparación si se proporcionan
        if compare_energies is not None:
            ax.bar(bands + width/2, compare_energies, width,
                  alpha=0.6, color='red', label="Señal de Entrada")
        
        if show_std:
            if compare_energies is not None:
                x_pos = bands - width/2
            else:
                x_pos = bands
            ax.errorbar(x_pos, energies, yerr=std,
                       fmt='none', color='black', alpha=0.2,
                       capsize=5)
        
        ax.set_xlabel("Banda de frecuencia")
        ax.set_ylabel("Energía relativa" if normalize else "Energía")
        ax.set_title(f"Distribución de energía - {self.command.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if normalize:
            ax.set_ylim(0, max(energies) * 1.2)  # Dar un poco de espacio arriba
        
    def plot_std_comparison(self, ax, normalize=False):
        """Graficar la variabilidad por banda"""
        bands = np.arange(self.bands)
        std = self.std_energy
        
        if normalize:
            # Normalizar usando el rango de las energías medias
            min_energy = np.min(self.mean_energy)
            max_energy = np.max(self.mean_energy)
            std = std / (max_energy - min_energy)  # Escalar relativo al rango de energías
            
        ax.plot(bands, std, 'r-', label='Desviación estándar')
        ax.fill_between(bands, np.zeros_like(bands), std, 
                       alpha=0.2, color='red')
        
        ax.set_xlabel("Banda de frecuencia")
        ax.set_ylabel("Desviación estándar" + (" normalizada" if normalize else ""))
        ax.set_title(f"Variabilidad por banda - {self.command.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if normalize:
            ax.set_ylim(0, max(std) * 1.2)  # Dar un poco de espacio arriba