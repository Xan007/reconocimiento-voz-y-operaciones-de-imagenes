import numpy as np

def normalize_energies_for_display(energies):
    """
    Normaliza las energías para visualización.
    Convierte a dB y escala al rango [0, 1]
    """
    # Convertir a dB evitando log(0)
    min_value = 1e-10  # valor mínimo para evitar log(0)
    energies = np.maximum(energies, min_value)
    energies_db = 10 * np.log10(energies)
    
    # Normalizar al rango [0, 1]
    min_db = energies_db.min()
    max_db = energies_db.max()
    if max_db == min_db:
        return np.zeros_like(energies)
    return (energies_db - min_db) / (max_db - min_db)