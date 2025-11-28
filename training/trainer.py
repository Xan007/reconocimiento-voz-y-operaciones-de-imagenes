# trainer.py
"""
Etapa de entrenamiento: obtiene los vectores de umbrales de energía por comando.

PROCESO CORRECTO:
  1. Se procesan M grabaciones de cada comando
  2. Para cada grabación:
     a) Dividir el audio en N_SUBBANDS segmentos temporales
     b) Calcular la energía de cada segmento: E = (1/N) * sum(x[n]^2)
     c) Normalizar las energías para que sumen 1
  3. Promediar las energías de todas las grabaciones del comando
  4. Guardar como umbrales para reconocimiento

Algoritmo:
1. Leer grabaciones del comando
2. Normalizar y preparar audio
3. Dividir en subbandas temporales y calcular energía
4. Promediar energías de todas las grabaciones (vector de umbrales)
5. Guardar en JSON
"""

import os
import numpy as np
import json
from processing.audio_utils import read_wav, pad_or_trim, normalize_audio
from processing.fft_utils import analyze_signal
from config import COMANDOS_DIR, MODELOS_DIR

def compute_command_features(command_name):
    """
    Calcula las energías por subbanda temporal para TODAS las grabaciones de un comando.
    
    Parámetros:
    -----------
    command_name : str
        Nombre del comando (ej: "uno", "dos")
    
    Retorna:
    --------
    energies_list : list of arrays
        Lista de vectores de energía, uno por grabación
        Cada vector tiene N_SUBBANDS elementos
    
    Proceso:
    1. Dividir audio en N_SUBBANDS segmentos temporales
    2. Calcular energía de cada segmento: E = (1/N) * sum(x[n]^2)
    3. Normalizar para que sumen 1
    """
    command_path = os.path.join(COMANDOS_DIR, command_name)
    if not os.path.isdir(command_path):
        raise FileNotFoundError(f"No existe la carpeta {command_path}. Guarda tus grabaciones allí.")

    energies_list = []
    wav_files = [f for f in os.listdir(command_path) if f.lower().endswith(".wav")]
    
    print(f"\n📂 Procesando comando '{command_name}'...")
    print(f"   Grabaciones encontradas: {len(wav_files)}")
    
    for i, fname in enumerate(wav_files, 1):
        full_path = os.path.join(command_path, fname)
        print(f"   [{i}/{len(wav_files)}] Procesando {fname}...")
        
        # Leer y preparar audio
        x, _ = read_wav(full_path)
        x = normalize_audio(x)
        x = pad_or_trim(x)
        
        # Calcular energía por banda (según fórmula del documento)
        _, _, energies = analyze_signal(x)
        
        energies_list.append(energies)

    if len(energies_list) == 0:
        raise ValueError(f"No se encontraron grabaciones .wav en {command_path}")

    return energies_list

def train_command(command_name):
    """
    Entrena un comando calculando el vector de umbrales de energía.
    
    Algoritmo:
    1. Obtener todas las energías de las grabaciones
    2. Promediar para obtener el vector de umbrales
    3. Calcular desviación estándar (para análisis posterior)
    4. Guardar en JSON
    
    Parámetros:
    -----------
    command_name : str
        Nombre del comando
    
    Retorna:
    --------
    model : dict
        Diccionario con:
        - command: nombre del comando
        - mean_energy: vector de umbrales [E_c1, E_c2, ..., E_cN]
        - std_energy: desviación estándar por banda
        - num_samples: número de grabaciones usadas
    """
    energies_list = compute_command_features(command_name)
    energies_array = np.array(energies_list)
    
    # Vector de umbrales: promedio de energías por banda
    # Según documento: secuencia de umbrales E_c1, E_c2, E_c3, E_c4, ...
    mean_energy = np.mean(energies_array, axis=0)
    std_energy = np.std(energies_array, axis=0)

    # Convertir a float64 para precisión
    mean_energy = mean_energy.astype(np.float64)
    std_energy = std_energy.astype(np.float64)

    # Estructura del modelo
    model = {
        "command": command_name,
        "mean_energy": mean_energy.tolist(),  # Vector de umbrales E_c1, E_c2, ...
        "std_energy": std_energy.tolist(),    # Para análisis de variabilidad
        "num_samples": int(energies_array.shape[0]),  # M grabaciones
    }

    # Guardar modelo
    os.makedirs(MODELOS_DIR, exist_ok=True)
    out_path = os.path.join(MODELOS_DIR, f"{command_name}.json")
    json.dump(model, open(out_path, "w"), indent=4, ensure_ascii=False)

    print(f"✅ Modelo '{command_name}' entrenado:")
    print(f"   Grabaciones procesadas (M): {energies_array.shape[0]}")
    print(f"   Bandas de frecuencia (N): {len(mean_energy)}")
    print(f"   Vector de umbrales: {mean_energy}")
    print(f"   Desviación estándar: {std_energy}")
    print(f"   Guardado en: {out_path}\n")
    
    return model

def train_all():
    """
    Entrena automáticamente todos los comandos.
    
    Estructura esperada:
        data/comandos/
            ├── uno/
            │   ├── grabacion1.wav
            │   └── grabacion2.wav
            └── dos/
                ├── grabacion1.wav
                └── grabacion2.wav
    """
    command_names = sorted([d for d in os.listdir(COMANDOS_DIR)
                           if os.path.isdir(os.path.join(COMANDOS_DIR, d))])
    
    if not command_names:
        print("⚠️  No hay carpetas de comandos en data/comandos/")
        return

    print("=" * 60)
    print("🎓 ENTRENAMIENTO DE MODELOS")
    print("=" * 60)
    print(f"Comandos detectados: {command_names}\n")
    
    for name in command_names:
        train_command(name)
    
    print("=" * 60)
    print("✅ Entrenamiento completado")
    print("=" * 60)

if __name__ == "__main__":
    train_all()

