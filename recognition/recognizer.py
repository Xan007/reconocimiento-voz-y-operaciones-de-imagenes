# recognizer.py
"""
Etapa de reconocimiento: compara las energías de una grabación contra los modelos.

Según el documento:
  1. Entrada: comando a reconocer
  2. Pasar por filtros en paralelo (FFT) → obtener energía por banda
  3. Calcular vector de energías: [E_input1, E_input2, ..., E_inputN]
  4. Comparar con vectores de umbrales guardados: [E_c1, E_c2, ...], [E_d1, E_d2, ...]
  5. El comando se reconoce por MENOR DIFERENCIA

Fórmula de comparación (Distancia Euclidiana):
    d = sqrt(sum_i(E_input_i - E_modelo_i)^2)
    Se elige el modelo con menor distancia
"""

import os
import json
import numpy as np
from processing.audio_utils import read_wav, pad_or_trim, normalize_audio, record_and_prepare
from processing.fft_utils import analyze_signal
from config import MODELOS_DIR
from .model import Model

def load_models():
    """
    Carga todos los modelos .json guardados en data/modelos/.
    
    Retorna:
    --------
    models : dict
        Diccionario con los vectores de umbrales de cada comando
        {nombre_comando: objeto Model con [E_c1, E_c2, ..., E_cN]}
    """
    models = {}
    for fname in os.listdir(MODELOS_DIR):
        if not fname.endswith(".json"):
            continue
        # Saltar archivos de resultados de evaluación
        if fname == "evaluation_results.json":
            continue
        path = os.path.join(MODELOS_DIR, fname)
        with open(path, "r") as f:
            data = json.load(f)
        # Verificar que tiene la estructura de modelo
        if "command" not in data:
            continue
        model = Model(data)
        models[data["command"]] = model
    if not models:
        raise FileNotFoundError("No se encontraron modelos en data/modelos/. Entrena primero con trainer.py")
    return models

def compare_with_models(energies, models):
    """
    Compara un vector de energías contra todos los modelos entrenados.
    
    Algoritmo (según documento):
    1. Para cada modelo guardado:
        a) Obtener vector de umbrales: [E_c1, E_c2, ..., E_cN]
        b) Calcular distancia euclidiana:
           d = sqrt(sum_i(E_input_i - E_modelo_i)^2)
    2. Seleccionar el modelo con menor distancia
    3. Ese comando es el reconocido
    
    Parámetros:
    -----------
    energies : array
        Vector de energías de la entrada: [E_input1, E_input2, ..., E_inputN]
    models : dict
        Modelos cargados con sus vectores de umbrales
    
    Retorna:
    --------
    best_cmd : str o None
        Comando reconocido (el de menor distancia)
    diffs : dict
        Diccionario con la distancia a cada modelo
    """
    diffs = {}
    
    # Verificar que hay energía en la entrada
    total_energy = np.sum(energies)
    if total_energy < 1e-4:
        print("🔇 Energía demasiado baja: silencio")
        return None, {}

    # Comparar con cada modelo
    for cmd, model in models.items():
        # Vector de umbrales del modelo: [E_c1, E_c2, ..., E_cN]
        model_energies = model.mean_energy
        
        # Calcular distancia euclidiana (fórmula del documento)
        # d = sqrt(sum_i(E_input_i - E_modelo_i)^2)
        differences = energies - model_energies
        distance = np.sqrt(np.sum(differences ** 2))
        
        diffs[cmd] = float(distance)
        print(f"   {cmd.upper()}: distancia = {distance:.6f}")

    # Elegir el comando con MENOR distancia
    if not diffs:
        return None, diffs
        
    best_cmd = min(diffs, key=diffs.get)
    best_diff = diffs[best_cmd]

    print(f"\n✅ Comando reconocido: {best_cmd.upper()} (distancia={best_diff:.6f})")
    return best_cmd, diffs

def recognize_from_file(filename):
    """
    Reconoce el comando a partir de un archivo WAV.
    
    Pasos:
    1. Leer archivo WAV
    2. Normalizar y preparar
    3. Calcular energía por banda (FFT)
    4. Comparar con modelos
    
    Parámetros:
    -----------
    filename : str
        Ruta del archivo WAV a reconocer
    
    Retorna:
    --------
    best_cmd : str o None
        Comando reconocido
    """
    models = load_models()
    
    print(f"\n📁 Leyendo: {filename}")
    x, _ = read_wav(filename)
    x = normalize_audio(x)
    x = pad_or_trim(x)
    
    print("🔍 Calculando energía por banda (FFT)...")
    _, _, energies = analyze_signal(x)
    
    print(f"📊 Energías calculadas: {energies}")
    print("\n⚖️  Comparando con modelos:")
    best_cmd, diffs = compare_with_models(energies, models)
    
    if not best_cmd:
        print("❌ No se pudo reconocer ningún comando")
        return None
    
    return best_cmd

def recognize_from_mic():
    """
    Reconoce comando grabando desde micrófono en tiempo real.
    
    Pasos:
    1. Grabar audio (1 segundo)
    2. Normalizar y preparar
    3. Calcular energía por banda
    4. Comparar con modelos
    
    Retorna:
    --------
    best_cmd : str o None
        Comando reconocido
    """
    models = load_models()
    
    print("\n🎤 Grabando desde micrófono...")
    x = record_and_prepare()
    
    print("🔍 Calculando energía por banda (FFT)...")
    _, _, energies = analyze_signal(x)
    
    print(f"📊 Energías calculadas: {energies}")
    print("\n⚖️  Comparando con modelos:")
    best_cmd, diffs = compare_with_models(energies, models)
    
    if not best_cmd:
        print("❌ No se pudo reconocer ningún comando")
        return None
    
    return best_cmd
