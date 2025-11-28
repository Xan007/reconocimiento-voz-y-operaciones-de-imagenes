# recognizer.py
"""
Etapa de reconocimiento: compara las energías de una grabación contra los modelos.

PROCESO:
  1. Entrada: audio del comando a reconocer
  2. Dividir el audio en N_SUBBANDS segmentos temporales
  3. Calcular la energía de cada segmento: E = (1/N) * sum(x[n]^2)
  4. Normalizar las energías para que sumen 1
  5. Comparar con vectores de umbrales guardados
  6. El comando se reconoce por MENOR DIFERENCIA

Fórmula de comparación (Distancia Euclidiana):
    d = sqrt(sum_i(E_input_i - E_modelo_i)^2)
    Se elige el modelo con menor distancia
"""

import os
import json
import numpy as np
from processing.audio_utils import read_wav, pad_or_trim, normalize_audio, record_and_prepare
from processing.fft_utils import analyze_signal
from config import MODELOS_DIR, RECOGNITION_THRESHOLD
from .model import Model
from .distance_metrics import calculate_distance, AVAILABLE_METRICS

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

def compare_with_models(energies, models, distance_method='euclidean', threshold=None):
    """
    Compara un vector de energías contra todos los modelos entrenados.
    
    Algoritmo:
    1. Para cada modelo guardado:
        a) Obtener vector de energías: [E_c1, E_c2, ..., E_cN]
        b) Obtener vector de desviaciones: [σ_c1, σ_c2, ..., σ_cN]
        c) Calcular distancia usando el método seleccionado
    2. Seleccionar el modelo con menor distancia
    3. Verificar si la distancia está por debajo del umbral
    4. Si está por debajo del umbral, el comando es válido
    
    Parámetros:
    -----------
    energies : array
        Vector de energías de la entrada: [E_input1, E_input2, ..., E_inputN]
    models : dict
        Modelos cargados con sus vectores de umbrales y desviaciones
    distance_method : str
        Método de distancia a usar:
        - 'euclidean': Distancia euclidiana clásica
        - 'weighted_euclidean': Ponderada por estabilidad (1/std)
        - 'mahalanobis_diagonal': Mahalanobis con matriz diagonal
        - 'nll_gaussian': Verosimilitud gaussiana negativa
        - 'downweight_unstable': Reduce peso en bandas inestables
        - 'outlier_detection': Penaliza valores extremos
    threshold : float o None
        Umbral de distancia máxima para aceptar el comando.
        Si es None, usa RECOGNITION_THRESHOLD del config.
    
    Retorna:
    --------
    best_cmd : str o None
        Comando reconocido (el de menor distancia)
    diffs : dict
        Diccionario con la distancia a cada modelo
    is_valid : bool
        True si la distancia está por debajo del umbral
    """
    if threshold is None:
        threshold = RECOGNITION_THRESHOLD
    
    diffs = {}
    
    # Verificar que hay energía en la entrada
    total_energy = np.sum(energies)
    if total_energy < 1e-4:
        print("🔇 Energía demasiado baja: silencio")
        return None, {}, False

    # Comparar con cada modelo
    for cmd, model in models.items():
        # Obtener vectores del modelo
        model_energies = model.mean_energy
        model_std_energies = model.std_energy
        
        # Calcular distancia usando el método seleccionado
        try:
            distance = calculate_distance(
                energies, 
                model_energies, 
                model_std_energies,
                distance_method=distance_method
            )
        except ValueError as e:
            print(f"⚠️  Error en método de distancia: {e}")
            # Fallback a euclidiana
            distance = calculate_distance(energies, model_energies, model_std_energies)
        
        diffs[cmd] = float(distance)
        print(f"   {cmd.upper()}: distancia = {distance:.6f}")

    # Elegir el comando con MENOR distancia
    if not diffs:
        return None, diffs, False
        
    best_cmd = min(diffs, key=diffs.get)
    best_diff = diffs[best_cmd]
    
    # Verificar si está por debajo del umbral
    is_valid = best_diff < threshold
    
    if is_valid:
        print(f"\n✅ Comando reconocido: {best_cmd.upper()} (distancia={best_diff:.6f}, umbral={threshold}) [método: {distance_method}]")
    else:
        print(f"\n⚠️  Mejor coincidencia: {best_cmd.upper()} pero distancia={best_diff:.6f} > umbral={threshold}")
    
    return best_cmd, diffs, is_valid

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
    best_cmd, diffs, is_valid = compare_with_models(energies, models)
    
    if not best_cmd or not is_valid:
        print("❌ No se pudo reconocer ningún comando con suficiente confianza")
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
    best_cmd, diffs, is_valid = compare_with_models(energies, models)
    
    if not best_cmd or not is_valid:
        print("❌ No se pudo reconocer ningún comando con suficiente confianza")
        return None
    
    return best_cmd
