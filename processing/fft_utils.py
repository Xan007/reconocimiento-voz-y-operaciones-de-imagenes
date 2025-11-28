# fft_utils.py
"""
Operaciones básicas con FFT y cálculo de energía por subbandas TEMPORALES.

PROCESO CORRECTO:
1. Dividir el audio en N_SUBBANDS segmentos temporales (cortar en tiempo)
2. Para cada segmento: calcular la energía (promedio de x^2)
3. El vector de características son las energías de cada subbanda temporal
4. Normalizar para que las energías sumen 1
"""

import numpy as np
from config import N_SUBBANDS, TARGET_N, FS


def preprocess_signal(x):
    """
    Preprocesamiento básico de la señal:
    1. Elimina componente DC (media)
    2. Asegura tipo float64
    """
    x = np.asarray(x, dtype=np.float64)
    # Eliminar DC (componente de media)
    x = x - np.mean(x)
    return x


def split_audio_into_subbands(x, n_subbands=N_SUBBANDS):
    """
    Divide la señal de audio en n_subbands segmentos TEMPORALES (partes iguales en tiempo).
    
    Parámetros:
    -----------
    x : ndarray
        Señal de audio
    n_subbands : int
        Número de subbandas (segmentos temporales)
    
    Retorna:
    --------
    subbands : list of ndarray
        Lista con n_subbands segmentos de la señal
    """
    N = len(x)
    size = N // n_subbands
    subbands = []
    for i in range(n_subbands):
        start = i * size
        # La última subbanda toma el resto para no perder muestras
        end = (i + 1) * size if i < n_subbands - 1 else N
        subbands.append(x[start:end])
    return subbands


def compute_energy(segment):
    """
    Calcula la energía de un segmento de audio.
    Energía = promedio de x^2 (energía media por muestra)
    
    E = (1/N) * sum(x[n]^2)
    """
    N = len(segment)
    if N == 0:
        return 0.0
    return np.sum(segment ** 2) / N


def compute_energy_per_subband(x, n_subbands=N_SUBBANDS):
    """
    Proceso principal:
    1. Divide el audio en n_subbands segmentos temporales
    2. Calcula la energía de cada segmento
    3. Retorna vector de energías
    
    Parámetros:
    -----------
    x : ndarray
        Señal de audio preprocesada
    n_subbands : int
        Número de subbandas temporales
    
    Retorna:
    --------
    energies : ndarray
        Vector con la energía de cada subbanda (shape: n_subbands,)
    """
    subbands = split_audio_into_subbands(x, n_subbands)
    energies = []
    for sb in subbands:
        E = compute_energy(sb)
        energies.append(E)
    return np.array(energies, dtype=np.float64)


def analyze_signal(x):
    """
    Pipeline completo para extraer características de audio:
    
    1. Preprocesa la señal (quita DC)
    2. Divide el audio en N_SUBBANDS segmentos temporales
    3. Calcula la energía de cada segmento
    4. Normaliza para que las energías sumen 1
    
    Retorna:
    --------
    tuple: (None, None, energías_normalizadas)
        - Los primeros dos valores son None (antes eran freqs y magnitud de FFT)
        - energías_normalizadas: vector de características normalizado
    """
    x = preprocess_signal(x)
    
    # DEBUG DETALLADO: Ver cada subbanda
    subbands = split_audio_into_subbands(x, N_SUBBANDS)
    print(f"\n{'='*60}")
    print(f"DEBUG SUBBANDAS TEMPORALES")
    print(f"{'='*60}")
    print(f"Audio total: {len(x)} muestras ({len(x)/FS:.3f} s)")
    for i, sb in enumerate(subbands):
        start_time = (i * len(sb)) / FS
        end_time = ((i + 1) * len(sb)) / FS
        max_val = np.max(np.abs(sb))
        energy_raw = np.sum(sb ** 2) / len(sb)
        print(f"  Subbanda {i+1}: {len(sb)} muestras ({start_time:.3f}s - {end_time:.3f}s)")
        print(f"    Max amplitud: {max_val:.6f}")
        print(f"    Energía (raw): {energy_raw:.10f}")
    print(f"{'='*60}\n")
    
    energies = compute_energy_per_subband(x)

    # Normalización (fracción de energía por banda)
    total = np.sum(energies)
    if total > 0:
        energies = energies / total

    # Depuración 
    print(f"Subbandas temporales: {N_SUBBANDS}")
    print(f"Energías ANTES de normalizar: {compute_energy_per_subband(x)}")
    print(f"Energías DESPUÉS de normalizar: {energies}")
    print(f"Energía total (normalizada): {np.sum(energies):.4f}")

    # Retornamos None para freqs y X_mag ya que no usamos FFT
    return None, None, energies


# ============ FUNCIONES AUXILIARES (para compatibilidad) ============

def compute_fft(x):
    """
    Calcula la FFT de una señal x y devuelve el espectro y las frecuencias asociadas.
    Solo usa la mitad positiva del espectro (por simetría de señales reales).
    (Mantenida para compatibilidad si se necesita visualizar espectro)
    """
    N = len(x)
    X = np.fft.fft(x, n=N)
    freqs = np.fft.fftfreq(N, d=1 / FS)
    half_N = N // 2
    return X[:half_N], freqs[:half_N]


def magnitude_spectrum(X):
    """
    Calcula la magnitud del espectro |X(k)|.
    (Mantenida para compatibilidad)
    """
    return np.abs(X)
