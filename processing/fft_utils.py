# fft_utils.py
"""
Operaciones básicas con FFT y cálculo de energía por subbandas.
Usa solo lo visto en clase: FFT, magnitud, energía promedio.
Incluye mejoras: eliminación de DC, ventana Hamming y normalización.
"""

import numpy as np
from config import N_SUBBANDS, TARGET_N, FS


def preprocess_signal(x):
    """
    Preprocesamiento básico de la señal antes de la FFT:
    1. Elimina componente DC (media)
    2. Aplica ventana Hamming
    3. Asegura tipo float64
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)                      # quitar DC
    w = np.hamming(len(x))                  # ventana
    x = x * w
    return x


def compute_fft(x):
    """
    Calcula la FFT de una señal x y devuelve el espectro y las frecuencias asociadas.
    Solo usa la mitad positiva del espectro (por simetría de señales reales).
    """
    N = len(x)
    X = np.fft.fft(x, n=N)
    freqs = np.fft.fftfreq(N, d=1 / FS)
    half_N = N // 2
    return X[:half_N], freqs[:half_N]


def magnitude_spectrum(X):
    """
    Calcula la magnitud del espectro |X(k)|.
    """
    return np.abs(X)


def split_into_subbands(X_mag, n_subbands=N_SUBBANDS):
    """
    Divide el espectro en n_subbands partes iguales.
    Retorna una lista de arrays (una por subbanda).
    """
    N = len(X_mag)
    size = N // n_subbands
    subbands = []
    for i in range(n_subbands):
        start = i * size
        end = (i + 1) * size if i < n_subbands - 1 else N
        subbands.append(X_mag[start:end])
    return subbands


def compute_energy_per_subband(X_mag, n_subbands=N_SUBBANDS):
    """
    Calcula la energía de cada subbanda:
        E = (1/N_i) * sum(|X(k)|^2)
    Retorna un vector con una energía por subbanda.
    """
    subbands = split_into_subbands(X_mag, n_subbands)
    energies = []
    for sb in subbands:
        N_i = len(sb)
        if N_i == 0:
            energies.append(0.0)
        else:
            # energía proporcional a |X|^2
            E = (1.0 / N_i) * np.sum(sb ** 2)
            energies.append(E)
    return np.array(energies, dtype=np.float64)


def analyze_signal(x):
    """
    Pipeline completo:
    1. Preprocesa la señal (quita DC, aplica ventana)
    2. Calcula FFT
    3. Obtiene magnitud
    4. Calcula energía por subbanda
    5. Normaliza para que las energías sumen 1
    Devuelve (frecuencias, magnitud, energías_normalizadas)
    """
    x = preprocess_signal(x)
    X, freqs = compute_fft(x)
    X_mag = magnitude_spectrum(X)
    energies = compute_energy_per_subband(X_mag)

    # Normalización (fracción de energía por banda)
    total = np.sum(energies)
    if total > 0:
        energies = energies / total

    # Depuración opcional
    print("Energía total (normalizada):", np.sum(energies))
    print("Máximo banda:", np.max(energies))
    print("Mínimo banda:", np.min(energies))

    return freqs, X_mag, energies
