"""
dost.py

Implementación de la Discrete Orthonormal Stockwell Transform (DOST) para
cifrado de imágenes.

Basado en las ecuaciones del documento (ecuaciones 1.19 y 1.20):

DOST Directa (Eq. 1.19):
    DOST(x) = DFT(x[n] · e^(-jωn))
    
DOST Inversa (Eq. 1.20):
    x[n] = e^(jωn) · DFT^(-1)(DOST(x))

Se aplican estas ecuaciones 1D a cada fila y columna para la versión 2D.
La DOST combina características de la Transformada de Fourier y la 
Transformada Wavelet, proporcionando localización tiempo-frecuencia.

NOTA: Solo se usan las implementaciones que aplican literalmente
las ecuaciones 1.19 y 1.20. Las versiones "_fast" han sido comentadas
ya que no corresponden a las ecuaciones requeridas por el profesor.
"""

import numpy as np
import time


def dost_1d(x: np.ndarray) -> np.ndarray:
    """
    Aplica la DOST 1D a un vector (Ecuación 1.19).
    
    DOST(x) = DFT(x[n] · e^(-jωn))
    
    donde ω = 2π/N * k para diferentes bandas de frecuencia.
    
    Versión vectorizada: evita loops innecesarios.
    """
    N = len(x)
    n = np.arange(N, dtype=np.float64)
    
    # Frecuencia angular normalizada (centro del espectro)
    omega = np.pi  
    
    # Modulación vectorizada: x[n] · e^(-jωn)
    modulated = x * np.exp(-1j * omega * n / N * 2)
    
    # Aplicar DFT vectorizada
    dost_result = np.fft.fft(modulated)
    
    return dost_result


def idost_1d(dost_x: np.ndarray) -> np.ndarray:
    """
    Aplica la DOST Inversa 1D (Ecuación 1.20).
    
    x[n] = e^(jωn) · DFT^(-1)(DOST(x))
    
    Versión vectorizada: evita loops innecesarios.
    """
    N = len(dost_x)
    n = np.arange(N, dtype=np.float64)
    
    # Frecuencia angular normalizada (debe coincidir con dost_1d)
    omega = np.pi
    
    # Aplicar DFT inversa vectorizada
    ifft_result = np.fft.ifft(dost_x)
    
    # Demodulación vectorizada: e^(jωn) · resultado
    x = ifft_result * np.exp(1j * omega * n / N * 2)
    
    return x


def dost_2d(f: np.ndarray) -> np.ndarray:
    """
    Aplica la DOST 2D a una imagen/matriz usando las ecuaciones 1.19 y 1.20.
    
    Se aplica la DOST 1D (Eq. 1.19) a cada fila y luego a cada columna.
    Versión OPTIMIZADA: O(N² log N) en lugar de O(N⁴).
    
    Parameters:
        f: Matriz de entrada (imagen en escala de grises)
        
    Returns:
        Matriz transformada (compleja)
    """
    f = f.astype(np.complex128)
    N, M = f.shape
    
    # Aplicar DOST a cada fila (Ec. 1.19) - vectorizado
    result = np.zeros_like(f, dtype=np.complex128)
    for i in range(N):
        result[i, :] = dost_1d(f[i, :])
    
    # Aplicar DOST a cada columna (Ec. 1.19) - vectorizado
    for j in range(M):
        result[:, j] = dost_1d(result[:, j])
    
    return result


def idost_2d(dost_f: np.ndarray) -> np.ndarray:
    """
    Aplica la DOST Inversa 2D usando las ecuaciones 1.19 y 1.20.
    
    Se aplica la DOST inversa 1D (Ec. 1.20) a cada columna y luego a cada fila.
    Versión OPTIMIZADA: O(N² log N) en lugar de O(N⁴).
    
    Parameters:
        dost_f: Matriz transformada (compleja)
        
    Returns:
        Matriz reconstruida
    """
    dost_f = dost_f.astype(np.complex128)
    N, M = dost_f.shape
    
    # Aplicar DOST inversa a cada columna (Ec. 1.20) - vectorizado
    result = np.zeros_like(dost_f, dtype=np.complex128)
    for j in range(M):
        result[:, j] = idost_1d(dost_f[:, j])
    
    # Aplicar DOST inversa a cada fila (Ec. 1.20) - vectorizado
    for i in range(N):
        result[i, :] = idost_1d(result[i, :])
    
    return result


# ============================================================================
# IMPLEMENTACIÓN OPTIMIZADA CON FFT2D (ecuaciones 1.19 y 1.20 en 2D)
# ============================================================================

def dost_2d_optimized(f: np.ndarray) -> np.ndarray:
    """
    Aplica la DOST 2D de forma OPTIMIZADA usando FFT2D.
    
    Implementa las ecuaciones 1.19 y 1.20 en 2D:
    - Modulación 2D: f(x,y) · e^(-jωx*x) · e^(-jωy*y)
    - FFT2D
    
    Esta versión es equivalente a aplicar 1D por filas y columnas,
    pero mucho más rápida gracias a la FFT2D vectorizada.
    
    O(N² log N) complexity con factor constante bajo.
    """
    start_time = time.time()
    
    f = f.astype(np.complex128)
    N, M = f.shape
    
    # Crear vectores de índices
    n_rows = np.arange(N, dtype=np.float64).reshape(-1, 1)
    m_cols = np.arange(M, dtype=np.float64).reshape(1, -1)
    
    omega = np.pi
    
    # Modulación 2D vectorizada: e^(-jω*n/N*2) * e^(-jω*m/M*2)
    mod_rows = np.exp(-1j * omega * n_rows / N * 2)
    mod_cols = np.exp(-1j * omega * m_cols / M * 2)
    
    # Aplicar modulación
    modulated = f * mod_rows * mod_cols
    
    # FFT 2D vectorizado
    result = np.fft.fft2(modulated)
    
    elapsed = time.time() - start_time
    print(f"[DOST 2D] Tamaño={N}x{M} Tiempo: {elapsed:.4f}s")
    
    return result


def idost_2d_optimized(dost_f: np.ndarray) -> np.ndarray:
    """
    Aplica la DOST Inversa 2D de forma OPTIMIZADA usando IFFT2D.
    
    Implementa las ecuaciones 1.19 y 1.20 en 2D (versión inversa):
    - IFFT2D
    - Demodulación 2D: resultado · e^(jωx*x) · e^(jωy*y)
    
    O(N² log N) complexity con factor constante bajo.
    """
    start_time = time.time()
    
    dost_f = dost_f.astype(np.complex128)
    N, M = dost_f.shape
    
    # Crear vectores de índices
    n_rows = np.arange(N, dtype=np.float64).reshape(-1, 1)
    m_cols = np.arange(M, dtype=np.float64).reshape(1, -1)
    
    omega = np.pi
    
    # IFFT 2D vectorizado
    ifft_result = np.fft.ifft2(dost_f)
    
    # Demodulación 2D vectorizada: e^(jω*n/N*2) * e^(jω*m/M*2)
    demod_rows = np.exp(1j * omega * n_rows / N * 2)
    demod_cols = np.exp(1j * omega * m_cols / M * 2)
    
    result = ifft_result * demod_rows * demod_cols
    
    elapsed = time.time() - start_time
    print(f"[IDOST 2D] Tamaño={N}x{M} Tiempo: {elapsed:.4f}s")
    
    return result


# Usar la versión optimizada como default
dost_2d = dost_2d_optimized
idost_2d = idost_2d_optimized


def normalize_complex_for_display(arr: np.ndarray, use_log_scale: bool = True) -> np.ndarray:
    """
    Normaliza una matriz compleja para visualización.
    
    Usa la magnitud (valor absoluto) del array complejo.
    Si use_log_scale=True, aplica escala logarítmica para mejor visualización.
    """
    # Obtener magnitud
    magnitude = np.abs(arr)
    
    if use_log_scale:
        # Escala logarítmica: log(1 + |valor|)
        magnitude = np.log1p(magnitude)
    
    # Normalizar a 0-255
    min_val = np.min(magnitude)
    max_val = np.max(magnitude)
    
    if max_val - min_val > 0:
        normalized = (magnitude - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(magnitude)
    
    return (normalized * 255).astype(np.uint8)


# Aliases para compatibilidad (usando las ecuaciones 1.19 y 1.20)
dost_forward = dost_2d_optimized
dost_inverse = idost_2d_optimized
