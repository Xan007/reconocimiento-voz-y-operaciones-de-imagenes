"""
frdct.py

Implementación de la Transformada Discreta del Coseno Fraccionaria (FrDCT) 2D
para cifrado de imágenes.

Basado en las ecuaciones 1.14 y 1.15:

FrDCT 2D (Eq. 1.14):
    F^α(u,v) = (C(u)C(v) / sqrt(NM)) * 
               Σx Σy f(x,y) * cos[π/N * (u + α/2)(x + 1/2)] * cos[π/M * (v + α/2)(y + 1/2)]

FrDCT Inversa 2D (Eq. 1.15):
    f(x,y) = (1/NM) *
             Σu Σv C(u)C(v) F^α(u,v) * cos[π/N * (u + α/2)(x + 1/2)] * cos[π/M * (v + α/2)(y + 1/2)]

Factor de normalización C(k):
    C(k) = 1/sqrt(2) si k = 0
    C(k) = 1         si k ≠ 0

NOTA IMPORTANTE:
La ecuación 1.15 tal como está escrita usa los MISMOS kernels de coseno que la 1.14.
Para que el par (forward, inverse) sea invertible, la inversa debe usar la TRANSPUESTA
de la matriz de transformación. Esto se implementa usando C(u)² en la inversa,
lo cual es equivalente a aplicar T^T cuando T es la matriz de la ecuación 1.14.
"""

import numpy as np
import io
import base64
import time
from scipy.fftpack import dct, idct

# Import PIL con fallback
try:
    from PIL import Image
except ImportError:
    Image = None


def C(k: int) -> float:
    """
    Factor de normalización C(k).
    
    C(k) = 1/sqrt(2) si k = 0
    C(k) = 1         si k ≠ 0
    """
    if k == 0:
        return 1.0 / np.sqrt(2.0)
    else:
        return 1.0


# ============================================================================
# IMPLEMENTACIÓN DIRECTA DE LAS ECUACIONES 1.14 Y 1.15 (FORMA EXPLÍCITA)
# ============================================================================

def frdct2d_eq114(f: np.ndarray, alpha: float) -> np.ndarray:
    """
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ F^α(u,v) = (C(u)C(v) / √(NM)) * Σx Σy f(x,y) *                       ║
    ║            cos[π/N * (u + α/2)(x + 1/2)] * cos[π/M * (v + α/2)(y + 1/2)] ║
    ╚════════════════════════════════════════════════════════════════════════╝

    """
    start_time = time.time()
    
    f = f.astype(np.float64)
    N, M = f.shape

    pi_over_N = np.pi / N
    pi_over_M = np.pi / M
    alpha_half = alpha / 2.0
    norm_nm = 1.0 / np.sqrt(N * M)
    
    # Precalcular C(u) y C(v)
    C_u_arr = np.array([C(u) for u in range(N)], dtype=np.float64)
    C_v_arr = np.array([C(v) for v in range(M)], dtype=np.float64)
    
    # Precalcular cosenos para x e y
    cos_x_lookup = np.cos(pi_over_N * np.outer(np.arange(N) + alpha_half, np.arange(N) + 0.5))
    cos_y_lookup = np.cos(pi_over_M * np.outer(np.arange(M) + alpha_half, np.arange(M) + 0.5))
    
    # Calcular F^α(u,v) - FÓRMULA DEL PROFESOR
    F_alpha = np.zeros((N, M), dtype=np.float64)
    
    for u in range(N):
        Cu = C_u_arr[u]
        for v in range(M):
            Cv = C_v_arr[v]
            acc = 0.0
            for x in range(N):
                cos_x = cos_x_lookup[u, x]
                for y in range(M):
                    acc += f[x, y] * cos_x * cos_y_lookup[v, y]
            F_alpha[u, v] = Cu * Cv * norm_nm * acc
    
    elapsed = time.time() - start_time
    print(f"[FrDCT 2D] α={alpha:.2f} Tamaño={N}x{M} Tiempo: {elapsed:.4f}s")
    
    return F_alpha


def ifrdct2d_eq115(F_alpha: np.ndarray, alpha: float) -> np.ndarray:
    """
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ f(x,y) = (1/NM) * Σu Σv C(u)C(v) F^α(u,v) *                          ║
    ║          cos[π/N * (u + α/2)(x + 1/2)] * cos[π/M * (v + α/2)(y + 1/2)] ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    start_time = time.time()
    
    # Extraer parte real si es complejo (de DOST)
    if np.iscomplexobj(F_alpha):
        print(f"[IFrDCT 2D] Detectado entrada compleja, extrayendo parte real...")
        F_alpha = np.real(F_alpha)
    
    F_alpha = F_alpha.astype(np.float64)
    N, M = F_alpha.shape

    pi_over_N = np.pi / N
    pi_over_M = np.pi / M
    alpha_half = alpha / 2.0
    norm_nm = 1.0 / (N * M)
    
    # Precalcular C(u) y C(v)
    C_u_arr = np.array([C(u) for u in range(N)], dtype=np.float64)
    C_v_arr = np.array([C(v) for v in range(M)], dtype=np.float64)
    
    # Precalcular cosenos para x e y
    cos_x_lookup = np.cos(pi_over_N * np.outer(np.arange(N) + alpha_half, np.arange(N) + 0.5))
    cos_y_lookup = np.cos(pi_over_M * np.outer(np.arange(M) + alpha_half, np.arange(M) + 0.5))
    
    # Calcular f(x,y) - FÓRMULA DEL PROFESOR
    f = np.zeros((N, M), dtype=np.float64)
    
    for x in range(N):
        for y in range(M):
            acc = 0.0
            for u in range(N):
                Cu = C_u_arr[u]
                cos_x = cos_x_lookup[u, x]
                for v in range(M):
                    Cv = C_v_arr[v]
                    acc += Cu * Cv * F_alpha[u, v] * cos_x * cos_y_lookup[v, y]
            f[x, y] = norm_nm * acc
    
    elapsed = time.time() - start_time
    print(f"[IFrDCT 2D] α={alpha:.2f} Tamaño={N}x{M} Tiempo: {elapsed:.4f}s")
    
    return f


def encrypt_image(image_array: np.ndarray, alpha: float, use_fast: bool = True) -> np.ndarray:
    """Cifra una imagen usando FrDCT con parámetro α.
    """
    # Siempre usar la forma explícita pedida por el profesor
    return frdct2d_eq114(image_array, alpha)


def decrypt_image(encrypted: np.ndarray, alpha: float, use_fast: bool = True) -> np.ndarray:
    """Descifra una imagen usando IFrDCT con parámetro α.
    """
    # Siempre usar la forma explícita pedida por el profesor
    return ifrdct2d_eq115(encrypted, alpha)


def normalize_for_display(arr: np.ndarray, use_log_scale: bool = False) -> np.ndarray:
    """
    Normaliza un array para visualización (0-255).
    
    Si use_log_scale=True, aplica escala logarítmica para mejor visualización
    de coeficientes de transformada (resalta detalles en rangos amplios).
    """
    arr = arr.astype(np.float64)
    
    if use_log_scale:
        # Escala logarítmica: log(1 + |valor|)
        arr = np.log1p(np.abs(arr))
    
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    if max_val - min_val > 0:
        normalized = (arr - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(arr)
    
    return (normalized * 255).astype(np.uint8)


def pil_to_base64(pil_image) -> str:
    """Convierte imagen PIL a string base64."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def base64_to_array(base64_str: str) -> np.ndarray:
    """Convierte string base64 a array numpy en escala de grises."""
    if Image is None:
        raise ImportError("PIL/Pillow is required")
    
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    img_bytes = base64.b64decode(base64_str)
    pil_img = Image.open(io.BytesIO(img_bytes))
    gray_img = pil_img.convert('L')
    
    return np.array(gray_img)


def array_to_base64(arr: np.ndarray, normalize: bool = True, use_log_scale: bool = False) -> str:
    """
    Convierte array numpy a string base64 con prefijo data:image.
    
    Si normalize=True y use_log_scale=True, usa escala logarítmica
    para mejor visualización de coeficientes de transformada.
    """
    if Image is None:
        raise ImportError("PIL/Pillow is required")
    
    if normalize:
        arr = normalize_for_display(arr, use_log_scale=use_log_scale)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    pil_img = Image.fromarray(arr, mode='L')
    b64 = pil_to_base64(pil_img)
    
    return f'data:image/png;base64,{b64}'
