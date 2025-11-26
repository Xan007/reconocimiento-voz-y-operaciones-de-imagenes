"""
Módulo de procesamiento de imágenes interactivo.
Integra reconocimiento de voz con procesamiento de imagen.
"""

import numpy as np
import base64
import io
from PIL import Image
from scipy import fft
from scipy.ndimage import sobel


def convert_to_grayscale(image_array):
    """Convierte imagen RGB a escala de grises."""
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Usar fórmula estándar RGB a gris
        r = image_array[:, :, 0]
        g = image_array[:, :, 1]
        b = image_array[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.astype(np.uint8)
    return image_array


def compute_dct_magnitude(image_gray):
    """Calcula magnitud DCT 2D de imagen gris."""
    # DCT 2D
    dct_2d = fft.dctn(image_gray, type=2, norm='ortho')
    
    # Magnitud (valor absoluto)
    magnitude = np.abs(dct_2d)
    
    return magnitude.tolist()


def apply_dct_compression(image_gray, compression_percent):
    """
    Aplica compresión DCT eliminando coeficientes pequeños.
    
    Parámetros:
    -----------
    image_gray : array
        Imagen en escala de grises [256, 256]
    compression_percent : float
        Porcentaje de coeficientes a retener (0-100)
    
    Retorna:
    --------
    compressed : array
        Imagen comprimida
    magnitude : array
        Magnitud DCT original
    stats : dict
        Estadísticas de compresión
    """
    # DCT 2D
    dct_2d = fft.dctn(image_gray, type=2, norm='ortho')
    
    # Magnitud para visualización
    magnitude = np.abs(dct_2d)
    
    # Calcular umbral
    flat_coeff = np.abs(dct_2d).flatten()
    threshold_idx = int(len(flat_coeff) * (compression_percent / 100.0))
    threshold = np.sort(flat_coeff)[threshold_idx]
    
    # Crear máscara: mantener coeficientes por encima del umbral
    mask = np.abs(dct_2d) >= threshold
    dct_compressed = dct_2d * mask
    
    # DCT inversa
    compressed = fft.idctn(dct_compressed, type=2, norm='ortho')
    compressed = np.clip(compressed, 0, 255).astype(np.uint8)
    
    # Calcular estadísticas
    coef_total = 256 * 256
    coef_retained = np.sum(mask)
    coef_removed = coef_total - coef_retained
    
    # Error
    error = compressed.astype(float) - image_gray.astype(float)
    error_rms = np.sqrt(np.mean(error ** 2))
    error_max = np.max(np.abs(error))
    
    stats = {
        'total_coeficientes': int(coef_total),
        'coef_retenidos': int(coef_retained),
        'coef_eliminados': int(coef_removed),
        'porcentaje_compresion': float(compression_percent),
        'error_rms': float(error_rms),
        'error_maximo': float(error_max),
        'reduccion_bytes': f"{100 - compression_percent:.1f}%"
    }
    
    return compressed, magnitude, stats


def reconstruct_from_dct(image_gray, compression_percent, mode='grey'):
    """
    Reconstruye imagen desde DCT comprimida.
    
    Parámetros:
    -----------
    image_gray : array
        Imagen original en gris
    compression_percent : float
        Porcentaje de compresión
    mode : str
        'grey' para escala de grises, 'color' para color
    
    Retorna:
    --------
    compressed : array
        Imagen reconstruida
    stats : dict
        Estadísticas
    """
    compressed, _, stats = apply_dct_compression(image_gray, compression_percent)
    
    return compressed, stats


def pil_to_base64(pil_image):
    """Convierte imagen PIL a base64."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


def numpy_to_pil(array):
    """Convierte array numpy a imagen PIL."""
    if len(array.shape) == 2:
        # Escala de grises
        return Image.fromarray(array, mode='L')
    else:
        # RGB
        return Image.fromarray(array, mode='RGB')


def base64_to_numpy(b64_string):
    """Convierte base64 a array numpy."""
    try:
        # Si es un string de bytes, decodificar
        if isinstance(b64_string, str):
            img_data = base64.b64decode(b64_string)
        else:
            img_data = base64.b64decode(b64_string.decode())
        
        img = Image.open(io.BytesIO(img_data))
        return np.array(img)
    except Exception as e:
        print(f"Error decodificando base64: {e}")
        return None
