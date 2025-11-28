"""
encryption.py

Pipeline de encriptación y desencriptación de imágenes RGB usando FrDCT, DOST y Arnold.

🔐 Algorithm 5: Encryption Pseudocode by employing FrDCT and DOST
    Step-1: The RGB colored image is first divided into three planes R, G and B respectively
    Step-2: All three R, G and B components are transformed by employing FrDCT with 
            fractional order α1, α2 and α3 respectively
    Step-3: The output obtained after FrDCT is transformed by employing DOST at each plane
    Step-3.5: Apply compression (set smallest X% of coefficients to 0) [NEW]
    Step-4: Apply Arnold Transform with chaotic parameter (a) and iterations (k) for encryption
    Step-5: The final output matrices are concatenated to obtain the RGB encrypted image

🔓 Algorithm 6: Decryption Pseudocode (inverse process)
    Step-1: The RGB encrypted image is first divided into three different planes R, G and B
    Step-2: Apply inverse Arnold Transform with the same parameters (a, k)
    Step-3: All three R, G and B components are inverse transformed by employing inverse DOST
    Step-4: The output is inverse transformed by employing inverse FrDCT with -α1, -α2, -α3
    Step-5: The final output matrices are concatenated to obtain the RGB decrypted image
"""

import numpy as np
import io
import base64
import time
from typing import Dict, Tuple, Optional

# Import PIL
try:
    from PIL import Image
except ImportError:
    Image = None

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'static', 'lib'))

from frdct import frdct2d, ifrdct2d
from .dost import dost_2d, idost_2d
from .arnold import arnold_transform, arnold_inverse


def compress_coefficients(data: np.ndarray, compression_percent: int) -> np.ndarray:
    """
    Aplica compresión a los coeficientes de la transformada.
    
    Algoritmo:
    1. Calcula cuántos ceros necesita (X% del total de coeficientes)
    2. Cuenta ceros existentes en la matriz
    3. Si faltan ceros, agrega más quitando los coeficientes más cercanos a 0
    
    Parameters:
        data: Matriz de coeficientes (puede ser compleja)
        compression_percent: Porcentaje de coeficientes a convertir en 0 (0-99)
        
    Returns:
        Matriz con X% de los coeficientes más pequeños puestos a 0
    """
    if compression_percent <= 0:
        return data.copy()
    
    if compression_percent >= 100:
        return np.zeros_like(data)
    
    # Paso 1: Calcular cuántos ceros necesitamos (X% del total)
    total_elements = data.size
    target_zeros = int(np.ceil(total_elements * compression_percent / 100))
    
    # Paso 2: Contar ceros existentes
    magnitudes = np.abs(data)
    existing_zeros = np.sum(magnitudes == 0)
    
    # Paso 3: Si faltan ceros, agregar eliminando los más cercanos a 0
    if existing_zeros < target_zeros:
        # Cantidad de ceros adicionales que necesitamos
        zeros_needed = target_zeros - existing_zeros
        
        # Obtener los índices ordenados por magnitud (de menor a mayor)
        # Excluir los que ya son 0
        nonzero_mask = magnitudes > 0
        nonzero_mags = magnitudes[nonzero_mask]
        
        # Encontrar el umbral: el valor más pequeño de los que queremos poner a 0
        # Ordenamos de menor a mayor y tomamos el índice zeros_needed
        if len(nonzero_mags) > 0:
            sorted_nonzero_mags = np.sort(nonzero_mags.flatten())
            if zeros_needed <= len(sorted_nonzero_mags):
                threshold = sorted_nonzero_mags[zeros_needed - 1]
            else:
                threshold = sorted_nonzero_mags[-1]
            
            # Poner a 0 todos los coeficientes con magnitud <= umbral
            mask = magnitudes <= threshold
            compressed = np.where(mask, 0, data)
        else:
            compressed = data.copy()
    else:
        # Ya hay suficientes ceros (o más), no hacemos nada
        compressed = data.copy()
    
    return compressed


class EncryptionResult:
    """Resultado de la encriptación RGB con todas las etapas intermedias."""
    
    def __init__(self):
        self.original: Optional[np.ndarray] = None
        # Después de FrDCT
        self.after_frdct_r: Optional[np.ndarray] = None
        self.after_frdct_g: Optional[np.ndarray] = None
        self.after_frdct_b: Optional[np.ndarray] = None
        # Después de DOST
        self.after_dost_r: Optional[np.ndarray] = None
        self.after_dost_g: Optional[np.ndarray] = None
        self.after_dost_b: Optional[np.ndarray] = None
        # Después de Arnold (matriz cifrada final)
        self.encrypted_r: Optional[np.ndarray] = None
        self.encrypted_g: Optional[np.ndarray] = None
        self.encrypted_b: Optional[np.ndarray] = None
        self.encrypted_rgb: Optional[np.ndarray] = None
        
        # Parámetros usados
        self.alpha_r: float = 0.5
        self.alpha_g: float = 0.5
        self.alpha_b: float = 0.5
        self.arnold_a: int = 1
        self.arnold_k: int = 1
        
    def to_dict(self) -> Dict:
        """Convierte el resultado a diccionario con imágenes en base64."""
        return {
            'original': array_to_base64_rgb(self.original),
            'after_frdct': array_to_base64_rgb_from_channels(
                self.after_frdct_r, self.after_frdct_g, self.after_frdct_b, is_complex=True, use_log_scale=True
            ),
            'after_frdct_channels': {
                'r': channel_to_base64(self.after_frdct_r, is_complex=True, use_log_scale=True),
                'g': channel_to_base64(self.after_frdct_g, is_complex=True, use_log_scale=True),
                'b': channel_to_base64(self.after_frdct_b, is_complex=True, use_log_scale=True)
            },
            'after_dost': array_to_base64_rgb_from_channels(
                self.after_dost_r, self.after_dost_g, self.after_dost_b, is_complex=True, use_log_scale=True
            ),
            'after_dost_channels': {
                'r': channel_to_base64(self.after_dost_r, is_complex=True, use_log_scale=True),
                'g': channel_to_base64(self.after_dost_g, is_complex=True, use_log_scale=True),
                'b': channel_to_base64(self.after_dost_b, is_complex=True, use_log_scale=True)
            },
            'encrypted': array_to_base64_rgb_from_channels(
                self.encrypted_r, self.encrypted_g, self.encrypted_b, is_complex=True
            ),
            'encrypted_channels': {
                'r': channel_to_base64(self.encrypted_r, is_complex=True, use_log_scale=False),
                'g': channel_to_base64(self.encrypted_g, is_complex=True, use_log_scale=False),
                'b': channel_to_base64(self.encrypted_b, is_complex=True, use_log_scale=False)
            },
            'params': {
                'alpha_r': self.alpha_r,
                'alpha_g': self.alpha_g,
                'alpha_b': self.alpha_b,
                'arnold_a': self.arnold_a,
                'arnold_k': self.arnold_k
            }
        }


class DecryptionResult:
    """Resultado de la desencriptación RGB con todas las etapas intermedias."""
    
    def __init__(self):
        self.encrypted: Optional[np.ndarray] = None
        # Step 1: Después de Arnold inverso
        self.after_arnold_inv_r: Optional[np.ndarray] = None
        self.after_arnold_inv_g: Optional[np.ndarray] = None
        self.after_arnold_inv_b: Optional[np.ndarray] = None
        # Step 2: Después de IDOST
        self.after_idost_r: Optional[np.ndarray] = None
        self.after_idost_g: Optional[np.ndarray] = None
        self.after_idost_b: Optional[np.ndarray] = None
        # Step 3: Después de IFrDCT (imagen descifrada)
        self.after_ifrdct_r: Optional[np.ndarray] = None
        self.after_ifrdct_g: Optional[np.ndarray] = None
        self.after_ifrdct_b: Optional[np.ndarray] = None
        self.decrypted_rgb: Optional[np.ndarray] = None
        
        # Parámetros usados
        self.alpha_r: float = 0.5
        self.alpha_g: float = 0.5
        self.alpha_b: float = 0.5
        self.arnold_a: int = 1
        self.arnold_k: int = 1
        
    def to_dict(self) -> Dict:
        """Convierte el resultado a diccionario con imágenes en base64."""
        return {
            'encrypted': array_to_base64_rgb(self.encrypted),
            'after_arnold_inv': array_to_base64_rgb_from_channels(
                self.after_arnold_inv_r, self.after_arnold_inv_g, self.after_arnold_inv_b, is_complex=True
            ),
            'after_idost': array_to_base64_rgb_from_channels(
                self.after_idost_r, self.after_idost_g, self.after_idost_b, is_complex=True, use_log_scale=True
            ),
            'after_ifrdct': array_to_base64_rgb_from_channels(
                self.after_ifrdct_r, self.after_ifrdct_g, self.after_ifrdct_b, is_complex=True, use_log_scale=True
            ),
            'decrypted': array_to_base64_rgb(self.decrypted_rgb),
            'params': {
                'alpha_r': self.alpha_r,
                'alpha_g': self.alpha_g,
                'alpha_b': self.alpha_b,
                'arnold_a': self.arnold_a,
                'arnold_k': self.arnold_k
            }
        }


def array_to_base64_rgb(arr: Optional[np.ndarray]) -> Optional[str]:
    """Convierte array RGB numpy a base64."""
    if arr is None:
        return None
    
    try:
        if Image is None:
            return None
        
        # Normalizar a 0-255
        arr_normalized = arr.copy().astype(np.float64)
        arr_min, arr_max = arr_normalized.min(), arr_normalized.max()
        if arr_max - arr_min > 1e-10:
            arr_normalized = (arr_normalized - arr_min) / (arr_max - arr_min) * 255
        arr_normalized = np.clip(arr_normalized, 0, 255).astype(np.uint8)
        
        pil_img = Image.fromarray(arr_normalized, mode='RGB')
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        buffer.seek(0)
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f'data:image/png;base64,{b64}'
    except Exception as e:
        print(f"Error en array_to_base64_rgb: {e}")
        return None


def normalize_channel_linear(ch):
    """Normalización lineal a [0, 255] (para imágenes originales/descifradas)."""
    if np.iscomplexobj(ch):
        ch = np.abs(ch)
    ch = ch.astype(np.float64)
    ch_min, ch_max = ch.min(), ch.max()
    if ch_max - ch_min > 1e-10:
        ch = (ch - ch_min) / (ch_max - ch_min) * 255
    else:
        ch = np.zeros_like(ch)
    return ch.astype(np.uint8)


def normalize_channel_logarithmic(ch):
    """
    Normalización logarítmica a [0, 255] (para FrDCT y DOST).
    Usa log10(1 + magnitude) para mejor visualización de datos transformados.
    """
    if np.iscomplexobj(ch):
        ch = np.abs(ch)
    ch = ch.astype(np.float64)
    
    # Aplicar log10 a la magnitud (sumando 1 para evitar log(0))
    ch_log = np.log10(np.maximum(ch, 1e-10))
    
    # Normalizar el resultado logarítmico a [0, 255]
    log_min = ch_log.min()
    log_max = ch_log.max()
    if log_max - log_min > 1e-10:
        ch_norm = (ch_log - log_min) / (log_max - log_min) * 255
    else:
        ch_norm = np.zeros_like(ch_log)
    
    return np.clip(ch_norm, 0, 255).astype(np.uint8)


def channel_to_base64(ch: Optional[np.ndarray], 
                      is_complex: bool = False,
                      use_log_scale: bool = False) -> Optional[str]:
    """
    Convierte un canal individual a imagen grayscale base64.
    
    Parameters:
        ch: Canal de entrada
        is_complex: Si True, toma la magnitud
        use_log_scale: Si True, usa escala logarítmica
    """
    if ch is None:
        return None
    
    try:
        if Image is None:
            return None
        
        # Elegir función de normalización
        normalize_fn = normalize_channel_logarithmic if use_log_scale else normalize_channel_linear
        
        if is_complex:
            ch_norm = normalize_fn(np.abs(ch))
        else:
            ch_norm = normalize_fn(ch)
        
        pil_img = Image.fromarray(ch_norm, mode='L')
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        buffer.seek(0)
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f'data:image/png;base64,{b64}'
    except Exception as e:
        print(f"Error en channel_to_base64: {e}")
        return None


def array_to_base64_rgb_from_channels(r: Optional[np.ndarray], 
                                       g: Optional[np.ndarray], 
                                       b: Optional[np.ndarray],
                                       is_complex: bool = False,
                                       use_log_scale: bool = False) -> Optional[str]:
    """
    Convierte 3 canales separados a una imagen RGB base64.
    
    Parameters:
        r, g, b: Canales de entrada
        is_complex: Si True, toma la magnitud
        use_log_scale: Si True, usa escala logarítmica (para FrDCT, DOST)
                       Si False, usa escala lineal (para imágenes normales)
    """
    if r is None or g is None or b is None:
        return None
    
    try:
        if Image is None:
            return None
        
        # Elegir función de normalización
        normalize_fn = normalize_channel_logarithmic if use_log_scale else normalize_channel_linear
        
        if is_complex:
            r_norm = normalize_fn(np.abs(r))
            g_norm = normalize_fn(np.abs(g))
            b_norm = normalize_fn(np.abs(b))
        else:
            r_norm = normalize_fn(r)
            g_norm = normalize_fn(g)
            b_norm = normalize_fn(b)
        
        rgb = np.stack([r_norm, g_norm, b_norm], axis=-1)
        
        pil_img = Image.fromarray(rgb, mode='RGB')
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        buffer.seek(0)
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f'data:image/png;base64,{b64}'
    except Exception as e:
        print(f"Error en array_to_base64_rgb_from_channels: {e}")
        return None


def apply_arnold_to_complex(matrix: np.ndarray, a: int, k: int, inverse: bool = False) -> np.ndarray:
    """
    Aplica Arnold Transform a una matriz compleja.
    Arnold trabaja con la magnitud/fase o parte real/imaginaria por separado.
    """
    if not np.iscomplexobj(matrix):
        if inverse:
            return arnold_inverse(matrix, a, k)
        else:
            return arnold_transform(matrix, a, k)
    
    # Para matrices complejas, aplicar Arnold a la parte real e imaginaria por separado
    real_part = np.real(matrix)
    imag_part = np.imag(matrix)
    
    if inverse:
        real_transformed = arnold_inverse(real_part.astype(np.float64), a, k)
        imag_transformed = arnold_inverse(imag_part.astype(np.float64), a, k)
    else:
        real_transformed = arnold_transform(real_part.astype(np.float64), a, k)
        imag_transformed = arnold_transform(imag_part.astype(np.float64), a, k)
    
    return real_transformed + 1j * imag_transformed


def encrypt_image_rgb(image: np.ndarray, 
                      alpha_r: float = 0.5,
                      alpha_g: float = 0.5, 
                      alpha_b: float = 0.5,
                      arnold_a: int = 1,
                      arnold_k: int = 1) -> EncryptionResult:
    """
    Encripta una imagen RGB siguiendo el Algorithm 5.
    
    Algorithm 5: Encryption by employing FrDCT, DOST and Arnold
        Step-1: Split RGB image into R, G, B planes
        Step-2: Apply FrDCT with α1, α2, α3 to each plane
        Step-3: Apply DOST to each FrDCT result
        Step-4: Apply Arnold Transform with parameters (a, k) for encryption
        Step-5: Concatenate to obtain RGB encrypted image
    
    Parameters:
        image: Imagen RGB de entrada (H x W x 3)
        alpha_r: Parámetro fraccionario para canal R [0, 2)
        alpha_g: Parámetro fraccionario para canal G [0, 2)
        alpha_b: Parámetro fraccionario para canal B [0, 2)
        arnold_a: Parámetro caótico de Arnold (a >= 1)
        arnold_k: Número de iteraciones de Arnold (k >= 1)
        
    Returns:
        EncryptionResult con todas las etapas intermedias
    """
    start_total = time.time()
    print(f"\n[ENCRYPTION START] Tamaño imagen: {image.shape}")
    
    result = EncryptionResult()
    result.alpha_r = alpha_r
    result.alpha_g = alpha_g
    result.alpha_b = alpha_b
    result.arnold_a = arnold_a
    result.arnold_k = arnold_k
    
    # Asegurar que la imagen sea RGB
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Step 1: Input image - Split into R, G, B planes
    result.original = image.astype(np.float64)
    R = result.original[:, :, 0]
    G = result.original[:, :, 1]
    B = result.original[:, :, 2]
    
    # Step 2: Apply FrDCT to each channel with respective α
    result.after_frdct_r = frdct2d(R, alpha_r)
    result.after_frdct_g = frdct2d(G, alpha_g)
    result.after_frdct_b = frdct2d(B, alpha_b)
    
    # Step 3: Apply DOST to each FrDCT result
    result.after_dost_r = dost_2d(result.after_frdct_r)
    result.after_dost_g = dost_2d(result.after_frdct_g)
    result.after_dost_b = dost_2d(result.after_frdct_b)
    
    # Step 4: Apply Arnold Transform for encryption operation E(α, S)
    result.encrypted_r = apply_arnold_to_complex(result.after_dost_r, arnold_a, arnold_k, inverse=False)
    result.encrypted_g = apply_arnold_to_complex(result.after_dost_g, arnold_a, arnold_k, inverse=False)
    result.encrypted_b = apply_arnold_to_complex(result.after_dost_b, arnold_a, arnold_k, inverse=False)
    
    # Step 5: Concatenate to obtain RGB encrypted image (for visualization)
    def normalize_to_uint8(arr):
        mag = np.abs(arr)
        mag_min, mag_max = mag.min(), mag.max()
        if mag_max - mag_min > 1e-10:
            normalized = (mag - mag_min) / (mag_max - mag_min) * 255
        else:
            normalized = np.zeros_like(mag)
        return normalized.astype(np.uint8)
    
    enc_r = normalize_to_uint8(result.encrypted_r)
    enc_g = normalize_to_uint8(result.encrypted_g)
    enc_b = normalize_to_uint8(result.encrypted_b)
    
    result.encrypted_rgb = np.stack([enc_r, enc_g, enc_b], axis=-1)
    
    elapsed_total = time.time() - start_total
    print(f"[ENCRYPTION END] Tiempo total: {elapsed_total:.4f}s\n")
    
    return result


def decrypt_image_rgb(encrypted_r: np.ndarray,
                      encrypted_g: np.ndarray,
                      encrypted_b: np.ndarray,
                      alpha_r: float = 0.5,
                      alpha_g: float = 0.5,
                      alpha_b: float = 0.5,
                      arnold_a: int = 1,
                      arnold_k: int = 1) -> DecryptionResult:
    """
    Desencripta una imagen RGB siguiendo el Algorithm 6.
    
    Algorithm 6: Decryption (inverse of Algorithm 5)
        Step-1: Split RGB encrypted image into R, G, B planes
        Step-2: Apply inverse Arnold Transform with same parameters (a, k)
        Step-3: Apply inverse DOST to each channel
        Step-4: Apply inverse FrDCT with same α values
        Step-5: Concatenate to obtain RGB decrypted image
    
    NOTA SOBRE ALPHA:
    El documento menciona "−α" pero esto se refiere conceptualmente a la "inversa".
    Matemáticamente, según las ecuaciones 1.14 y 1.15:
    - FrDCT: F^α(u,v) usa cos[π/N * (u + α/2)(x + 1/2)]
    - IFrDCT: f(x,y) TAMBIÉN usa cos[π/N * (u + α/2)(x + 1/2)] (MISMO kernel)
    Por lo tanto, usamos el MISMO α (no negativo) para la desencriptación.
    Esto es consistente con la naturaleza ortogonal de la matriz FrDCT.
    
    Parameters:
        encrypted_r, encrypted_g, encrypted_b: Matrices cifradas (complejas)
        alpha_r, alpha_g, alpha_b: Parámetros fraccionarios usados en cifrado
        arnold_a: Parámetro caótico de Arnold usado en cifrado
        arnold_k: Número de iteraciones de Arnold usado en cifrado
        
    Returns:
        DecryptionResult con todas las etapas intermedias
    """
    start_total = time.time()
    print(f"\n[DECRYPTION START] Tamaño imagen: {encrypted_r.shape}")
    
    result = DecryptionResult()
    result.alpha_r = alpha_r
    result.alpha_g = alpha_g
    result.alpha_b = alpha_b
    result.arnold_a = arnold_a
    result.arnold_k = arnold_k
    
    # Asegurar que los datos sean complejos
    enc_r = encrypted_r.astype(np.complex128) if not np.iscomplexobj(encrypted_r) else encrypted_r
    enc_g = encrypted_g.astype(np.complex128) if not np.iscomplexobj(encrypted_g) else encrypted_g
    enc_b = encrypted_b.astype(np.complex128) if not np.iscomplexobj(encrypted_b) else encrypted_b
    
    # Crear imagen cifrada para visualización
    def normalize_to_uint8(arr):
        mag = np.abs(arr)
        mag_min, mag_max = mag.min(), mag.max()
        if mag_max - mag_min > 1e-10:
            normalized = (mag - mag_min) / (mag_max - mag_min) * 255
        else:
            normalized = np.zeros_like(mag)
        return normalized.astype(np.uint8)
    
    result.encrypted = np.stack([
        normalize_to_uint8(enc_r),
        normalize_to_uint8(enc_g),
        normalize_to_uint8(enc_b)
    ], axis=-1)
    
    # Step 2: Apply inverse Arnold Transform (decryption operation)
    result.after_arnold_inv_r = apply_arnold_to_complex(enc_r, arnold_a, arnold_k, inverse=True)
    result.after_arnold_inv_g = apply_arnold_to_complex(enc_g, arnold_a, arnold_k, inverse=True)
    result.after_arnold_inv_b = apply_arnold_to_complex(enc_b, arnold_a, arnold_k, inverse=True)
    
    # Step 3: Apply inverse DOST to each channel
    result.after_idost_r = idost_2d(result.after_arnold_inv_r.copy())
    result.after_idost_g = idost_2d(result.after_arnold_inv_g.copy())
    result.after_idost_b = idost_2d(result.after_arnold_inv_b.copy())
    
    # Step 4: Apply inverse FrDCT with the same α
    start_ifrdct_r = time.time()
    result.after_ifrdct_r = ifrdct2d(result.after_idost_r, alpha_r)
    elapsed_ifrdct_r = time.time() - start_ifrdct_r
    print(f"[IFrDCT 2D] α={alpha_r:.2f} Tamaño={result.after_idost_r.shape[0]}x{result.after_idost_r.shape[1]} Tiempo: {elapsed_ifrdct_r:.4f}s")
    
    start_ifrdct_g = time.time()
    result.after_ifrdct_g = ifrdct2d(result.after_idost_g, alpha_g)
    elapsed_ifrdct_g = time.time() - start_ifrdct_g
    print(f"[IFrDCT 2D] α={alpha_g:.2f} Tamaño={result.after_idost_g.shape[0]}x{result.after_idost_g.shape[1]} Tiempo: {elapsed_ifrdct_g:.4f}s")
    
    start_ifrdct_b = time.time()
    result.after_ifrdct_b = ifrdct2d(result.after_idost_b, alpha_b)
    elapsed_ifrdct_b = time.time() - start_ifrdct_b
    print(f"[IFrDCT 2D] α={alpha_b:.2f} Tamaño={result.after_idost_b.shape[0]}x{result.after_idost_b.shape[1]} Tiempo: {elapsed_ifrdct_b:.4f}s")
    
    # Step 5: Concatenate to obtain RGB decrypted image
    def normalize_channel(ch):
        """
        Asegura rango válido [0, 255] para la imagen final.
        """
        ch_real = np.real(ch).astype(np.float64)
        return np.clip(ch_real, 0, 255).astype(np.uint8)
    
    dec_r = normalize_channel(result.after_ifrdct_r)
    dec_g = normalize_channel(result.after_ifrdct_g)
    dec_b = normalize_channel(result.after_ifrdct_b)
    
    result.decrypted_rgb = np.stack([dec_r, dec_g, dec_b], axis=-1)
    
    elapsed_total = time.time() - start_total
    print(f"[DECRYPTION END] Tiempo total: {elapsed_total:.4f}s\n")
    
    return result


# ============== Funciones de compatibilidad ==============

def encrypt_image(image: np.ndarray, alpha: float, arnold_a: int = 1, arnold_k: int = 1) -> EncryptionResult:
    """Función de compatibilidad - usa el mismo alpha para los 3 canales."""
    return encrypt_image_rgb(image, alpha, alpha, alpha, arnold_a, arnold_k)


def decrypt_image(encrypted: np.ndarray, alpha: float, arnold_a: int = 1, arnold_k: int = 1) -> DecryptionResult:
    """Función de compatibilidad para desencriptar."""
    if isinstance(encrypted, dict):
        return decrypt_image_rgb(
            encrypted['r'], encrypted['g'], encrypted['b'],
            alpha, alpha, alpha, arnold_a, arnold_k
        )
    
    if len(encrypted.shape) == 3:
        r = encrypted[:, :, 0].astype(np.complex128)
        g = encrypted[:, :, 1].astype(np.complex128)
        b = encrypted[:, :, 2].astype(np.complex128)
        return decrypt_image_rgb(r, g, b, alpha, alpha, alpha, arnold_a, arnold_k)
    
    return decrypt_image_rgb(encrypted, encrypted, encrypted, alpha, alpha, alpha, arnold_a, arnold_k)


def save_encrypted_data(result: EncryptionResult, filepath: str) -> None:
    """Guarda los datos cifrados completos (complejos) en un archivo .npz."""
    enc_r = result.encrypted_r if result.encrypted_r is not None else np.array([])
    enc_g = result.encrypted_g if result.encrypted_g is not None else np.array([])
    enc_b = result.encrypted_b if result.encrypted_b is not None else np.array([])
    
    np.savez_compressed(
        filepath,
        encrypted_r_real=np.real(enc_r),
        encrypted_r_imag=np.imag(enc_r),
        encrypted_g_real=np.real(enc_g),
        encrypted_g_imag=np.imag(enc_g),
        encrypted_b_real=np.real(enc_b),
        encrypted_b_imag=np.imag(enc_b),
        alpha_r=result.alpha_r,
        alpha_g=result.alpha_g,
        alpha_b=result.alpha_b,
        arnold_a=result.arnold_a,
        arnold_k=result.arnold_k
    )


def load_encrypted_data(filepath: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Carga datos cifrados desde un archivo .npz."""
    data = np.load(filepath)
    
    encrypted = {
        'r': data['encrypted_r_real'] + 1j * data['encrypted_r_imag'],
        'g': data['encrypted_g_real'] + 1j * data['encrypted_g_imag'],
        'b': data['encrypted_b_real'] + 1j * data['encrypted_b_imag']
    }
    
    params = {
        'alpha_r': float(data['alpha_r']),
        'alpha_g': float(data['alpha_g']),
        'alpha_b': float(data['alpha_b']),
        'arnold_a': int(data['arnold_a']),
        'arnold_k': int(data['arnold_k'])
    }
    
    return encrypted, params


# ============== Compresión de imagen con DCT ==============

from scipy import fft as scipy_fft


def aplicar_algoritmo_ceros(coeficientes: np.ndarray, conservar: int) -> np.ndarray:
    """Anula coeficientes de menor magnitud preservando posiciones."""
    plano = coeficientes.reshape(-1)
    total = plano.size
    ceros_por_aplicar = max(0, total - conservar)
    if ceros_por_aplicar == 0:
        return coeficientes.copy()

    resultado = plano.copy()
    indices_ordenados = np.argsort(np.abs(resultado))
    cantidad_a_convertir = ceros_por_aplicar - 1
    posicion_umbral = min(cantidad_a_convertir + 1, total - 1)
    valor_umbral = np.abs(resultado[indices_ordenados[posicion_umbral]])

    indice = 0
    ceros_colocados = 0
    while ceros_colocados < ceros_por_aplicar and indice < total:
        indice_actual = indices_ordenados[indice]
        valor_actual = np.abs(resultado[indice_actual])
        if valor_actual == valor_umbral:
            j = indice
            while (
                j + 1 < total
                and np.abs(resultado[indices_ordenados[j]]) == np.abs(resultado[indices_ordenados[j + 1]])
            ):
                j += 1
            k = indice
            while k <= j and ceros_colocados < ceros_por_aplicar:
                resultado[indices_ordenados[k]] = 0.0
                ceros_colocados += 1
                k += 1
            indice = j + 1
        else:
            resultado[indice_actual] = 0.0
            ceros_colocados += 1
            indice += 1

    return resultado.reshape(coeficientes.shape)


def comprimir_imagen_dct(imagen_rgb: np.ndarray, porcentaje: float) -> Dict:
    """
    Comprime una imagen RGB usando DCT-2D.
    
    Proceso:
    1. Aplica DCT-2D a cada canal RGB
    2. Elimina el X% de los coeficientes más pequeños
    3. Aplica IDCT-2D para reconstruir
    
    Parameters:
        imagen_rgb: Imagen RGB (H x W x 3) como uint8 o float
        porcentaje: Porcentaje de coeficientes a eliminar (0-100)
        
    Returns:
        Dict con imagen comprimida y coeficientes conservados
    """
    imagen_color = imagen_rgb.astype(np.float32)
    canales_reconstruidos = []
    coeficientes_conservados = 0
    
    for indice in range(imagen_color.shape[2]):
        canal = imagen_color[..., indice]
        coeficientes = scipy_fft.dctn(canal, type=2, norm="ortho")
        total_coeficientes = coeficientes.size
        eliminar = int(total_coeficientes * (porcentaje / 100.0))
        eliminar = min(eliminar, total_coeficientes - 1) if total_coeficientes > 1 else 0
        conservar = total_coeficientes - eliminar
        conservar = max(1, conservar)
        filtrados = aplicar_algoritmo_ceros(coeficientes, conservar)
        reconstruido = scipy_fft.idctn(filtrados, type=2, norm="ortho")
        canales_reconstruidos.append(reconstruido)
        coeficientes_conservados = conservar
    
    imagen_reconstruida = np.stack(canales_reconstruidos, axis=-1)
    imagen_reconstruida = np.clip(imagen_reconstruida, 0.0, 255.0).astype(np.uint8)
    
    return {
        'imagen': imagen_reconstruida,
        'coeficientes_conservados': coeficientes_conservados,
        'porcentaje': porcentaje
    }


def comprimir_imagen_multiples_niveles(imagen_rgb: np.ndarray) -> Dict:
    """
    Comprime una imagen a múltiples niveles (30%, 50%, 80%) y retorna todas las versiones.
    
    Parameters:
        imagen_rgb: Imagen RGB de entrada
        
    Returns:
        Dict con la imagen original y las versiones comprimidas como base64
    """
    # Asegurar que la imagen sea RGB uint8
    if imagen_rgb.dtype != np.uint8:
        imagen_rgb = np.clip(imagen_rgb, 0, 255).astype(np.uint8)
    
    if len(imagen_rgb.shape) == 2:
        imagen_rgb = np.stack([imagen_rgb, imagen_rgb, imagen_rgb], axis=-1)
    elif imagen_rgb.shape[2] == 4:
        imagen_rgb = imagen_rgb[:, :, :3]
    
    resultados = {
        'original': array_to_base64_rgb(imagen_rgb),
        'compressed_30': None,
        'compressed_50': None,
        'compressed_80': None
    }
    
    # Comprimir al 30%
    comp_30 = comprimir_imagen_dct(imagen_rgb, 30)
    resultados['compressed_30'] = array_to_base64_rgb(comp_30['imagen'])
    
    # Comprimir al 50%
    comp_50 = comprimir_imagen_dct(imagen_rgb, 50)
    resultados['compressed_50'] = array_to_base64_rgb(comp_50['imagen'])
    
    # Comprimir al 80%
    comp_80 = comprimir_imagen_dct(imagen_rgb, 80)
    resultados['compressed_80'] = array_to_base64_rgb(comp_80['imagen'])
    
    return resultados
