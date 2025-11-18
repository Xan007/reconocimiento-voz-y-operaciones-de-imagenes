"""Módulo para procesamiento de imágenes con DCT 2D - Usando algoritmos de SeñalesCorte3"""
import cv2
import numpy as np
from scipy import fft
from dataclasses import dataclass


@dataclass
class ResultadoCompresionImagen:
    """Resultado del procesamiento de compresión de imagen"""
    porcentaje_compresion: float
    imagen_color_reconstruida: np.ndarray
    imagen_gris_reconstruida: np.ndarray
    dct_visual: np.ndarray
    coeficientes_conservados: int
    coeficientes_originales: np.ndarray = None
    coeficientes_filtrados: np.ndarray = None


def aplicar_algoritmo_ceros(coeficientes: np.ndarray, conservar: int) -> np.ndarray:
    """
    Anula coeficientes de menor magnitud preservando posiciones.
    Algoritmo del proyecto SeñalesCorte3.
    """
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


def load_image(image_path, max_size=512):
    """
    Carga una imagen y la redimensiona si es necesario.
    
    Args:
        image_path: ruta a la imagen
        max_size: tamaño máximo de la imagen
    
    Returns:
        imagen en RGB y escala de grises
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("No se pudo cargar la imagen")
    
    # Convertir BGR a RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionar si es muy grande
    h, w = img_rgb.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Convertir a escala de grises
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    return img_rgb, img_gray


def comprimir_imagen(imagen_rgb: np.ndarray, imagen_gris: np.ndarray, porcentaje: float) -> ResultadoCompresionImagen:
    """
    Comprime imagen usando DCT 2D - Algoritmo del proyecto SeñalesCorte3.
    
    Args:
        imagen_rgb: imagen en color RGB
        imagen_gris: imagen en escala de grises
        porcentaje: porcentaje de compresión (0-100)
    
    Returns:
        ResultadoCompresionImagen con imágenes reconstruidas
    """
    imagen_color = imagen_rgb.astype(np.float32)
    canales_reconstruidos = []
    coeficientes_conservados = 0
    
    # Procesar cada canal RGB
    for indice in range(imagen_color.shape[2]):
        canal = imagen_color[..., indice]
        coeficientes = fft.dctn(canal, type=2, norm="ortho")
        total_coeficientes = coeficientes.size
        eliminar = int(total_coeficientes * (porcentaje / 100.0))
        eliminar = min(eliminar, total_coeficientes - 1) if total_coeficientes > 1 else 0
        conservar = total_coeficientes - eliminar
        conservar = max(1, conservar)
        filtrados = aplicar_algoritmo_ceros(coeficientes, conservar)
        reconstruido = fft.idctn(filtrados, type=2, norm="ortho")
        canales_reconstruidos.append(reconstruido)
        coeficientes_conservados = conservar
    
    imagen_color_reconstruida = np.stack(canales_reconstruidos, axis=-1)
    imagen_color_reconstruida = np.clip(imagen_color_reconstruida, 0.0, 255.0).astype(np.float32)

    # Procesar imagen en escala de grises
    imagen_gris_float = imagen_gris.astype(np.float32)
    coeficientes_gris = fft.dctn(imagen_gris_float, type=2, norm="ortho")
    total_gris = coeficientes_gris.size
    eliminar_gris = int(total_gris * (porcentaje / 100.0))
    eliminar_gris = min(eliminar_gris, total_gris - 1) if total_gris > 1 else 0
    conservar_gris = total_gris - eliminar_gris
    conservar_gris = max(1, conservar_gris)
    coeficientes_gris_filtrados = aplicar_algoritmo_ceros(coeficientes_gris, conservar_gris)
    imagen_gris_reconstruida = fft.idctn(coeficientes_gris_filtrados, type=2, norm="ortho")
    imagen_gris_reconstruida = np.clip(imagen_gris_reconstruida, 0.0, 255.0).astype(np.float32)

    # Visualización de magnitud DCT
    magnitud = np.abs(coeficientes_gris_filtrados)
    magnitud = np.log1p(magnitud)
    maximo = np.max(magnitud)
    if maximo > 0:
        magnitud = magnitud / maximo
    dct_visual = magnitud

    return ResultadoCompresionImagen(
        porcentaje_compresion=porcentaje,
        imagen_color_reconstruida=imagen_color_reconstruida,
        imagen_gris_reconstruida=imagen_gris_reconstruida,
        dct_visual=dct_visual,
        coeficientes_conservados=conservar_gris,
        coeficientes_originales=coeficientes_gris,
        coeficientes_filtrados=coeficientes_gris_filtrados,
    )


def process_image_full_pipeline(image_path, filter_percent=0):
    """
    Ejecuta el pipeline completo de procesamiento de imagen usando algoritmos de SeñalesCorte3.
    
    Args:
        image_path: ruta a la imagen
        filter_percent: porcentaje de compresión (0-100)
    
    Returns:
        dict con:
            - rgb: imagen original en color
            - grayscale: imagen en escala de grises
            - resultado_compresion: ResultadoCompresionImagen con imágenes reconstruidas
            - shape: dimensiones originales
            - coeficientes_conservados: número de coeficientes mantenidos
    """
    # Paso 1: Cargar imagen
    img_rgb, img_gray = load_image(image_path)
    
    # Paso 2: Comprimir imagen
    resultado = comprimir_imagen(img_rgb, img_gray, filter_percent)
    
    return {
        'rgb': img_rgb,
        'grayscale': img_gray,
        'resultado_compresion': resultado,
        'shape': img_gray.shape,
        'filter_percent': filter_percent,
        'coeficientes_conservados': resultado.coeficientes_conservados,
        'imagen_color_reconstruida': resultado.imagen_color_reconstruida,
        'imagen_gris_reconstruida': resultado.imagen_gris_reconstruida,
        'dct_visual': resultado.dct_visual
    }


if __name__ == '__main__':
    # Test
    pass
