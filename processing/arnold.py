"""
arnold.py

Implementación de la Transformación de Arnold para cifrado de imágenes.
USANDO EXCLUSIVAMENTE FORMA MATRICIAL (solo imágenes cuadradas).

Basado en el documento - Sección 9 (Transformación de Arnold):

📌 FORMA MATRICIAL DIRECTA (Ecuación 1.26):
    [x']   [1    1  ] [x]
    [y'] = [a  a+1  ] [y]  mod N

📌 FORMA MATRICIAL INVERSA (Ecuación 1.27):
    [x]   [a+1  -1] [x']
    [y] = [-a    1] [y']  mod N

⚠️ NOTA: Solo se soportan imágenes CUADRADAS (N x N).
Si la imagen no es cuadrada, debe ser recortada antes de usar.

La clave de cifrado es (a, k) donde:
- a: parámetro de la matriz de Arnold (a >= 1)
- k: número de iteraciones (k >= 1)
"""

import numpy as np


def arnold_transform(image: np.ndarray, a: int = 1, iterations: int = 1) -> np.ndarray:
    """
    Aplica la Transformación de Arnold a imagen CUADRADA usando forma MATRICIAL.
    
    Ecuación 1.26 (VECTORIZADO para velocidad):
    [x']   [1    1  ] [x]
    [y'] = [a  a+1  ] [y]  mod N
    
    Parameters:
        image: Imagen cuadrada (N x N)
        a: Parámetro de la transformación (a >= 1)
        iterations: Número de aplicaciones (k >= 1)
        
    Returns:
        Imagen transformada
        
    Raises:
        ValueError: Si la imagen no es cuadrada
    """
    N = image.shape[0]
    if image.shape[0] != image.shape[1]:
        raise ValueError(f"Arnold Transform requiere imagen cuadrada (N x N). Recibida: {image.shape[0]} x {image.shape[1]}")
    
    result = image.copy()
    
    for _ in range(iterations):
        # Generar todas las coordenadas (x, y) de una vez
        y_coords, x_coords = np.meshgrid(np.arange(N), np.arange(N))
        
        # Aplicar transformación matricial: [x', y'] = A @ [x, y] mod N
        # x' = (1*x + 1*y) mod N = (x + y) mod N
        # y' = (a*x + (a+1)*y) mod N
        x_new = (x_coords + y_coords) % N
        y_new = (a * x_coords + (a + 1) * y_coords) % N
        
        # Crear nueva imagen permutando índices
        new_image = np.zeros_like(result)
        new_image[x_new, y_new] = result[x_coords, y_coords]
        result = new_image
    
    return result


def arnold_inverse(image: np.ndarray, a: int = 1, iterations: int = 1) -> np.ndarray:
    """
    Aplica la Transformación de Arnold INVERSA a imagen CUADRADA usando forma MATRICIAL.
    
    Ecuación 1.27 (VECTORIZADO para velocidad):
    [x]   [a+1  -1] [x']
    [y] = [-a    1] [y']  mod N
    
    Parameters:
        image: Imagen cifrada (N x N)
        a: Parámetro de Arnold usado en cifrado (a >= 1)
        iterations: Número de iteraciones usado en cifrado (k >= 1)
        
    Returns:
        Imagen descifrada
        
    Raises:
        ValueError: Si la imagen no es cuadrada
    """
    N = image.shape[0]
    if image.shape[0] != image.shape[1]:
        raise ValueError(f"Arnold Transform Inversa requiere imagen cuadrada (N x N). Recibida: {image.shape[0]} x {image.shape[1]}")
    
    result = image.copy()
    
    for _ in range(iterations):
        # Generar todas las coordenadas (x', y') de una vez
        y_prime_coords, x_prime_coords = np.meshgrid(np.arange(N), np.arange(N))
        
        # Aplicar transformación inversa matricial: [x, y] = A_inv @ [x', y'] mod N
        # x = ((a+1)*x' - 1*y') mod N = ((a+1)*x' - y') mod N
        # y = (-a*x' + 1*y') mod N
        x = ((a + 1) * x_prime_coords - y_prime_coords) % N
        y = (-a * x_prime_coords + y_prime_coords) % N
        
        # Crear nueva imagen permutando índices
        new_image = np.zeros_like(result)
        new_image[x, y] = result[x_prime_coords, y_prime_coords]
        result = new_image
    
    return result


def arnold_period(N: int, a: int = 1) -> int:
    """
    Calcula el período de la Transformación de Arnold para una imagen NxN.
    
    El período es el número de iteraciones necesarias para volver
    a la imagen original.
    
    Parameters:
        N: Tamaño de la imagen cuadrada
        a: Parámetro de la transformación
        
    Returns:
        Período de la transformación
    """
    # Crear una imagen de prueba con valores únicos
    test = np.arange(N * N).reshape(N, N)
    original = test.copy()
    
    period = 1
    transformed = arnold_transform(test, a, 1)
    
    while not np.array_equal(transformed, original) and period < N * N:
        transformed = arnold_transform(transformed, a, 1)
        period += 1
    
    return period


def is_square_image(image: np.ndarray) -> bool:
    """
    Verifica si una imagen es cuadrada.
    
    Parameters:
        image: Imagen a verificar (2D o 3D con canales)
        
    Returns:
        True si es cuadrada, False si no
    """
    if len(image.shape) == 2:
        return image.shape[0] == image.shape[1]
    elif len(image.shape) == 3:
        return image.shape[0] == image.shape[1]
    return False
