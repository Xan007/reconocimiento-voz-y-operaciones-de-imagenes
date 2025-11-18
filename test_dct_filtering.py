"""Test para verificar que el filtrado DCT está funcionando correctamente"""
import numpy as np
from processing.image_processing import apply_dct_threshold, dct_2d, idct_2d
import cv2

# Crear una imagen de prueba simple (patrón)
img = np.ones((256, 256), dtype=np.uint8) * 128

# Agregar algunos patrones simples
img[50:100, 50:100] = 255  # Cuadrado blanco
img[150:200, 150:200] = 0   # Cuadrado negro

# Calcular DCT
dct_coeff = dct_2d(img)
print(f"Total de coeficientes DCT: {dct_coeff.size}")
print(f"Rango de valores DCT: [{np.min(dct_coeff):.4f}, {np.max(dct_coeff):.4f}]")

# Probar diferentes porcentajes de filtrado
percentages = [100, 50, 20, 10, 5]

for pct in percentages:
    dct_filtered = apply_dct_threshold(dct_coeff, pct)
    
    # Contar coeficientes no cero
    non_zero = np.count_nonzero(dct_filtered)
    actual_percent = (non_zero / dct_filtered.size) * 100
    
    # Reconstruir imagen
    img_reconstructed = idct_2d(dct_filtered)
    
    # Calcular diferencia
    diff = np.abs(img.astype(float) - img_reconstructed.astype(float))
    mse = np.mean(diff ** 2)
    
    print(f"\n{pct}% coeficientes:")
    print(f"  Coeficientes mantenidos: {non_zero}/{dct_filtered.size} ({actual_percent:.2f}%)")
    print(f"  MSE (error): {mse:.4f}")
    print(f"  Diferencia máxima: {np.max(diff):.2f}")
    print(f"  Diferencia promedio: {np.mean(diff):.2f}")

print("\n✓ Test completado")
