#!/usr/bin/env python3
"""Test compresión con casos edge."""

from processing.encryption import compress_coefficients
import numpy as np

print("TEST: Ceros preexistentes")
print("-" * 50)

data = np.arange(1, 101).reshape(10, 10).astype(float)
data[0, :5] = 0  # Poner 5 ceros manualmente
existing_zeros = (data == 0).sum()

print(f"Array original: 100 elementos")
print(f"Ceros preexistentes: {existing_zeros}")

compressed = compress_coefficients(data, 20)
final_zeros = (compressed == 0).sum()
expected_zeros = int(np.ceil(100 * 0.20))

print(f"\nCompresión: 20%")
print(f"  Ceros esperados (total): {expected_zeros}")
print(f"  Ceros reales (total): {final_zeros}")
print(f"  Resultado: {'✓ OK' if final_zeros == expected_zeros else '✗ FALLA'}")

# Verificar que los valores más pequeños fueron eliminados
original_sorted = np.sort(data[data != 0].flatten())
compressed_nonzero = compressed[compressed != 0].flatten()
print(f"\n  Valor mínimo en original (no-cero): {original_sorted[0]}")
print(f"  Valor mínimo en comprimido: {compressed_nonzero.min() if len(compressed_nonzero) > 0 else 'N/A'}")
print(f"  Valor máximo en comprimido: {compressed_nonzero.max() if len(compressed_nonzero) > 0 else 'N/A'}")

# El umbral debería ser el 20-ésimo elemento ordenado
if len(original_sorted) >= expected_zeros - existing_zeros:
    threshold = original_sorted[expected_zeros - existing_zeros - 1]
    print(f"  Umbral esperado: {threshold}")
    print(f"  ✓ Los más pequeños fueron eliminados" if compressed_nonzero.min() > threshold or len(compressed_nonzero) == 0 else "  ✗ Algo raro")
