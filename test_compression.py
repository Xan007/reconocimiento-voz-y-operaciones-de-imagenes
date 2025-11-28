#!/usr/bin/env python3
"""
Script para verificar que la compresión funciona como se espera:
- compression_percent = 20 significa poner el 20% de coeficientes a 0
- Se cuentan los ceros existentes y se agregan los que faltan
- Los coeficientes eliminados son los más cercanos a 0 (menor magnitud)
"""

import numpy as np
from processing.encryption import compress_coefficients

# Crear un array de prueba con valores conocidos
print("=" * 70)
print("TEST 1: Array simple con valores reales")
print("=" * 70)

# 100 elementos con valores de 1 a 100
data = np.arange(1, 101, dtype=np.float64).reshape(10, 10)
print(f"\nDatos originales: {data.size} elementos (1 a 100)")
print(f"Primeros 5 valores: {data.flatten()[:5]}")
print(f"Últimos 5 valores: {data.flatten()[-5:]}")

# Comprimir al 20%
compressed = compress_coefficients(data, compression_percent=20)
zeros_count = np.sum(compressed == 0)
nonzero_count = np.sum(compressed != 0)

print(f"\nCompresión: 20%")
print(f"  Ceros esperados: {int(np.ceil(data.size * 0.20))} ({20}%)")
print(f"  Ceros reales: {zeros_count}")
print(f"  Valores != 0: {nonzero_count}")
print(f"  ✓ CORRECTO" if zeros_count == int(np.ceil(data.size * 0.20)) else f"  ✗ ERROR")

# Verificar que los valores puestos a 0 son los más pequeños
original_nonzero = data[data != 0]
compressed_nonzero = compressed[compressed != 0]
print(f"\n  Valor mínimo en original: {original_nonzero.min()}")
print(f"  Valor mínimo en comprimido: {compressed_nonzero.min()}")
print(f"  Valor máximo en comprimido: {compressed_nonzero.max()}")

# Verificar que los valores eliminados son los pequeños
if len(compressed_nonzero) > 0:
    threshold_expected = sorted(data.flatten())[int(np.ceil(data.size * 0.20)) - 1]
    print(f"  Umbral esperado (20-ésimo valor ordenado): {threshold_expected}")
    smallest_kept = compressed_nonzero.min()
    print(f"  Valor más pequeño que se mantuvo: {smallest_kept}")
    print(f"  ✓ Los valores pequeños fueron eliminados" if smallest_kept > threshold_expected else f"  ✗ Algo está mal")

# Test 2: Array con complejos
print("\n" + "=" * 70)
print("TEST 2: Array con números complejos")
print("=" * 70)

data_complex = np.random.randn(8, 8) + 1j * np.random.randn(8, 8)
print(f"\nDatos complejos: {data_complex.size} elementos")

compressed_complex = compress_coefficients(data_complex, compression_percent=30)
zeros_count_complex = np.sum(compressed_complex == 0)
target_zeros_complex = int(np.ceil(data_complex.size * 0.30))

print(f"Compresión: 30%")
print(f"  Ceros esperados: {target_zeros_complex} ({30}%)")
print(f"  Ceros reales: {zeros_count_complex}")
print(f"  ✓ CORRECTO" if zeros_count_complex == target_zeros_complex else f"  ✗ ERROR")

# Test 3: Array que ya tiene ceros
print("\n" + "=" * 70)
print("TEST 3: Array que ya tiene ceros")
print("=" * 70)

data_with_zeros = np.arange(1, 101, dtype=np.float64).reshape(10, 10)
# Poner 10 ceros manualmente
data_with_zeros[0, :5] = 0
existing_zeros = np.sum(data_with_zeros == 0)

print(f"\nDatos originales con {existing_zeros} ceros preexistentes")
print(f"Total de elementos: {data_with_zeros.size}")

compressed_with_zeros = compress_coefficients(data_with_zeros, compression_percent=20)
total_zeros = np.sum(compressed_with_zeros == 0)
target_zeros_total = int(np.ceil(data_with_zeros.size * 0.20))

print(f"\nCompresión: 20%")
print(f"  Ceros preexistentes: {existing_zeros}")
print(f"  Ceros esperados totales: {target_zeros_total}")
print(f"  Ceros reales totales: {total_zeros}")
print(f"  ✓ CORRECTO" if total_zeros == target_zeros_total else f"  ✗ ERROR")

# Test 4: Comprimir al 0% (no debe cambiar)
print("\n" + "=" * 70)
print("TEST 4: Compresión al 0% (sin cambios)")
print("=" * 70)

data_test = np.arange(1, 101, dtype=np.float64).reshape(10, 10)
compressed_0 = compress_coefficients(data_test, compression_percent=0)
print(f"\nCompresión: 0%")
print(f"  Iguales: {np.allclose(data_test, compressed_0)}")
print(f"  ✓ CORRECTO" if np.allclose(data_test, compressed_0) else f"  ✗ ERROR")

# Test 5: Comprimir al 100% (todo a 0)
print("\n" + "=" * 70)
print("TEST 5: Compresión al 100% (todo a cero)")
print("=" * 70)

data_test = np.arange(1, 101, dtype=np.float64).reshape(10, 10)
compressed_100 = compress_coefficients(data_test, compression_percent=100)
all_zeros = np.all(compressed_100 == 0)
print(f"\nCompresión: 100%")
print(f"  Todo ceros: {all_zeros}")
print(f"  ✓ CORRECTO" if all_zeros else f"  ✗ ERROR")

print("\n" + "=" * 70)
print("RESUMEN DE TESTS")
print("=" * 70)
print("Si todos muestran ✓ CORRECTO, la compresión funciona como se espera.")
