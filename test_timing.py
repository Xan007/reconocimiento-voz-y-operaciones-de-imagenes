#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test timing output from encryption functions"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
from PIL import Image
from processing.encryption import encrypt_image_rgb, decrypt_image_rgb

print("=" * 60)
print("TIMING TEST: Creating 64x64 random RGB image...")
print("=" * 60)

# Crear imagen de prueba pequeña: 64x64 RGB
test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8).astype(np.float64)
print(f"Test image shape: {test_image.shape}, dtype: {test_image.dtype}")

print("\n" + "=" * 60)
print("STARTING ENCRYPTION...")
print("=" * 60)

try:
    # Encriptar
    enc_result = encrypt_image_rgb(
        test_image,
        alpha_r=0.5,
        alpha_g=0.5,
        alpha_b=0.5,
        arnold_a=1,
        arnold_k=1,
        compression=0
    )
    
    print("\n" + "=" * 60)
    print("STARTING DECRYPTION...")
    print("=" * 60)
    
    # Desencriptar
    dec_result = decrypt_image_rgb(
        enc_result.encrypted_r,
        enc_result.encrypted_g,
        enc_result.encrypted_b,
        alpha_r=0.5,
        alpha_g=0.5,
        alpha_b=0.5,
        arnold_a=1,
        arnold_k=1,
        compression=0
    )
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)
    print(f"Original shape: {test_image.shape}")
    print(f"Decrypted shape: {dec_result.decrypted_rgb.shape}")
    print(f"MSE between original and decrypted: {np.mean((test_image - dec_result.decrypted_rgb)**2):.6f}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
