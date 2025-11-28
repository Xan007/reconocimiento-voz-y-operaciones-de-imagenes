from processing.frdct import encrypt_image, decrypt_image
import numpy as np

# Test con imagen aleatoria 32x32
img = (np.random.rand(32, 32) * 255).astype(np.float64)

print("Alpha | Max Error    | Cond Number (8x8)")
print("-" * 45)

alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
for alpha in alphas:
    T = build_frdct_matrix(8, alpha)
    cond = np.linalg.cond(T)
    enc = encrypt_image(img, alpha)
    dec = decrypt_image(enc, alpha)
    max_err = np.max(np.abs(img - dec))
    print(f"{alpha:5.2f} | {max_err:12.2e} | {cond:12.2e}")

print()
enc0 = encrypt_image(img, 0.0)
print('enc0 min, max:', float(enc0.min()), float(enc0.max()))
dec0 = decrypt_image(enc0, 0.0)
print('dec0 min, max:', float(dec0.min()), float(dec0.max()))
