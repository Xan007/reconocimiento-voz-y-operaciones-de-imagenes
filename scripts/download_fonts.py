"""
Descarga todas las fuentes (webfonts) de Font Awesome localmente.
"""

import urllib.request
from pathlib import Path

FONTS = [
    ('fa-solid-900.woff2', 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/webfonts/fa-solid-900.woff2'),
    ('fa-solid-900.ttf', 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/webfonts/fa-solid-900.ttf'),
    ('fa-brands-400.woff2', 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/webfonts/fa-brands-400.woff2'),
    ('fa-brands-400.ttf', 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/webfonts/fa-brands-400.ttf'),
]

webfonts_dir = Path(__file__).parent.parent / 'static' / 'webfonts'
webfonts_dir.mkdir(parents=True, exist_ok=True)

print("📥 Descargando fuentes de Font Awesome...\n")
for filename, url in FONTS:
    filepath = webfonts_dir / filename
    if filepath.exists():
        size = filepath.stat().st_size / (1024*1024)
        print(f"✓ {filename} ya existe ({size:.1f} MB)")
        continue
    
    print(f"⏳ Descargando {filename}...", end=' ', flush=True)
    try:
        urllib.request.urlretrieve(url, filepath)
        size = filepath.stat().st_size / (1024*1024)
        print(f"✓ ({size:.1f} MB)")
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n✅ Todas las fuentes descargadas correctamente")
