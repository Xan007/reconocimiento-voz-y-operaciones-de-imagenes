"""
Script para descargar librerías externas localmente (una sola vez).
Permite que el proyecto funcione completamente offline.

Uso: python scripts/download_offline_assets.py
"""

import os
import urllib.request
import sys
from pathlib import Path

# URLs de las librerías necesarias
ASSETS = {
    'plotly-2.26.0.min.js': 'https://cdn.plot.ly/plotly-2.26.0.min.js',
    'font-awesome-6.4.0.min.css': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
}

# URLs de fuentes de Font Awesome
FONTS = {
    'fa-solid-900.woff2': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/webfonts/fa-solid-900.woff2',
    'fa-solid-900.ttf': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/webfonts/fa-solid-900.ttf',
    'fa-brands-400.woff2': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/webfonts/fa-brands-400.woff2',
    'fa-brands-400.ttf': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/webfonts/fa-brands-400.ttf',
}

def download_assets():
    """Descarga todas las librerías necesarias"""
    lib_dir = Path(__file__).parent.parent / 'static' / 'lib'
    lib_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Descargando librerías a {lib_dir}...\n")
    
    for filename, url in ASSETS.items():
        filepath = lib_dir / filename
        
        # Saltar si ya existe
        if filepath.exists():
            print(f"✓ {filename} ya existe (saltando)")
            continue
        
        print(f"⏳ Descargando {filename}...", end=' ')
        sys.stdout.flush()
        
        try:
            urllib.request.urlretrieve(url, filepath)
            size_kb = filepath.stat().st_size / 1024
            print(f"✓ ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    # Actualizar rutas en el CSS descargado
    css_path = lib_dir / 'font-awesome-6.4.0.min.css'
    if css_path.exists():
        print("\n🔧 Actualizando rutas en CSS de Font Awesome...")
        content = css_path.read_text()
        content = content.replace(
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/webfonts/',
            '/static/webfonts/'
        )
        css_path.write_text(content)
        print("   ✓ Rutas actualizadas")
    
    # Descargar fuentes
    webfonts_dir = Path(__file__).parent.parent / 'static' / 'webfonts'
    webfonts_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Descargando fuentes de Font Awesome a {webfonts_dir}...\n")
    
    for filename, url in FONTS.items():
        filepath = webfonts_dir / filename
        
        if filepath.exists():
            print(f"✓ {filename} ya existe")
            continue
        
        print(f"⏳ Descargando {filename}...", end=' ')
        sys.stdout.flush()
        
        try:
            urllib.request.urlretrieve(url, filepath)
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✓ ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    print(f"\n✅ Todas las librerías y fuentes descargadas correctamente")
    print(f"📍 El proyecto ahora funcionará sin internet")
    return True

if __name__ == '__main__':
    if not download_assets():
        sys.exit(1)
