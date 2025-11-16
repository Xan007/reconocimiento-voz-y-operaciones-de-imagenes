#!/usr/bin/env python
"""
Script de prueba para verificar que la aplicación Flask funciona correctamente
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, models
import numpy as np
from config import FS, TARGET_N

def test_models_loaded():
    """Verifica que los modelos se cargaron correctamente"""
    print("🔍 Verificando carga de modelos...")
    if not models:
        print("❌ No hay modelos cargados")
        return False
    
    for name, model in models.items():
        print(f"✅ Modelo '{name}': {len(model.mean_energy)} bandas, {model.num_samples} muestras")
    
    return True

def test_api_routes():
    """Verifica que los endpoints de API estén disponibles"""
    print("\n🔍 Verificando rutas API...")
    
    routes = [
        ('/', 'GET', 'Página principal'),
        ('/models', 'GET', 'Página de modelos'),
        ('/api/health', 'GET', 'Health check'),
        ('/api/models', 'GET', 'Lista de modelos'),
    ]
    
    with app.test_client() as client:
        for route, method, desc in routes:
            try:
                if method == 'GET':
                    response = client.get(route)
                    status = '✅' if response.status_code in [200, 301, 302] else '❌'
                    print(f"{status} {method:6} {route:20} - {desc} ({response.status_code})")
            except Exception as e:
                print(f"❌ {method:6} {route:20} - Error: {e}")

def test_audio_processing():
    """Verifica que el procesamiento de audio funciona"""
    print("\n🔍 Verificando procesamiento de audio...")
    
    from processing.fft_utils import analyze_signal
    from processing.audio_utils import normalize_audio, pad_or_trim
    
    try:
        # Crear audio de prueba (ruido blanco)
        x = np.random.randn(FS) * 0.1
        x = normalize_audio(x.astype(np.float32))
        x = pad_or_trim(x, TARGET_N)
        
        freqs, spectrum, energies = analyze_signal(x)
        
        print(f"✅ Audio procesado: {len(x)} muestras")
        print(f"✅ Frecuencias: {len(freqs)} componentes")
        print(f"✅ Energías: {len(energies)} bandas, suma={np.sum(energies):.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Error procesando audio: {e}")
        return False

def main():
    print("=" * 60)
    print("  Pruebas de la Aplicación Flask")
    print("=" * 60)
    print(f"📊 Configuración: FS={FS} Hz, TARGET_N={TARGET_N}")
    print()
    
    all_ok = True
    
    all_ok &= test_models_loaded()
    all_ok &= test_audio_processing()
    test_api_routes()
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ Todas las verificaciones pasaron")
        print("\n🚀 Puedes iniciar la aplicación con:")
        print("   python run_app.py")
        return 0
    else:
        print("❌ Algunas verificaciones fallaron")
        return 1

if __name__ == '__main__':
    sys.exit(main())
