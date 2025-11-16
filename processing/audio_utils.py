# audio_utils.py
"""
Funciones para grabar, leer, guardar y normalizar audio.
Cumple con el requisito: todas las señales tendrán TARGET_N muestras.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample
from config import FS, DURATION, TARGET_N, CHANNELS

def list_input_devices():
    """
    Lista SOLO los dispositivos de entrada de audio disponibles.
    
    Retorna:
    --------
    devices : list of dict
        Lista de dispositivos de ENTRADA con información:
        [
            {'id': 0, 'name': 'Micrófono (Realtek)', 'channels': 2},
            {'id': 1, 'name': 'Auriculares USB', 'channels': 1},
            ...
        ]
    
    Nota: Se excluyen dispositivos de salida (speakers, headphones)
    """
    try:
        devices_raw = sd.query_devices()
    except Exception as e:
        print(f"Error consultando dispositivos: {e}")
        return []
    
    input_devices = []
    
    # Iterar sobre todos los dispositivos
    for i, device in enumerate(devices_raw):
        if not isinstance(device, dict):
            continue
        
        # FILTRO IMPORTANTE: Solo dispositivos de entrada
        # max_input_channels > 0 significa que tiene capacidad de entrada
        max_input_channels = device.get('max_input_channels', 0)
        if max_input_channels <= 0:
            # Este es un dispositivo de salida, IGNORAR
            continue
        
        # Obtener información
        input_devices.append({
            'id': i,
            'name': device.get('name', f'Dispositivo {i}'),
            'channels': max_input_channels,
            'default': device.get('default_input', False),  # Usar default_input, no default
            'latency': device.get('latency', (0, 0))
        })
    
    return input_devices

def print_input_devices():
    """
    Imprime lista formateada de SOLO dispositivos de entrada disponibles.
    Excluye dispositivos de salida (speakers, headphones, etc).
    """
    devices = list_input_devices()
    
    if not devices:
        print("No hay dispositivos de ENTRADA de audio disponibles")
        return
    
    print("\n" + "="*70)
    print("DISPOSITIVOS DE ENTRADA DE AUDIO DISPONIBLES")
    print("="*70)
    print(f"{'ID':>3} | {'Nombre':40s} | {'Canales':>8} | {'Estado':12s}")
    print("-"*70)
    
    for device in devices:
        status = "DEFAULT" if device['default'] else ""
        print(f"{device['id']:3d} | {device['name']:40s} | {device['channels']:8d} | {status:12s}")
    
    print("="*70 + "\n")

def record(duration=DURATION, fs=FS, channels=CHANNELS, device=None):
    """
    Graba audio desde el micrófono y devuelve un numpy array float32, mono.
    
    Parámetros:
    -----------
    duration : float
        Duración de grabación en segundos
    fs : int
        Frecuencia de muestreo en Hz
    channels : int
        Número de canales (1=mono, 2=estéreo)
    device : int o str, optional
        ID del dispositivo de entrada (ver list_input_devices())
        Si es None, usa el dispositivo por defecto del sistema
    
    Retorna:
    --------
    data : ndarray float32
        Audio grabado, mono, con shape (N_samples,)
    """
    print(f"🎤 Grabando {duration:.2f} s a {fs} Hz desde dispositivo {device}...")
    sd.default.samplerate = fs
    sd.default.channels = channels
    if device is not None:
        sd.default.device = device
    data = sd.rec(int(duration * fs), dtype='float32')
    sd.wait()
    # si channels>1, convertir a mono (promedio)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.flatten()

def save_wav(filename, data, fs=FS):
    """
    Guarda un vector float32 como WAV (valor entre -1 y 1).
    """
    sf.write(filename, data, samplerate=fs, subtype='PCM_16')
    print(f"Guardado: {filename}")

def read_wav(filename, target_fs=FS):
    """
    Lee un wav y lo devuelve como numpy float32 mono. Hace resample si la fs difiere.
    """
    data, fs = sf.read(filename, always_2d=False)
    # convertir a mono si necesario
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    # convertir a float32
    data = data.astype('float32')
    if fs != target_fs:
        n_samples = int(len(data) * target_fs / fs)
        data = resample(data, n_samples)
    return data, target_fs

def normalize_audio(x):
    """
    Normaliza señal a rango [-1, 1] (dividiendo entre el valor absoluto máximo).
    Evita división por cero.
    """
    mx = np.max(np.abs(x))
    if mx == 0:
        return x
    return x / mx

def pad_or_trim(x, target_n=TARGET_N):
    """
    Asegura que x tenga exactamente target_n muestras:
    - Si es más corta, hace padding con ceros al final.
    - Si es más larga, la recorta (trim) al inicio o al centro.
    Aquí uso recorte centrado (mantener el centro de la señal).
    """
    n = len(x)
    if n == target_n:
        return x
    if n < target_n:
        pad = np.zeros(target_n - n, dtype=x.dtype)
        return np.concatenate([x, pad])
    # n > target_n: recorte centrado
    start = max(0, (n - target_n)//2)
    return x[start:start + target_n]

# Función de utilidad completa: grabar -> normalizar -> pad/trim -> devolver
def record_and_prepare(duration=DURATION, fs=FS, channels=CHANNELS, target_n=TARGET_N, device=None):
    raw = record(duration=duration, fs=fs, channels=channels, device=device)
    norm = normalize_audio(raw)
    prepared = pad_or_trim(norm, target_n=target_n)
    return prepared
