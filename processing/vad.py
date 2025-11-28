# vad.py
"""
Voice Activity Detection usando Silero VAD.
Detecta si hay voz humana real en el audio (ignora ruido de ventiladores, etc.)
"""

import torch
import numpy as np
from config import FS

# Cargar modelo Silero VAD (se carga una sola vez)
_vad_model = None
_vad_utils = None


def load_vad_model():
    """Carga el modelo Silero VAD (solo una vez)"""
    global _vad_model, _vad_utils
    
    if _vad_model is None:
        print("🔊 Cargando modelo Silero VAD...")
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        _vad_model = model
        _vad_utils = utils
        print("✅ Modelo Silero VAD cargado")
    
    return _vad_model, _vad_utils


def detect_voice(audio_data, sample_rate=FS, threshold=0.5):
    """
    Detecta si hay voz humana en el audio usando Silero VAD.
    
    Parámetros:
    -----------
    audio_data : ndarray
        Audio en formato float32, valores entre -1 y 1
    sample_rate : int
        Frecuencia de muestreo del audio
    threshold : float
        Umbral de confianza (0-1). Mayor = más estricto
    
    Retorna:
    --------
    dict con:
        - has_voice: bool - True si se detectó voz
        - confidence: float - Probabilidad promedio de voz (0-1)
        - max_confidence: float - Probabilidad máxima detectada
        - speech_segments: list - Segmentos donde hay voz
    """
    model, utils = load_vad_model()
    
    # Convertir a tensor
    audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
    
    # Silero VAD funciona mejor a 16kHz
    if sample_rate != 16000:
        # Resamplear a 16kHz
        import torchaudio
        if not hasattr(torchaudio, 'transforms'):
            # Fallback: resample manual
            from scipy.signal import resample
            num_samples_16k = int(len(audio_data) * 16000 / sample_rate)
            audio_resampled = resample(audio_data, num_samples_16k)
            audio_tensor = torch.from_numpy(audio_resampled.astype(np.float32))
            sample_rate = 16000
        else:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)
            sample_rate = 16000
    
    # Obtener probabilidades de voz por chunks
    # Silero procesa en ventanas de 512 muestras a 16kHz (32ms)
    window_size = 512
    speech_probs = []
    
    for i in range(0, len(audio_tensor) - window_size, window_size):
        chunk = audio_tensor[i:i + window_size]
        if len(chunk) == window_size:
            prob = model(chunk, sample_rate).item()
            speech_probs.append(prob)
    
    if not speech_probs:
        return {
            'has_voice': False,
            'confidence': 0.0,
            'max_confidence': 0.0,
            'speech_segments': []
        }
    
    # Calcular estadísticas
    avg_prob = np.mean(speech_probs)
    max_prob = np.max(speech_probs)
    
    # Encontrar segmentos con voz
    speech_segments = []
    in_speech = False
    start_idx = 0
    
    for i, prob in enumerate(speech_probs):
        if prob >= threshold and not in_speech:
            in_speech = True
            start_idx = i
        elif prob < threshold and in_speech:
            in_speech = False
            # Convertir índices a tiempo
            start_time = (start_idx * window_size) / 16000
            end_time = (i * window_size) / 16000
            speech_segments.append({
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time
            })
    
    # Si terminó en voz
    if in_speech:
        start_time = (start_idx * window_size) / 16000
        end_time = (len(speech_probs) * window_size) / 16000
        speech_segments.append({
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time
        })
    
    # Calcular duración total de voz
    total_speech_duration = sum(seg['duration'] for seg in speech_segments)
    
    # Criterio para detectar voz: 
    # - Al menos 0.1 segundos de voz detectada
    # - O probabilidad máxima > threshold
    has_voice = max_prob >= threshold and total_speech_duration >= 0.1
    
    return {
        'has_voice': has_voice,
        'confidence': avg_prob,
        'max_confidence': max_prob,
        'speech_segments': speech_segments,
        'total_speech_duration': total_speech_duration
    }


def preload_vad():
    """Pre-carga el modelo VAD al iniciar la aplicación"""
    try:
        load_vad_model()
        return True
    except Exception as e:
        print(f"⚠️ Error cargando VAD: {e}")
        return False
