# -*- coding: utf-8 -*-
"""
Aplicación Flask para reconocimiento de comandos por análisis FFT.
Interfaz web para las funcionalidades de reconocimiento en tiempo real
y análisis de modelos.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import numpy as np
import io
import base64
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import soundfile as sf
from scipy.io import wavfile
from PIL import Image

# Intentar importar pydub (opcional)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    print("[!] pydub no disponible (algunas funciones de audio limitadas)")
    PYDUB_AVAILABLE = False
except Exception:
    print("[!] pydub no disponible (algunas funciones de audio limitadas)")
    PYDUB_AVAILABLE = False

from config import FS, TARGET_N, MODELOS_DIR, DATA_DIR
from processing.audio_utils import normalize_audio, pad_or_trim, read_wav, list_input_devices
from processing.fft_utils import analyze_signal
from recognition.recognizer import load_models, compare_with_models
from recognition.model import Model
from datetime import datetime

# ============ AUDIO LOG SYSTEM (Últimos 5 audios) ============
AUDIO_LOG_DIR = os.path.join(DATA_DIR, 'audio_logs')
os.makedirs(AUDIO_LOG_DIR, exist_ok=True)
MAX_AUDIO_LOGS = 5

def save_audio_log(audio_bytes):
    """Guarda los últimos 5 audios de 1 segundo en audio_logs/"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # YYYYMMDD_HHMMSS_ms
        filename = f'audio_{timestamp}.wav'
        filepath = os.path.join(AUDIO_LOG_DIR, filename)
        
        # Guardar el audio
        with open(filepath, 'wb') as f:
            f.write(audio_bytes)
        
        # Limpiar audios antiguos, mantener solo los últimos 5
        audio_files = sorted([f for f in os.listdir(AUDIO_LOG_DIR) if f.startswith('audio_') and f.endswith('.wav')])
        if len(audio_files) > MAX_AUDIO_LOGS:
            for old_file in audio_files[:-MAX_AUDIO_LOGS]:
                try:
                    os.remove(os.path.join(AUDIO_LOG_DIR, old_file))
                    print(f"🗑️ Audio log eliminado: {old_file}")
                except Exception as e:
                    print(f"⚠️ Error eliminando {old_file}: {e}")
        
        print(f"💾 Audio guardado: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error guardando audio log: {e}")
        return False

# Configuración de la aplicación
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB máximo para soportar imágenes grandes
app.config['UPLOAD_FOLDER'] = os.path.join(DATA_DIR, 'uploads')

# Manejador de error para Request Entity Too Large
@app.errorhandler(413)
def request_entity_too_large(error):
    """Maneja errores de payload demasiado grande"""
    return jsonify({
        'success': False,
        'error': 'El archivo es demasiado grande (máximo 200MB). Intenta con una imagen más pequeña.'
    }), 413

# Crear carpeta de uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Middleware para cacheo de recursos estáticos (archivos locales)
@app.after_request
def add_cache_headers(response):
    """Añade headers de cache para recursos estáticos"""
    # Para APIs (/api/*), NO cachear
    if request.path.startswith('/api/'):
        response.cache_control.max_age = 0
        response.cache_control.no_cache = True
        response.cache_control.no_store = True
        response.cache_control.must_revalidate = True
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    elif response.content_type and any(ext in response.content_type for ext in 
                                     ['javascript', 'css', 'font', 'image']):
        # Cache agresivo: 30 días para archivos estáticos
        response.cache_control.max_age = 2592000
        response.cache_control.public = True
    elif response.status_code == 200:
        # Para HTML: cache corto
        response.cache_control.max_age = 3600
        response.cache_control.public = True
    return response

# Cargar modelos al iniciar
try:
    models = load_models()
except FileNotFoundError as e:
    models = {}
    print(f"⚠️ Advertencia: {e}")


@app.route('/')
def index():
    """Página principal con reconocimiento en tiempo real"""
    return render_template('index.html', fs=FS, target_n=TARGET_N)


@app.route('/audio-logs')
def audio_logs_page():
    """Página para ver audios grabados"""
    return render_template('audio_logs.html')


@app.route('/models')
def models_page():
    """Página de análisis de modelos"""
    model_names = list(models.keys())
    return render_template('models.html', model_names=model_names)


@app.route('/api/recognize', methods=['POST'])
def recognize():
    """
    Endpoint para reconocimiento de audio.
    Acepta audio en múltiples formatos desde el navegador.
    """
    try:
        audio_data = None
        sr = FS

        if 'audio' in request.files:
            # Archivo cargado como multipart/form-data
            audio_file = request.files['audio']
            audio_bytes = audio_file.read()
            
            # 💾 Guardar el audio en el log (últimos 5)
            save_audio_log(audio_bytes)
            
            # Intentar decodificar con múltiples métodos
            try:
                # Método 1: soundfile
                audio_data, sr = sf.read(io.BytesIO(audio_bytes))
                print(f"✅ Audio decodificado con soundfile: sr={sr}, shape={audio_data.shape}")
            except Exception as e1:
                try:
                    # Método 2: scipy.io.wavfile
                    sr, audio_data = wavfile.read(io.BytesIO(audio_bytes))
                    audio_data = audio_data.astype(np.float32)
                    if np.max(np.abs(audio_data)) > 1.0:
                        audio_data = audio_data / 32768.0  # Normalizar de 16-bit
                    print(f"✅ Audio decodificado con scipy.wavfile: sr={sr}, shape={audio_data.shape}")
                except Exception as e2:
                    if PYDUB_AVAILABLE:
                        try:
                            # Método 3: pydub (para Ogg, WebM, MP3, etc)
                            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
                            sr = audio.frame_rate
                            audio_np = np.array(audio.get_array_of_samples(), dtype=np.float32)
                            if audio.channels == 2:
                                audio_np = audio_np.reshape((-1, 2))
                                audio_data = np.mean(audio_np, axis=1)
                            else:
                                audio_data = audio_np
                            # Normalizar
                            if np.max(np.abs(audio_data)) > 1.0:
                                audio_data = audio_data / 32768.0
                            print(f"✅ Audio decodificado con pydub: sr={sr}, shape={audio_data.shape}")
                        except Exception as e3:
                            print(f"❌ Error decodificando audio - soundfile: {e1}")
                            print(f"❌ Error decodificando audio - scipy: {e2}")
                            print(f"❌ Error decodificando audio - pydub: {e3}")
                            return jsonify({
                                'error': 'No se pudo decodificar el audio. Intenta de nuevo.',
                                'success': False
                            }), 400
                    else:
                        print(f"❌ Error decodificando audio - soundfile: {e1}")
                        print(f"❌ Error decodificando audio - scipy: {e2}")
                        return jsonify({
                            'error': 'No se pudo decodificar el audio. Intenta con formato WAV.',
                            'success': False
                        }), 400

        elif request.data:
            # Datos raw desde Web Audio API (Float32Array del navegador)
            try:
                audio_array = np.frombuffer(request.data, dtype=np.float32)
                audio_data = audio_array
                sr = FS
                print(f"✅ Audio raw procesado: shape={audio_data.shape}")
            except Exception as e:
                return jsonify({
                    'error': f'Error procesando datos raw: {str(e)}',
                    'success': False
                }), 400
        else:
            return jsonify({
                'error': 'No audio data provided',
                'success': False
            }), 400

        if audio_data is None or len(audio_data) == 0:
            return jsonify({
                'error': 'No se pudo procesar el audio',
                'success': False
            }), 400

        # Convertir a mono si es necesario
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Asegurar tipo float32
        audio_data = audio_data.astype(np.float32)

        # DEBUG: Ver audio ANTES de cualquier procesamiento
        print(f"\n{'='*60}")
        print(f"DEBUG AUDIO ORIGINAL (antes de normalizar/pad_trim)")
        print(f"{'='*60}")
        print(f"  Longitud: {len(audio_data)} muestras ({len(audio_data)/sr:.3f} s)")
        print(f"  Sample rate recibido: {sr}")
        print(f"  Max amplitud: {np.max(np.abs(audio_data)):.6f}")
        # Ver energía en tercios del audio original
        third = len(audio_data) // 3
        e1 = np.sum(audio_data[:third]**2) / third if third > 0 else 0
        e2 = np.sum(audio_data[third:2*third]**2) / third if third > 0 else 0
        e3 = np.sum(audio_data[2*third:]**2) / (len(audio_data) - 2*third) if len(audio_data) > 2*third else 0
        print(f"  Energía tercio 1 (0-{third/sr:.3f}s): {e1:.10f}")
        print(f"  Energía tercio 2 ({third/sr:.3f}-{2*third/sr:.3f}s): {e2:.10f}")
        print(f"  Energía tercio 3 ({2*third/sr:.3f}s-fin): {e3:.10f}")
        print(f"{'='*60}\n")

        # Normalizar y preparar
        audio_data = normalize_audio(audio_data)
        audio_data = pad_or_trim(audio_data, TARGET_N)

        # Analizar señal (ahora solo retorna energías por subbandas temporales)
        _, _, energies = analyze_signal(audio_data)

        # Comparar con modelos (usando métrica de config.py)
        from config import DISTANCE_METRIC, RECOGNITION_THRESHOLD
        best_cmd, diffs, is_valid = compare_with_models(energies, models, distance_method=DISTANCE_METRIC)

        # DEBUG: Verificar energías normalizadas
        print(f"DEBUG: Sum of energies: {np.sum(energies)}")
        print(f"DEBUG: Energies: {energies}")
        print(f"DEBUG: Max energy: {np.max(energies)}")
        print(f"DEBUG: is_valid: {is_valid}, threshold: {RECOGNITION_THRESHOLD}")

        # Calcular FFT solo para visualización
        from processing.fft_utils import compute_fft, magnitude_spectrum
        
        audio_for_fft = audio_data
        spectrum, freqs = compute_fft(audio_for_fft)
        spectrum_mag = magnitude_spectrum(spectrum)

        # Preparar gráficas
        plot_data = {
            'waveform': {
                'x': (np.arange(len(audio_data)) / FS).tolist(),
                'y': audio_data.tolist(),
            },
            'spectrum': {
                'x': freqs.tolist(),
                'y': spectrum_mag.tolist(),
            },
            'energies': energies.tolist(),
            'models_comparison': {}
        }

        # Comparaciones con modelos
        for model_name, model_obj in models.items():
            # Sanitizar el valor de diferencia (convertir inf a un valor grande)
            diff_value = diffs.get(model_name, 0.0)
            if np.isinf(diff_value) or np.isnan(diff_value):
                diff_value = 999999.0
            
            plot_data['models_comparison'][model_name] = {
                'model_energies': model_obj.mean_energy.tolist(),
                'model_std': model_obj.std_energy.tolist(),
                'input_energies': energies.tolist(),
                'difference': float(diff_value)
            }

        # Calcular la mejor diferencia para incluir en la respuesta
        best_diff = min(diffs.values()) if diffs else 999999.0
        if np.isinf(best_diff) or np.isnan(best_diff):
            best_diff = 999999.0
        
        # Sanitizar todas las diferencias para JSON
        sanitized_diffs = {}
        for k, v in diffs.items():
            if np.isinf(v) or np.isnan(v):
                sanitized_diffs[k] = 999999.0
            else:
                sanitized_diffs[k] = float(v)
        
        response = {
            'success': True,
            'recognized_command': best_cmd,
            'differences': sanitized_diffs,
            'is_confident': is_valid,  # True si la distancia está por debajo del umbral
            'threshold': RECOGNITION_THRESHOLD,
            'best_distance': float(best_diff),
            'energy_total': float(np.sum(energies)),
            'plot_data': plot_data,
            'models': list(models.keys())
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error en reconocimiento: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 400


@app.route('/api/models/<model_name>/data')
def get_model_data(model_name):
    """
    Obtiene datos de un modelo específico para visualización.
    Recarga los datos del JSON para evitar cache.
    """
    try:
        from recognition.model import Model
        import json
        import os
        
        model_file = os.path.join(MODELOS_DIR, f'{model_name}.json')
        if not os.path.exists(model_file):
            return jsonify({'error': f'Modelo {model_name} no encontrado'}), 404
        
        # Cargar modelo fresco desde JSON
        with open(model_file, 'r') as f:
            data = json.load(f)
        
        model = Model(data)
        n_bands = len(model.mean_energy)

        response_data = {
            'model_name': model_name,
            'n_bands': n_bands,
            'num_samples': model.num_samples,
            'mean_energy': model.mean_energy.tolist(),
            'std_energy': model.std_energy.tolist(),
            'total_energy': float(np.sum(model.mean_energy)),
            'bands': [f'B{i+1}' for i in range(n_bands)]
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/models')
def get_all_models():
    """Obtiene lista de todos los modelos disponibles"""
    model_list = []
    for model_name, model_obj in models.items():
        model_list.append({
            'name': model_name,
            'num_samples': model_obj.num_samples,
            'num_bands': len(model_obj.mean_energy),
            'total_energy': float(np.sum(model_obj.mean_energy))
        })
    return jsonify(model_list)


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'models_loaded': len(models),
        'model_names': list(models.keys())
    })


@app.route('/api/devices')
def get_devices():
    """
    Obtiene lista de dispositivos de entrada de audio disponibles.
    
    Retorna:
    --------
    devices : list of dict
        Cada dispositivo contiene: id, name, channels, default
    """
    try:
        devices = list_input_devices()
        return jsonify({
            'success': True,
            'devices': devices,
            'default_device': next((d['id'] for d in devices if d['default']), None)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'devices': []
        }), 400


@app.route('/training')
def training():
    """Página de entrenamiento de modelos"""
    try:
        from config import N_SUBBANDS
        return render_template('training.html', n_subbands=N_SUBBANDS)
    except Exception as e:
        return render_template('error.html', error=str(e)), 500


@app.route('/settings')
def settings():
    """Página de configuración del sistema"""
    try:
        from config import N_SUBBANDS, DISTANCE_METRIC
        return render_template('settings.html', n_subbands=N_SUBBANDS, distance_metric=DISTANCE_METRIC)
    except Exception as e:
        return render_template('error.html', error=str(e)), 500


@app.route('/api/devices', methods=['GET'])
def get_input_devices():
    """API para obtener lista de dispositivos de entrada de audio"""
    try:
        devices = list_input_devices()
        return jsonify({'success': True, 'devices': devices})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/audio-logs', methods=['GET'])
def get_audio_logs():
    """API para obtener lista de audios guardados (últimos 5)"""
    try:
        audio_files = sorted([f for f in os.listdir(AUDIO_LOG_DIR) if f.startswith('audio_') and f.endswith('.wav')])
        audio_files = audio_files[-MAX_AUDIO_LOGS:]  # Últimos 5
        return jsonify({
            'success': True,
            'audio_logs': audio_files,
            'count': len(audio_files)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/audio-logs/<filename>', methods=['GET'])
def download_audio_log(filename):
    """Descargar un audio del log"""
    try:
        # Validar que sea un archivo de audio válido
        if not filename.startswith('audio_') or not filename.endswith('.wav'):
            return jsonify({'success': False, 'error': 'Archivo inválido'}), 400
        
        filepath = os.path.join(AUDIO_LOG_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Archivo no encontrado'}), 404
        
        from flask import send_file
        return send_file(filepath, mimetype='audio/wav', download_name=filename, as_attachment=True)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/audio-logs/<filename>', methods=['DELETE'])
def delete_audio_log(filename):
    """Eliminar un audio del log"""
    try:
        # Validar que sea un archivo de audio válido
        if not filename.startswith('audio_') or not filename.endswith('.wav'):
            return jsonify({'success': False, 'error': 'Archivo inválido'}), 400
        
        filepath = os.path.join(AUDIO_LOG_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Archivo no encontrado'}), 404
        
        os.remove(filepath)
        print(f"🗑️ Audio eliminado: {filename}")
        return jsonify({'success': True, 'message': 'Audio eliminado'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/audio-logs/delete-all', methods=['POST'])
def delete_all_audio_logs():
    """Eliminar todos los audios del log"""
    try:
        audio_files = [f for f in os.listdir(AUDIO_LOG_DIR) if f.startswith('audio_') and f.endswith('.wav')]
        for filename in audio_files:
            filepath = os.path.join(AUDIO_LOG_DIR, filename)
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"⚠️ Error eliminando {filename}: {e}")
        
        print(f"🗑️ Todos los audios fueron eliminados ({len(audio_files)} archivos)")
        return jsonify({'success': True, 'message': f'{len(audio_files)} audios eliminados'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/config', methods=['POST'])
def update_config():
    """API para actualizar N_SUBBANDS, DISTANCE_METRIC y MICROPHONE_ID"""
    try:
        data = request.get_json()
        n_subbands = int(data.get('n_subbands', 8))
        distance_metric = data.get('distance_metric', 'euclidean')
        microphone_id = data.get('microphone_id')
        
        # Validar subbandas
        if n_subbands < 2 or n_subbands > 32:
            return jsonify({'success': False, 'error': 'n_subbands debe estar entre 2 y 32'}), 400
        
        # Validar métrica de distancia
        valid_metrics = ['euclidean', 'weighted_euclidean', 'nll_gaussian']
        if distance_metric not in valid_metrics:
            return jsonify({'success': False, 'error': f'distance_metric debe ser una de: {valid_metrics}'}), 400
        
        # Validar microphone_id - solo validar que sea número entero
        if microphone_id is not None:
            try:
                microphone_id = int(microphone_id)
                # Aceptar cualquier índice >= 0 (validación en frontend)
                if microphone_id < 0:
                    return jsonify({'success': False, 'error': 'microphone_id debe ser >= 0'}), 400
            except ValueError:
                return jsonify({'success': False, 'error': 'microphone_id debe ser un número entero'}), 400
        
        # Actualizar config.py
        config_path = os.path.join(os.path.dirname(__file__), 'config.py')
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Reemplazar valores
        import re
        content = re.sub(r'N_SUBBANDS = \d+', f'N_SUBBANDS = {n_subbands}', content)
        content = re.sub(r"DISTANCE_METRIC = '[a-z_]+'", f"DISTANCE_METRIC = '{distance_metric}'", content)
        
        # Actualizar MICROPHONE_ID
        if microphone_id is not None:
            content = re.sub(r'MICROPHONE_ID = [^#\n]+', f'MICROPHONE_ID = {microphone_id}', content)
        else:
            content = re.sub(r'MICROPHONE_ID = [^#\n]+', 'MICROPHONE_ID = None', content)
        
        with open(config_path, 'w') as f:
            f.write(content)
        
        msg = f'Configuración actualizada: subbandas={n_subbands}, métrica={distance_metric}, micrófono={microphone_id or "defecto"}'
        return jsonify({'success': True, 'message': msg})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train_models():
    """API para entrenar modelos"""
    try:
        from training.trainer import train_all
        train_all()
        
        # Recargar modelos
        global models
        models = load_models()
        
        return jsonify({'success': True, 'message': 'Entrenamiento completado exitosamente'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/evaluation')
def evaluation():
    """Página de evaluación de modelos"""
    return render_template('evaluation.html')


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """
    API para ejecutar evaluación con método de distancia seleccionado.
    
    Parámetros POST:
    - distance_method: str - 'euclidean', 'weighted_euclidean', etc.
                              Si es 'all', evalúa todos los métodos
    """
    try:
        from evaluate_model import evaluate_model, evaluate_all_methods
        
        # Obtener el método de distancia solicitado
        distance_method = request.json.get('distance_method', 'euclidean') if request.is_json else 'euclidean'
        
        # Si se solicita 'all', evaluar todos los métodos
        if distance_method == 'all':
            all_results = evaluate_all_methods()
            
            if not all_results:
                return jsonify({'success': False, 'error': 'No se pudo completar la evaluación'}), 500
            
            # Preparar respuesta con resultados de todos los métodos
            # Ordenar por margen de error (menor es mejor)
            sorted_methods = sorted(
                all_results.items(),
                key=lambda x: 100.0 - x[1]['porcentaje_acierto']
            )
            
            best_method = sorted_methods[0][0] if sorted_methods else None
            
            return jsonify({
                'success': True,
                'all_results': {
                    method: {
                        'total_aciertos': result['total_aciertos'],
                        'total_evaluaciones': result['total_evaluaciones'],
                        'porcentaje_acierto': result['porcentaje_acierto'],
                        'margen_error': 100.0 - result['porcentaje_acierto'],
                        'distance_method': result['distance_method'],
                        'por_comando': result['por_comando'],
                        'predicciones': result.get('predicciones', [])
                    }
                    for method, result in all_results.items()
                },
                'best_method': best_method,
                'comparison_order': [method for method, _ in sorted_methods]
            })
        else:
            # Evaluar con el método específico solicitado
            results = evaluate_model(distance_method=distance_method)
            
            if results is None:
                return jsonify({'success': False, 'error': 'No se pudo completar la evaluación'}), 500
            
            return jsonify({
                'success': True,
                'results': {
                    'total_aciertos': results['total_aciertos'],
                    'total_evaluaciones': results['total_evaluaciones'],
                    'porcentaje_acierto': results['porcentaje_acierto'],
                    'margen_error': 100.0 - results['porcentaje_acierto'],
                    'distance_method': results['distance_method'],
                    'por_comando': results['por_comando'],
                    'predicciones': results.get('predicciones', [])
                }
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/record')
def record():
    """Página para grabar audios de entrenamiento"""
    import os
    from pathlib import Path
    
    # Obtener lista de comandos
    comandos_dir = os.path.join(os.path.dirname(__file__), 'data', 'comandos')
    if os.path.exists(comandos_dir):
        comandos = sorted([d for d in os.listdir(comandos_dir) 
                          if os.path.isdir(os.path.join(comandos_dir, d))])
    else:
        comandos = []
    
    return render_template('record.html', comandos=comandos)


@app.route('/api/save-recording', methods=['POST'])
def save_recording():
    """API para guardar grabación de audio"""
    try:
        comando = request.form.get('comando', '').strip()
        nombre_sujeto = request.form.get('nombre_sujeto', 'desconocido').strip()
        
        if not comando:
            return jsonify({'success': False, 'error': 'Comando requerido'}), 400
        
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No hay audio en la solicitud'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'success': False, 'error': 'No se seleccionó archivo'}), 400
        
        # Crear directorio del comando si no existe
        comando_path = os.path.join(os.path.dirname(__file__), 'data', 'comandos', comando)
        os.makedirs(comando_path, exist_ok=True)
        
        # Obtener siguiente número de grabación
        files = [f for f in os.listdir(comando_path) if f.lower().endswith('.wav')]
        next_num = len(files) + 1
        
        # Crear nombre de archivo con nombre del sujeto
        safe_sujeto = "".join(c for c in nombre_sujeto if c.isalnum() or c in ('-', '_')).lower()
        if not safe_sujeto:
            safe_sujeto = "desconocido"
        
        filename = f"grabacion_{next_num:03d}_{safe_sujeto}.wav"
        filepath = os.path.join(comando_path, filename)
        
        # Guardar archivo WAV directamente (generado por Web Audio API)
        try:
            audio_bytes = audio_file.read()
            
            # Leer como WAV usando scipy
            try:
                sr, audio_data = wavfile.read(io.BytesIO(audio_bytes))
                audio_data = audio_data.astype(np.float32)
                if np.max(np.abs(audio_data)) > 1.0:
                    audio_data = audio_data / 32768.0
            except Exception as e:
                return jsonify({'success': False, 'error': f'Error leyendo WAV: {str(e)}'}), 400
            
            # Normalizar a mono si es necesario
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resamplear a FS si es diferente
            if sr != FS:
                from scipy import signal
                num_samples = int(len(audio_data) * FS / sr)
                audio_data = signal.resample(audio_data, num_samples)
            
            # Normalizar amplitud
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.95
            
            # Guardar
            wavfile.write(filepath, FS, (audio_data * 32767).astype(np.int16))
            
            return jsonify({
                'success': True, 
                'message': f'Grabación guardada: {filename}',
                'filename': filename
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error guardando audio: {str(e)}'}), 400
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    """Manejador de página no encontrada"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    """Manejador de error interno"""
    return render_template('500.html', error=str(e)), 500


@app.route('/image-processing')
def image_processing():
    """Página interactiva de procesamiento de imagen con voz"""
    return render_template('image_processing.html')


@app.route('/encryption-interactive')
def encryption_interactive():
    """Página de encriptación interactiva con comandos de voz"""
    return render_template('encryption_interactive.html')


@app.route('/api/image/grayscale', methods=['POST'])
def api_grayscale():
    """Convierte imagen a escala de grises"""
    try:
        from processing.image_interactive import pil_to_base64
        from PIL import Image
        
        data = request.get_json()
        image_base64 = data.get('image_base64')
        
        if not image_base64:
            return jsonify({'success': False, 'error': 'No hay imagen en base64'}), 400
        
        # Decodificar base64
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        img_bytes = base64.b64decode(image_base64)
        pil_img = Image.open(io.BytesIO(img_bytes))
        
        # Redimensionar a 256x256
        pil_img = pil_img.resize((256, 256), Image.LANCZOS)
        
        # Convertir a escala de grises
        gray_pil = pil_img.convert('L')
        
        # Convertir a base64 con prefijo data:image
        base64_img = pil_to_base64(gray_pil)
        full_base64 = f'data:image/png;base64,{base64_img}'
        
        return jsonify({
            'success': True,
            'image': full_base64
        })
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/image/compress', methods=['POST'])
def api_compress():
    """Comprime imagen usando DCT"""
    try:
        from processing.image_interactive import apply_dct_compression, pil_to_base64
        from PIL import Image
        
        data = request.get_json()
        image_base64 = data.get('image_base64')
        compression_percent = data.get('compression_percent', 50)
        
        if not image_base64:
            return jsonify({'success': False, 'error': 'No hay imagen'}), 400
        
        # Decodificar base64
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        img_bytes = base64.b64decode(image_base64)
        pil_img = Image.open(io.BytesIO(img_bytes))
        gray_array = np.array(pil_img.convert('L'))
        
        # Aplicar compresión
        compressed, magnitude, stats = apply_dct_compression(gray_array, compression_percent)
        
        # Convertir a base64
        compressed_pil = Image.fromarray(compressed.astype('uint8'))
        base64_img = pil_to_base64(compressed_pil)
        full_base64 = f'data:image/png;base64,{base64_img}'
        
        # Convertir magnitud a lista serializable
        magnitude_list = magnitude.tolist() if isinstance(magnitude, np.ndarray) else magnitude
        
        return jsonify({
            'success': True,
            'compressed_image': full_base64,
            'dct_magnitude': magnitude_list,
            'statistics': stats
        })
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/image/reconstruct', methods=['POST'])
def api_reconstruct():
    """Reconstruye imagen desde DCT comprimida"""
    try:
        from processing.image_interactive import reconstruct_from_dct, pil_to_base64
        from PIL import Image
        
        data = request.get_json()
        image_base64 = data.get('image_base64')
        compression_percent = data.get('compression_percent', 50)
        mode = data.get('mode', 'grey')
        
        if not image_base64:
            return jsonify({'success': False, 'error': 'No hay imagen'}), 400
        
        # Decodificar base64
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        img_bytes = base64.b64decode(image_base64)
        pil_img = Image.open(io.BytesIO(img_bytes))
        gray_array = np.array(pil_img.convert('L'))
        
        # Reconstruir
        reconstructed, stats = reconstruct_from_dct(gray_array, compression_percent, mode)
        
        # Convertir a base64
        reconstructed_pil = Image.fromarray(reconstructed.astype('uint8'))
        base64_img = pil_to_base64(reconstructed_pil)
        full_base64 = f'data:image/png;base64,{base64_img}'
        
        return jsonify({
            'success': True,
            'compressed_image': full_base64,
            'statistics': stats
        })
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/images')
def images():
    """Página para procesamiento de imágenes con DCT 2D"""
    return render_template('images.html')


@app.route('/api/process-image', methods=['POST'])
def process_image():
    """API para procesar imagen: escala grises -> DCT 2D -> invertir DCT"""
    try:
        from processing.image_processing import process_image_full_pipeline
        
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No hay imagen en la solicitud'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'No se seleccionó archivo'}), 400
        
        # Obtener porcentaje de compresión del formulario
        filter_percent = request.form.get('filter_percent', 0)
        try:
            filter_percent = min(100, max(0, int(filter_percent)))
        except (ValueError, TypeError):
            filter_percent = 0
        
        # Validar extensión
        allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}
        file_ext = image_file.filename.rsplit('.', 1)[1].lower() if '.' in image_file.filename else ''
        if file_ext not in allowed_extensions:
            return jsonify({'success': False, 'error': 'Formato no soportado. Use JPG, PNG, BMP o GIF'}), 400
        
        # Guardar temporalmente
        import tempfile
        temp_path = os.path.join(tempfile.gettempdir(), secure_filename(image_file.filename))
        image_file.save(temp_path)
        
        # Procesar imagen con compresión
        results = process_image_full_pipeline(temp_path, filter_percent=filter_percent)
        
        # Convertir imágenes a base64 para enviar al cliente
        def img_to_base64(img_array):
            """Convierte array numpy a imagen PNG base64"""
            from PIL import Image
            
            # Asegurar que está en formato uint8
            if img_array.dtype != np.uint8:
                if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
            
            if len(img_array.shape) == 2:  # Escala de grises
                img = Image.fromarray(img_array, mode='L')
            elif len(img_array.shape) == 3 and img_array.shape[2] == 3:  # RGB
                img = Image.fromarray(img_array, mode='RGB')
            else:
                raise ValueError(f"Formato de imagen no soportado: {img_array.shape}")
            
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f'data:image/png;base64,{img_base64}'
        
        # Preparar visualización DCT (convertir a uint8 si es necesario)
        dct_visual = results['dct_visual']
        if dct_visual.dtype == np.float32 or dct_visual.dtype == np.float64:
            dct_visual = (dct_visual * 255).astype(np.uint8)
        
        # Limpiar archivo temporal
        try:
            os.remove(temp_path)
        except:
            pass
        
        return jsonify({
            'success': True,
            'grayscale': img_to_base64(results['grayscale']),
            'dct_magnitude': img_to_base64(dct_visual),
            'reconstructed': img_to_base64(results['imagen_gris_reconstruida']),
            'reconstructed_color': img_to_base64(results['imagen_color_reconstruida']),
            'shape': results['shape'],
            'filter_percent': results['filter_percent'],
            'coeficientes_conservados': results['coeficientes_conservados']
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Error procesando imagen: {str(e)}'}), 500


@app.route('/frdct')
def frdct_page():
    """Página de cifrado de imágenes con FrDCT"""
    return render_template('frdct.html')


@app.route('/api/frdct/grayscale', methods=['POST'])
def api_frdct_grayscale():
    """Convierte imagen a escala de grises para FrDCT"""
    try:
        from PIL import Image
        from processing.frdct import array_to_base64
        
        data = request.get_json()
        image_base64 = data.get('image_base64')
        
        if not image_base64:
            return jsonify({'success': False, 'error': 'No hay imagen'}), 400
        
        # Decodificar base64
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        img_bytes = base64.b64decode(image_base64)
        pil_img = Image.open(io.BytesIO(img_bytes))
        
        # Convertir a escala de grises (mantener dimensiones originales)
        gray_pil = pil_img.convert('L')
        gray_array = np.array(gray_pil)
        
        # Convertir a base64
        result_base64 = array_to_base64(gray_array, normalize=False)
        
        return jsonify({
            'success': True,
            'image': result_base64,
            'width': gray_array.shape[1],
            'height': gray_array.shape[0]
        })
    except Exception as e:
        print(f"Error en frdct/grayscale: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/frdct/encrypt', methods=['POST'])
def api_frdct_encrypt():
    """Cifra una imagen usando FrDCT"""
    try:
        from processing.frdct import base64_to_array, encrypt_image, array_to_base64
        
        data = request.get_json()
        image_base64 = data.get('image_base64')
        alpha = float(data.get('alpha', 0.5))

        # Validar rango de alpha (0 <= α < 2, ya que α=2 produce matriz singular)
        if alpha < 0.0 or alpha >= 2.0:
            return jsonify({'success': False, 'error': 'α debe estar en el rango [0, 2). α=2 produce una matriz singular no invertible.'}), 400
        
        if not image_base64:
            return jsonify({'success': False, 'error': 'No hay imagen'}), 400
        
        # Convertir base64 a array
        gray_array = base64_to_array(image_base64)
        
        print(f"📐 Cifrando imagen {gray_array.shape} con α = {alpha}")
        
        # Aplicar FrDCT (cifrar)
        encrypted = encrypt_image(gray_array, alpha, use_fast=True)
        
        # Convertir coeficientes a lista para JSON
        encrypted_list = encrypted.tolist()
        
        # Crear visualización normalizada de los coeficientes (escala logarítmica)
        encrypted_visual = array_to_base64(encrypted, normalize=True, use_log_scale=True)
        
        print(f"✅ Cifrado completado")
        
        return jsonify({
            'success': True,
            'encrypted_data': encrypted_list,
            'encrypted_visual': encrypted_visual,
            'alpha_used': alpha
        })
    except Exception as e:
        print(f"Error en frdct/encrypt: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/frdct/decrypt', methods=['POST'])
def api_frdct_decrypt():
    """Descifra una imagen usando FrDCT inversa"""
    try:
        from processing.frdct import decrypt_image, array_to_base64
        
        data = request.get_json()
        encrypted_data = data.get('encrypted_data')
        alpha = float(data.get('alpha', 0.5))

        # Validar rango de alpha (0 <= α < 2, ya que α=2 produce matriz singular)
        if alpha < 0.0 or alpha >= 2.0:
            return jsonify({'success': False, 'error': 'α debe estar en el rango [0, 2). α=2 produce una matriz singular no invertible.'}), 400
        
        if not encrypted_data:
            return jsonify({'success': False, 'error': 'No hay datos cifrados'}), 400
        
        # Convertir lista a array numpy
        encrypted_array = np.array(encrypted_data, dtype=np.float64)
        
        print(f"🔓 Descifrando imagen {encrypted_array.shape} con α = {alpha}")
        
        # Aplicar FrDCT inversa (descifrar)
        decrypted = decrypt_image(encrypted_array, alpha, use_fast=True)
        
        # Clip a rango válido y convertir a uint8
        decrypted_clipped = np.clip(decrypted, 0, 255)
        
        # Convertir a base64
        decrypted_base64 = array_to_base64(decrypted_clipped, normalize=False)
        
        print(f"✅ Descifrado completado")
        
        return jsonify({
            'success': True,
            'decrypted_image': decrypted_base64,
            'alpha_used': alpha
        })
    except Exception as e:
        print(f"Error en frdct/decrypt: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== ENCRIPTACIÓN COMPLETA (FrDCT + DOST) ====================

@app.route('/encryption')
def encryption_page():
    """Página de encriptación RGB con FrDCT + DOST"""
    return render_template('encryption.html')


@app.route('/api/encryption/crop', methods=['POST'])
def api_encryption_crop():
    """
    Recorta una imagen a un cuadrado seleccionado por el usuario.
    Recibe: image (base64), x, y, size (tamaño del cuadrado)
    """
    try:
        from PIL import Image
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        image_base64 = data.get('image')
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        size = int(data.get('size', 0))
        
        if not image_base64:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        if size <= 0:
            return jsonify({'success': False, 'error': 'Size must be greater than 0'}), 400
        
        # Decodificar imagen
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        try:
            img_bytes = base64.b64decode(image_base64)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to decode image: {str(e)}'}), 400
        
        orig_width, orig_height = pil_img.size
        
        # Validar que el recorte esté dentro de los límites
        if x < 0 or y < 0:
            return jsonify({'success': False, 'error': 'Coordinates must be >= 0'}), 400
        if x + size > orig_width or y + size > orig_height:
            return jsonify({'success': False, 'error': f'Crop area exceeds image bounds. Image: {orig_width}x{orig_height}, Crop: ({x},{y}) + {size}'}), 400
        
        # Recortar imagen (PIL usa box = (left, upper, right, lower))
        cropped = pil_img.crop((x, y, x + size, y + size))
        
        # Convertir a base64
        buffer = io.BytesIO()
        cropped.save(buffer, format='PNG')
        buffer.seek(0)
        cropped_base64 = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print(f"✂️ Image cropped from {orig_width}x{orig_height} to {size}x{size} at ({x},{y})")
        
        return jsonify({
            'success': True,
            'cropped_image': cropped_base64,
            'original_size': {'width': orig_width, 'height': orig_height},
            'crop': {'x': x, 'y': y, 'size': size}
        })
        
    except Exception as e:
        print(f"Error in encryption/crop: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/encryption/encrypt', methods=['POST'])
def api_encryption_encrypt():
    """
    Encripta una imagen RGB siguiendo el Algorithm 5:
    Step-1: Split RGB into R, G, B planes
    Step-2: Apply FrDCT with α1, α2, α3 to each plane
    Step-3: Apply DOST to each FrDCT result
    Step-4: Apply Arnold Transform with (a, k) for encryption
    Step-5: Concatenate to obtain RGB encrypted image
    """
    try:
        from processing.encryption import encrypt_image_rgb
        from PIL import Image
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
            
        image_base64 = data.get('image')
        
        # Obtener alphas para cada canal
        alpha_r = float(data.get('alpha_r', data.get('alpha', 0.5)))
        alpha_g = float(data.get('alpha_g', data.get('alpha', 0.5)))
        alpha_b = float(data.get('alpha_b', data.get('alpha', 0.5)))
        
        # Obtener parámetros de Arnold Transform
        arnold_a = int(data.get('arnold_a', 1))
        arnold_k = int(data.get('arnold_k', 1))
        
        if not image_base64:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        # Validar alphas
        for alpha, name in [(alpha_r, 'α_R'), (alpha_g, 'α_G'), (alpha_b, 'α_B')]:
            if alpha < 0.0 or alpha >= 2.0:
                return jsonify({'success': False, 'error': f'{name} must be in range [0, 2)'}), 400
        
        # Decodificar imagen como RGB
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        try:
            img_bytes = base64.b64decode(image_base64)
            pil_img = Image.open(io.BytesIO(img_bytes))
            rgb_array = np.array(pil_img.convert('RGB'), dtype=np.float64)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to decode image: {str(e)}'}), 400
        
        # Auto-recortar si la imagen no es cuadrada (requerido por Arnold Transform)
        height, width = rgb_array.shape[:2]
        if height != width:
            min_dim = min(height, width)
            # Recortar desde el centro
            start_y = (height - min_dim) // 2
            start_x = (width - min_dim) // 2
            rgb_array = rgb_array[start_y:start_y + min_dim, start_x:start_x + min_dim]
            print(f"✂️ Auto-recortada de {width}×{height} a {min_dim}×{min_dim}")
        
        print(f"🔐 Encrypting RGB image {rgb_array.shape} with α_R={alpha_r}, α_G={alpha_g}, α_B={alpha_b}, Arnold(a={arnold_a}, k={arnold_k})")
        
        # Algorithm 5: FrDCT → DOST → Arnold
        enc_result = encrypt_image_rgb(rgb_array, alpha_r, alpha_g, alpha_b, arnold_a, arnold_k)
        
        # Convertir a diccionario con imágenes base64
        result_dict = enc_result.to_dict()
        
        # Guardar datos cifrados completos (para desencriptación perfecta)
        encrypted_data = {
            'r_real': np.real(enc_result.encrypted_r).tolist(),
            'r_imag': np.imag(enc_result.encrypted_r).tolist(),
            'g_real': np.real(enc_result.encrypted_g).tolist(),
            'g_imag': np.imag(enc_result.encrypted_g).tolist(),
            'b_real': np.real(enc_result.encrypted_b).tolist(),
            'b_imag': np.imag(enc_result.encrypted_b).tolist(),
            'alpha_r': float(alpha_r),
            'alpha_g': float(alpha_g),
            'alpha_b': float(alpha_b),
            'arnold_a': int(arnold_a),
            'arnold_k': int(arnold_k)
        }
        
        print(f"✅ RGB Encryption completed successfully")
        
        return jsonify({
            'success': True,
            'original': result_dict['original'],
            'after_frdct': result_dict['after_frdct'],
            'after_dost': result_dict['after_dost'],
            'encrypted': result_dict['encrypted'],
            'encrypted_data': encrypted_data,
            'params': {
                'alpha_r': float(alpha_r),
                'alpha_g': float(alpha_g),
                'alpha_b': float(alpha_b),
                'arnold_a': int(arnold_a),
                'arnold_k': int(arnold_k)
            }
        })
        
    except Exception as e:
        print(f"Error in encryption/encrypt: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/encryption/decrypt', methods=['POST'])
def api_encryption_decrypt():
    """
    Desencripta una imagen RGB (Algorithm 6 - inverso del Algorithm 5):
    Step-1: Split encrypted RGB into R, G, B planes
    Step-2: Apply inverse Arnold Transform with same (a, k)
    Step-3: Apply inverse DOST to each channel
    Step-4: Apply inverse FrDCT with same α values
    Step-5: Concatenate to obtain RGB decrypted image
    """
    try:
        from processing.encryption import decrypt_image_rgb
        from PIL import Image
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
            
        encrypted_data = data.get('encrypted_data')
        image_base64 = data.get('image')
        
        # Obtener alphas
        alpha_r = float(data.get('alpha_r', data.get('alpha', 0.5)))
        alpha_g = float(data.get('alpha_g', data.get('alpha', 0.5)))
        alpha_b = float(data.get('alpha_b', data.get('alpha', 0.5)))
        
        # Obtener parámetros de Arnold Transform
        arnold_a = int(data.get('arnold_a', 1))
        arnold_k = int(data.get('arnold_k', 1))
        
        encrypted_r = None
        encrypted_g = None
        encrypted_b = None
        
        # Si tenemos datos cifrados completos RGB (de una encriptación previa)
        if encrypted_data and isinstance(encrypted_data, dict):
            if 'r_real' in encrypted_data:
                try:
                    encrypted_r = np.array(encrypted_data['r_real']) + 1j * np.array(encrypted_data['r_imag'])
                    encrypted_g = np.array(encrypted_data['g_real']) + 1j * np.array(encrypted_data['g_imag'])
                    encrypted_b = np.array(encrypted_data['b_real']) + 1j * np.array(encrypted_data['b_imag'])
                    
                    # NO sobrescribir los parámetros del request con los del archivo .enc
                    # El usuario puede modificar los parámetros en la UI y queremos respetarlos
                    # Los parámetros alpha_r, alpha_g, alpha_b, arnold_a, arnold_k 
                    # ya fueron obtenidos del request arriba
                except Exception as e:
                    print(f"Warning: Could not parse RGB encrypted_data: {e}")
        
        # Si solo tenemos una imagen (sin datos complejos)
        if encrypted_r is None and image_base64:
            try:
                if ',' in image_base64:
                    image_base64 = image_base64.split(',')[1]
                
                img_bytes = base64.b64decode(image_base64)
                pil_img = Image.open(io.BytesIO(img_bytes))
                rgb_array = np.array(pil_img.convert('RGB'), dtype=np.float64)
                
                # Tratar cada canal como datos complejos (solo magnitud)
                encrypted_r = rgb_array[:, :, 0].astype(np.complex128)
                encrypted_g = rgb_array[:, :, 1].astype(np.complex128)
                encrypted_b = rgb_array[:, :, 2].astype(np.complex128)
            except Exception as e:
                return jsonify({'success': False, 'error': f'Failed to decode image: {str(e)}'}), 400
        
        if encrypted_r is None:
            return jsonify({'success': False, 'error': 'No encrypted data or image provided'}), 400
        
        print(f"🔓 Decrypting RGB image with α_R={alpha_r}, α_G={alpha_g}, α_B={alpha_b}, Arnold(a={arnold_a}, k={arnold_k})")
        
        # Algorithm 6 (inverse): Arnold⁻¹ → IDOST → IFrDCT
        dec_result = decrypt_image_rgb(encrypted_r, encrypted_g, encrypted_b, alpha_r, alpha_g, alpha_b, arnold_a, arnold_k)
        
        # Convertir a diccionario con imágenes base64
        result_dict = dec_result.to_dict()
        
        print(f"✅ RGB Decryption completed successfully")
        
        return jsonify({
            'success': True,
            'encrypted': result_dict['encrypted'],
            'after_arnold_inv': result_dict['after_arnold_inv'],
            'after_idost': result_dict['after_idost'],
            'after_ifrdct': result_dict['after_ifrdct'],
            'decrypted': result_dict['decrypted'],
            'params': {
                'alpha_r': float(alpha_r),
                'alpha_g': float(alpha_g),
                'alpha_b': float(alpha_b),
                'arnold_a': int(arnold_a),
                'arnold_k': int(arnold_k)
            }
        })
        
    except Exception as e:
        print(f"Error in encryption/decrypt: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/encryption/compress', methods=['POST'])
def api_encryption_compress():
    """
    Comprime una imagen usando DCT-2D a múltiples niveles (30%, 50%, 80%).
    Retorna la imagen original y las versiones comprimidas.
    """
    try:
        from processing.encryption import comprimir_imagen_multiples_niveles
        from PIL import Image
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
            
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        # Decodificar imagen
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        try:
            img_bytes = base64.b64decode(image_base64)
            pil_img = Image.open(io.BytesIO(img_bytes))
            rgb_array = np.array(pil_img.convert('RGB'), dtype=np.uint8)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to decode image: {str(e)}'}), 400
        
        print(f"📦 Compressing image {rgb_array.shape} at 30%, 50%, 80%")
        
        # Comprimir a múltiples niveles
        result = comprimir_imagen_multiples_niveles(rgb_array)
        
        print(f"✅ Compression completed successfully")
        
        return jsonify({
            'success': True,
            'original': result['original'],
            'compressed_30': result['compressed_30'],
            'compressed_50': result['compressed_50'],
            'compressed_80': result['compressed_80']
        })
        
    except Exception as e:
        print(f"Error in compression: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# YOLOv8 OBJECT DETECTION/SEGMENTATION
# ============================================================================

# Cargar modelo YOLOv8 (lazy loading)
yolo_model = None
yolo_seg_model = None

def get_yolo_model():
    """Carga el modelo YOLOv8 de detección (lazy loading)"""
    global yolo_model
    if yolo_model is None:
        try:
            from ultralytics import YOLO
            # Usar modelo large para mejor precisión (yolov8l.pt)
            yolo_model = YOLO('yolov8l.pt')
            print("✅ YOLOv8 detection model loaded (large)")
        except Exception as e:
            print(f"⚠️ Error loading YOLOv8: {e}")
            return None
    return yolo_model

def get_yolo_seg_model():
    """Carga el modelo YOLOv8 de segmentación (lazy loading)"""
    global yolo_seg_model
    if yolo_seg_model is None:
        try:
            from ultralytics import YOLO
            # Usar modelo large de segmentación para mejor precisión
            yolo_seg_model = YOLO('yolov8l-seg.pt')
            print("✅ YOLOv8 segmentation model loaded (large)")
        except Exception as e:
            print(f"⚠️ Error loading YOLOv8-seg: {e}")
            return None
    return yolo_seg_model

@app.route('/api/yolo/detect', methods=['POST'])
def yolo_detect():
    """
    Detecta objetos en una imagen usando YOLOv8.
    Retorna bounding boxes, máscaras y clases detectadas.
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        image_base64 = data['image']
        mode = data.get('mode', 'box')  # 'box' o 'mask'
        confidence = data.get('confidence', 0.25)
        
        # Decodificar imagen
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        img_bytes = base64.b64decode(image_base64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array = np.array(pil_img)
        
        # Elegir modelo según modo
        if mode == 'mask':
            model = get_yolo_seg_model()
        else:
            model = get_yolo_model()
        
        if model is None:
            return jsonify({'success': False, 'error': 'YOLOv8 model not available'}), 500
        
        # Ejecutar inferencia
        results = model(img_array, conf=confidence, verbose=False)[0]
        
        detections = []
        masks_data = []
        
        # Procesar resultados
        if results.boxes is not None:
            boxes = results.boxes
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detection = {
                    'id': i,
                    'class': cls_name,
                    'class_id': cls_id,
                    'confidence': round(conf, 3),
                    'box': {
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1)
                    }
                }
                
                # Agregar máscara si está disponible
                if mode == 'mask' and results.masks is not None and i < len(results.masks):
                    mask = results.masks[i].data[0].cpu().numpy()
                    # Redimensionar máscara al tamaño de la imagen
                    import cv2
                    mask_resized = cv2.resize(mask, (pil_img.width, pil_img.height))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                    
                    # Convertir máscara a base64 PNG
                    mask_img = Image.fromarray(mask_binary, mode='L')
                    mask_buffer = io.BytesIO()
                    mask_img.save(mask_buffer, format='PNG')
                    mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
                    detection['mask'] = f"data:image/png;base64,{mask_base64}"
                
                detections.append(detection)
        
        # Generar imagen con anotaciones
        annotated_img = results.plot()
        annotated_pil = Image.fromarray(annotated_img)
        
        # Convertir a base64
        buffer = io.BytesIO()
        annotated_pil.save(buffer, format='PNG')
        annotated_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections),
            'annotated_image': f"data:image/png;base64,{annotated_base64}",
            'original_image': data['image'],  # Devolver imagen original para re-renderizar
            'image_size': {'width': pil_img.width, 'height': pil_img.height}
        })
        
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/yolo/render', methods=['POST'])
def yolo_render():
    """
    Re-renderiza la imagen con solo las detecciones visibles (toggle visibility).
    """
    try:
        import cv2
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        image_base64 = data['image']
        detections = data.get('detections', [])
        visible_ids = data.get('visible_ids', [])  # IDs de detecciones visibles
        mode = data.get('mode', 'box')  # 'box' o 'mask'
        
        # Decodificar imagen
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        img_bytes = base64.b64decode(image_base64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array = np.array(pil_img).copy()
        
        # Colores para cada clase (generados por hash del nombre)
        def get_color(class_name):
            import hashlib
            hash_val = int(hashlib.md5(class_name.encode()).hexdigest()[:6], 16)
            return ((hash_val >> 16) & 255, (hash_val >> 8) & 255, hash_val & 255)
        
        # Dibujar solo las detecciones visibles
        for det in detections:
            if det['id'] not in visible_ids:
                continue
            
            color = get_color(det['class'])
            box = det['box']
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            
            if mode == 'mask' and 'mask' in det:
                # Dibujar máscara semitransparente
                mask_base64 = det['mask']
                if ',' in mask_base64:
                    mask_base64 = mask_base64.split(',')[1]
                
                mask_bytes = base64.b64decode(mask_base64)
                mask_img = Image.open(io.BytesIO(mask_bytes)).convert('L')
                mask_array = np.array(mask_img)
                
                # Crear overlay coloreado
                mask_bool = mask_array > 127
                # Aplicar color solo donde la máscara es True
                for c in range(3):
                    img_array[:, :, c] = np.where(
                        mask_bool,
                        (img_array[:, :, c] * 0.5 + color[c] * 0.5).astype(np.uint8),
                        img_array[:, :, c]
                    )
                
                # Dibujar contorno de la máscara
                contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_array, contours, -1, color, 2)
            else:
                # Dibujar bounding box
                cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
            
            # Dibujar etiqueta
            label = f"{det['class']} {det['confidence']:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Fondo de la etiqueta
            cv2.rectangle(img_array, (x1, y1 - text_height - 10), (x1 + text_width + 4, y1), color, -1)
            cv2.putText(img_array, label, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), thickness)
        
        # Convertir a base64
        result_img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        result_img.save(buffer, format='PNG')
        result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'rendered_image': f"data:image/png;base64,{result_base64}"
        })
        
    except Exception as e:
        print(f"Error in YOLO render: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print(f"🎤 Aplicación iniciada. Modelos cargados: {list(models.keys())}")
    print(f"📊 Parámetros: FS={FS} Hz, TARGET_N={TARGET_N}")
    print("🌐 Abre http://localhost:5000 en tu navegador")
    print("📱 El proyecto está configurado para funcionar SIN internet")
    
    # Configurar timeout para requests grandes
    from werkzeug.serving import WSGIRequestHandler
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
