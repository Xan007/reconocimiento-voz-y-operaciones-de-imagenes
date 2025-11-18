"""
Aplicación Flask para reconocimiento de comandos por análisis FFT.
Interfaz web para las funcionalidades de reconocimiento en tiempo real
y análisis de modelos.
"""

import os
import json
import numpy as np
import io
import base64
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import soundfile as sf
from scipy.io import wavfile

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

# Configuración de la aplicación
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB máximo
app.config['UPLOAD_FOLDER'] = os.path.join(DATA_DIR, 'uploads')

# Crear carpeta de uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

        # Normalizar y preparar
        audio_data = normalize_audio(audio_data)
        audio_data = pad_or_trim(audio_data, TARGET_N)

        # Analizar señal
        freqs, spectrum, energies = analyze_signal(audio_data)

        # Comparar con modelos
        best_cmd, diffs = compare_with_models(energies, models)

        # DEBUG: Verificar energías normalizadas
        print(f"DEBUG: Sum of energies: {np.sum(energies)}")
        print(f"DEBUG: First 3 energies: {energies[:3]}")
        print(f"DEBUG: Max energy: {np.max(energies)}")

        # Preparar gráficas
        plot_data = {
            'waveform': {
                'x': (np.arange(len(audio_data)) / FS).tolist(),
                'y': audio_data.tolist(),
            },
            'spectrum': {
                'x': freqs.tolist(),
                'y': (20 * np.log10(np.abs(spectrum) + 1e-10)).tolist(),
            },
            'energies': energies.tolist(),
            'models_comparison': {}
        }

        # Comparaciones con modelos
        for model_name, model_obj in models.items():
            plot_data['models_comparison'][model_name] = {
                'model_energies': model_obj.mean_energy.tolist(),
                'model_std': model_obj.std_energy.tolist(),
                'input_energies': energies.tolist(),
                'difference': diffs.get(model_name, 0.0)
            }

        response = {
            'success': True,
            'recognized_command': best_cmd,
            'differences': diffs,
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
    """
    try:
        if model_name not in models:
            return jsonify({'error': f'Modelo {model_name} no encontrado'}), 404

        model = models[model_name]
        n_bands = len(model.mean_energy)

        data = {
            'model_name': model_name,
            'n_bands': n_bands,
            'num_samples': model.num_samples,
            'mean_energy': model.mean_energy.tolist(),
            'std_energy': model.std_energy.tolist(),
            'total_energy': float(np.sum(model.mean_energy)),
            'bands': [f'B{i+1}' for i in range(n_bands)]
        }

        return jsonify(data)

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


@app.route('/api/config', methods=['POST'])
def update_config():
    """API para actualizar N_SUBBANDS"""
    try:
        data = request.get_json()
        n_subbands = int(data.get('N_SUBBANDS', 8))
        
        if n_subbands < 2 or n_subbands > 32:
            return jsonify({'success': False, 'error': 'N_SUBBANDS debe estar entre 2 y 32'}), 400
        
        # Actualizar config.py
        config_path = os.path.join(os.path.dirname(__file__), 'config.py')
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Reemplazar N_SUBBANDS
        import re
        content = re.sub(r'N_SUBBANDS = \d+', f'N_SUBBANDS = {n_subbands}', content)
        
        with open(config_path, 'w') as f:
            f.write(content)
        
        return jsonify({'success': True, 'message': 'Subbandas actualizadas. Entrenar modelo ahora.'})
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
    """API para ejecutar evaluación"""
    try:
        from evaluate_model import evaluate_model
        results = evaluate_model()
        
        if results is None:
            return jsonify({'success': False, 'error': 'No se pudo completar la evaluación'}), 500
        
        return jsonify({'success': True, 'results': results})
    except Exception as e:
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


if __name__ == '__main__':
    print(f"🎤 Aplicación iniciada. Modelos cargados: {list(models.keys())}")
    print(f"📊 Parámetros: FS={FS} Hz, TARGET_N={TARGET_N}")
    print("🌐 Abre http://localhost:5000 en tu navegador")
    app.run(debug=True, host='0.0.0.0', port=5000)
