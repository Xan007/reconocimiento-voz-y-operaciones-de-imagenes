/**
 * Utilidades compartidas para configuración de audio y micrófono
 * AHORA USA EL MICRÓFONO GUARDADO EN TODOS LADOS
 */

/**
 * Obtiene el deviceId del micrófono seleccionado usando enumerateDevices
 * Mapea el índice guardado al deviceId real de la Web Audio API
 * 
 * @returns {Promise<string|null>} deviceId o null para usar predeterminado
 */
async function getSelectedMicrophoneDeviceId() {
    try {
        const microphoneIndex = localStorage.getItem('microphone_id');
        
        if (!microphoneIndex || microphoneIndex === '') {
            console.log('🎤 Usando micrófono predeterminado');
            return null;
        }
        
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter(device => device.kind === 'audioinput');
        
        // Obtener el deviceId del índice guardado
        const selectedDevice = audioInputs[parseInt(microphoneIndex)];
        
        if (selectedDevice) {
            console.log(`🎤 Usando micrófono: ${selectedDevice.label || 'Micrófono'} (${selectedDevice.deviceId})`);
            return selectedDevice.deviceId;
        } else {
            console.warn(`⚠️ Micrófono índice ${microphoneIndex} no encontrado, usando predeterminado`);
            return null;
        }
    } catch (error) {
        console.error('❌ Error obteniendo dispositivos:', error);
        return null;
    }
}

/**
 * Obtiene las opciones de micrófono para getUserMedia
 * Usa el micrófono guardado o el predeterminado
 * 
 * @returns {Promise<Object>} Opciones de audio para getUserMedia
 */
async function getAudioOptions() {
    const deviceId = await getSelectedMicrophoneDeviceId();
    
    if (deviceId) {
        return {
            audio: {
                deviceId: { exact: deviceId },
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            }
        };
    } else {
        return {
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            }
        };
    }
}

/**
 * Solicita acceso al micrófono usando la configuración guardada
 * 
 * @returns {Promise<MediaStream|null>} Stream de audio o null si falla
 */
async function requestMicrophoneAccess() {
    try {
        const options = await getAudioOptions();
        const stream = await navigator.mediaDevices.getUserMedia(options);
        console.log('✅ Acceso a micrófono concedido');
        return stream;
    } catch (error) {
        console.error('❌ Error accediendo al micrófono:', error);
        
        // Proporcionar mensajes más informativos según el tipo de error
        let errorMsg = 'Error accediendo al micrófono: ';
        switch(error.name) {
            case 'NotAllowedError':
                errorMsg += 'Permiso denegado. Verifica los permisos del navegador.';
                break;
            case 'NotFoundError':
                errorMsg += 'No se encontró el micrófono configurado. Intenta con el predeterminado.';
                console.warn('⚠️ Reintentando con micrófono predeterminado...');
                try {
                    const fallbackStream = await navigator.mediaDevices.getUserMedia({ 
                        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
                    });
                    console.log('✅ Usando micrófono predeterminado');
                    return fallbackStream;
                } catch (e) {
                    console.error('❌ Tampoco funcionó el micrófono predeterminado:', e);
                    return null;
                }
                break;
            case 'NotReadableError':
                errorMsg += 'El micrófono está siendo usado por otra aplicación.';
                break;
            case 'SecurityError':
                errorMsg += 'Acceso denegado por razones de seguridad.';
                break;
            case 'OverconstrainedError':
                errorMsg += 'Las restricciones de audio no pudieron ser satisfechas.';
                break;
            default:
                errorMsg += error.message || 'Error desconocido';
        }
        
        console.log('⚠️ ' + errorMsg);
        return null;
    }
}

/**
 * Inicia grabación de audio desde el micrófono
 * 
 * @returns {Promise<Object>} Objeto con stream y recorder, o null si falla
 */
async function startAudioRecording() {
    const stream = await requestMicrophoneAccess();
    
    if (!stream) {
        return null;
    }
    
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    
    let audioData = [];
    
    processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        audioData.push(new Float32Array(inputData));
    };
    
    source.connect(processor);
    processor.connect(audioContext.destination);
    
    return {
        stream,
        audioContext,
        processor,
        source,
        audioData,
        stop: function() {
            processor.disconnect();
            source.disconnect();
            stream.getTracks().forEach(track => track.stop());
            return new Float32Array(audioData.length * 4096);
        }
    };
}

console.log('✅ Audio configuration utilities loaded - USANDO MICRÓFONO GUARDADO EN TODOS LADOS');

