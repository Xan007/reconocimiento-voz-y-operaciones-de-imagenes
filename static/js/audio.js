/**
 * audio.js
 * Funcionalidades generales de audio y utilidades
 */

// Utilidades para manejo de audio
const AudioUtils = {
    /**
     * Convierte un Blob de audio a Float32Array
     */
    async blobToFloat32(blob) {
        const arrayBuffer = await blob.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        return await audioContext.decodeAudioData(arrayBuffer);
    },

    /**
     * Obtiene permiso del usuario para acceder al micrófono
     * Usa la configuración de micrófono guardada si está disponible
     */
    async requestMicrophoneAccess() {
        try {
            // Si la función compartida está disponible, usarla
            if (typeof getAudioOptions === 'function') {
                const options = await getAudioOptions();
                const stream = await navigator.mediaDevices.getUserMedia(options);
                return stream;
            }
            
            // Fallback si no está disponible
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            return stream;
        } catch (error) {
            console.error('Error al acceder al micrófono:', error);
            throw new Error('No se pudo acceder al micrófono. Verifica los permisos del navegador.');
        }
    },

    /**
     * Detiene todos los tracks de una stream
     */
    stopStream(stream) {
        stream.getTracks().forEach(track => track.stop());
    }
};

// Funciones de utilidad general
function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="close" onclick="this.parentElement.remove()">&times;</button>
    `;
    
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        setTimeout(() => alertDiv.remove(), 5000);
    }
}

// Inicialización cuando el documento está listo
document.addEventListener('DOMContentLoaded', () => {
    console.log('Aplicación de Reconocimiento de Comandos FFT cargada');
});
