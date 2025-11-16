#!/usr/bin/env python
"""
Script de diagnóstico de dispositivos de audio
Muestra TODOS los dispositivos, clasificados por tipo
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sounddevice as sd

def main():
    print("\n" + "="*80)
    print("  DIAGNÓSTICO COMPLETO DE DISPOSITIVOS DE AUDIO")
    print("="*80 + "\n")
    
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    if not isinstance(devices, list):
        print("Error: query_devices() no retornó lista")
        return 1
    
    print(f"Total de dispositivos encontrados: {len(devices)}\n")
    
    input_devices = []
    output_devices = []
    
    for i, device in enumerate(devices):
        if not isinstance(device, dict):
            print(f"[{i}] ERROR: No es diccionario")
            continue
        
        max_in = device.get('max_input_channels', 0)
        max_out = device.get('max_output_channels', 0)
        name = device.get('name', f'Dispositivo {i}')
        default_in = device.get('default_input', False)
        default_out = device.get('default_output', False)
        
        if max_in > 0:
            input_devices.append((i, name, max_in))
        if max_out > 0:
            output_devices.append((i, name, max_out))
        
        device_type = []
        if max_in > 0:
            device_type.append(f"ENTRADA ({max_in} ch)")
        if max_out > 0:
            device_type.append(f"SALIDA ({max_out} ch)")
        
        default_marker = ""
        if default_in:
            default_marker += " [DEFAULT INPUT]"
        if default_out:
            default_marker += " [DEFAULT OUTPUT]"
        
        print(f"[{i:2d}] {name:50s} | {' | '.join(device_type):25s} {default_marker}")
    
    print("\n" + "="*80)
    print("RESUMEN:")
    print("="*80)
    print(f"\nDisositivos de ENTRADA (Micrófonos):")
    if input_devices:
        for device_id, name, channels in input_devices:
            print(f"  [{device_id}] {name} ({channels} canal{'es' if channels > 1 else ''})")
    else:
        print("  ❌ NINGUNO ENCONTRADO")
    
    print(f"\nDispositivos de SALIDA (Speakers/Headphones):")
    if output_devices:
        for device_id, name, channels in output_devices:
            print(f"  [{device_id}] {name} ({channels} canal{'es' if channels > 1 else ''})")
    else:
        print("  ❌ NINGUNO ENCONTRADO")
    
    print("\n" + "="*80 + "\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
