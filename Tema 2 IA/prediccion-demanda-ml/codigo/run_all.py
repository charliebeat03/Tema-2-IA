#!/usr/bin/env python3
"""
Script para ejecutar todo el pipeline de ML sin dependencias externas
"""
import subprocess
import sys
import os

def run_script(script_name):
    """Ejecuta un script y maneja errores"""
    print(f"\n{'='*60}")
    print(f"Ejecutando: {script_name}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True)
        if result.returncode != 0:
            print(f"❌ Error ejecutando {script_name}")
            return False
        print(f"✓ {script_name} completado exitosamente")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Ejecuta el pipeline completo"""
    scripts = [
        'generar_dataset.py',
        'entrenar_modelo.py',
        'predecir.py'
    ]
    
    print("\n🚀 INICIANDO PIPELINE DE PREDICCIÓN DE DEMANDA")
    print("=" * 60)
    
    all_success = True
    for script in scripts:
        if not run_script(script):
            all_success = False
            break
    
    if all_success:
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("\n📊 Resumen:")
        print("  1. Dataset generado: dataset/ventas_panaderia.csv")
        print("  2. Modelo entrenado: model/modelo.json")
        print("  3. Predicción realizada correctamente")
        return 0
    else:
        print("\n" + "="*60)
        print("❌ PIPELINE FALLÓ")
        print("="*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())
