# Predicción de Demanda - Machine Learning

## Descripción
Este proyecto implementa un sistema de predicción de demanda para una panadería usando Machine Learning. **No requiere dependencias externas**, utiliza solo librerías estándar de Python.

## Requisitos
- Python 3.6+
- Sin dependencias externas

## Estructura del Proyecto
```
prediccion-demanda-ml/
├── codigo/
│   ├── generar_dataset.py      # Genera datos sintéticos de ventas
│   ├── entrenar_modelo.py      # Entrena modelo de regresión lineal
│   └── predecir.py             # Realiza predicciones
├── dataset/
│   └── ventas_panaderia.csv    # Dataset de entrenamiento (1460 registros)
└── model/
    └── modelo.json             # Modelo entrenado (formato JSON)
```

## Uso Rápido

### 1. Generar Dataset
```bash
cd prediccion-demanda-ml/codigo
python generar_dataset.py
```
Genera un dataset con 1 año de datos sintéticos de ventas de panadería (1460 registros).

### 2. Entrenar Modelo
```bash
python entrenar_modelo.py
```
Entrena un modelo de regresión lineal y muestra métricas:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

### 3. Hacer Predicciones
```bash
python predecir.py
```
Realiza predicción de ejemplo: ventas para un viernes de diciembre.

## Tecnología
- **Algoritmo**: Regresión Lineal Múltiple
- **Features**: Día de la semana y Mes
- **Serialización**: JSON (sin pickle o joblib)
- **Librerías**: Solo estándar (csv, json, random, math, datetime, os)

## Características del Modelo
- Coeficientes almacenados en JSON
- Predicción rápida sin dependencias
- Fácil de desplegar en cualquier sistema con Python 3

## Métricas de Desempeño
- MAE: 13.33
- RMSE: 17.05
- R²: -0.0368 
