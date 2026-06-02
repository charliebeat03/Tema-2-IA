import csv
import math
import json
import os

# Crear directorio de modelo si no existe
os.makedirs('../model', exist_ok=True)

# Cargar dataset
X = []  # Features: [dia_semana, mes]
y = []  # Target: ventas

with open('../dataset/ventas_panaderia.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        X.append([int(row['dia_semana']), int(row['mes'])])
        y.append(int(row['ventas']))

# División 80/20 (simple)
split_idx = int(len(X) * 0.8)
X_train = X[:split_idx]
y_train = y[:split_idx]
X_test = X[split_idx:]
y_test = y[split_idx:]

# Implementar Regresión Lineal Múltiple manualmente
def regresion_lineal(X, y):
    """Entrena un modelo de regresión lineal usando mínimos cuadrados"""
    n = len(X)
    features = len(X[0])
    
    # Añadir columna de 1s para intercepto
    X_aug = [[1] + row for row in X]
    
    # Calcular sumas necesarias para mínimos cuadrados
    sum_x = [[0] * (features + 1) for _ in range(features + 1)]
    sum_xy = [0] * (features + 1)
    
    for i in range(n):
        for j in range(features + 1):
            for k in range(features + 1):
                sum_x[j][k] += X_aug[i][j] * X_aug[i][k]
            sum_xy[j] += X_aug[i][j] * y[i]
    
    # Resolver sistema lineal (inversión de matriz 3x3)
    # Usar método de Gauss
    A = [row[:] for row in sum_x]
    b = sum_xy[:]
    
    # Eliminación de Gauss
    for i in range(features + 1):
        # Pivote
        max_row = i
        for k in range(i + 1, features + 1):
            if abs(A[k][i]) > abs(A[max_row][i]):
                max_row = k
        A[i], A[max_row] = A[max_row], A[i]
        b[i], b[max_row] = b[max_row], b[i]
        
        # Eliminación
        for k in range(i + 1, features + 1):
            if A[i][i] != 0:
                factor = A[k][i] / A[i][i]
                for j in range(i, features + 1):
                    A[k][j] -= factor * A[i][j]
                b[k] -= factor * b[i]
    
    # Sustitución hacia atrás
    coef = [0] * (features + 1)
    for i in range(features, -1, -1):
        coef[i] = b[i]
        for j in range(i + 1, features + 1):
            coef[i] -= A[i][j] * coef[j]
        if A[i][i] != 0:
            coef[i] /= A[i][i]
    
    return coef

# Entrenar modelo
coef = regresion_lineal(X_train, y_train)

# Función para predecir
def predecir(X, coef):
    predicciones = []
    for x in X:
        pred = coef[0] + sum(coef[i+1] * x[i] for i in range(len(x)))
        predicciones.append(pred)
    return predicciones

# Predicciones
y_pred = predecir(X_test, coef)

# Calcular métricas
def mean_absolute_error(y_true, y_pred):
    return sum(abs(y_true[i] - y_pred[i]) for i in range(len(y_true))) / len(y_true)

def rmse(y_true, y_pred):
    return math.sqrt(sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / len(y_true))

def r2_score(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
    ss_tot = sum((y_true[i] - mean_y) ** 2 for i in range(len(y_true)))
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

mae = mean_absolute_error(y_test, y_pred)
rmse_val = rmse(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse_val:.2f}")
print(f"R²: {r2:.4f}")

# Guardar modelo como JSON
modelo_data = {
    'tipo': 'regresion_lineal',
    'coeficientes': coef,
    'features': ['intercepto', 'dia_semana', 'mes'],
    'metricas': {
        'mae': mae,
        'rmse': rmse_val,
        'r2': r2
    }
}

with open('../model/modelo.json', 'w') as f:
    json.dump(modelo_data, f, indent=2)

print("Modelo guardado en model/modelo.json")