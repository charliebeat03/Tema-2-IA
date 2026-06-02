import json

# Cargar modelo
with open('../model/modelo.json', 'r') as f:
    modelo_data = json.load(f)

coef = modelo_data['coeficientes']

# Función para predecir
def predecir(X, coef):
    predicciones = []
    for x in X:
        pred = coef[0] + sum(coef[i+1] * x[i] for i in range(len(x)))
        predicciones.append(pred)
    return predicciones

# Ejemplo: predecir ventas para un viernes (día 4) de diciembre (mes 12)
nuevo = [[4, 12]]  # día_semana=4 (viernes), mes=12 (diciembre)
prediccion = predecir(nuevo, coef)
print(f"Para un viernes de diciembre, se predicen {prediccion[0]:.0f} unidades vendidas.")