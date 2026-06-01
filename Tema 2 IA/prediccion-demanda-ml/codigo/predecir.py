import joblib
import numpy as np

modelo = joblib.load('../modelos/modelo_random_forest.pkl')

# Ejemplo: predecir ventas para un viernes (día 4) de diciembre (mes 12)
nuevo = np.array([[4, 12]])  # día_semana=4 (viernes), mes=12 (diciembre)
prediccion = modelo.predict(nuevo)
print(f"Para un viernes de diciembre, se predicen {prediccion[0]:.0f} unidades vendidas.")