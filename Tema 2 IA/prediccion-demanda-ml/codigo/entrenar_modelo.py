import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Cargar dataset
df = pd.read_csv('../dataset/ventas_panaderia.csv')

# Feature engineering: usamos solo características temporales
X = df[['dia_semana', 'mes']]
y = df['ventas']

# Dividir en entrenamiento y prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)

# Métricas
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# Guardar modelo
joblib.dump(modelo, '../modelos/modelo_random_forest.pkl')
print("Modelo guardado en modelos/modelo_random_forest.pkl")

# Gráfico de importancia de características
importancias = modelo.feature_importances_
plt.barh(['Día de la semana', 'Mes'], importancias)
plt.xlabel('Importancia')
plt.title('Importancia de las variables')
plt.tight_layout()
plt.savefig('../capturas/importancia_variables.png')
plt.show()

# Gráfico de predicciones vs reales (muestra)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Ventas reales')
plt.ylabel('Ventas predichas')
plt.title('Predicciones vs Reales')
plt.tight_layout()
plt.savefig('../capturas/pred_vs_real.png')
plt.show()