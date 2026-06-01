import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
fechas = []
productos = []
ventas = []
dias_semana = []
meses = []

inicio = datetime(2024, 1, 1)
for i in range(365):
    fecha = inicio + timedelta(days=i)
    for producto in ['Pan francés', 'Pan dulce', 'Galletas', 'Pastel']:
        # Base de ventas diferente por producto
        base = {'Pan francés': 40, 'Pan dulce': 25, 'Galletas': 15, 'Pastel': 10}[producto]
        # Efecto día de la semana (viernes y sábado más ventas)
        dia = fecha.weekday()  # 0=lunes, 6=domingo
        factor_dia = 1.5 if dia in [4,5] else 0.8 if dia == 6 else 1.0
        # Efecto mes (diciembre más ventas)
        mes = fecha.month
        factor_mes = 1.3 if mes == 12 else 1.0
        # Ruido aleatorio
        ruido = np.random.normal(0, 0.1)
        venta = max(0, int(base * factor_dia * factor_mes * (1 + ruido)))
        
        fechas.append(fecha.strftime('%Y-%m-%d'))
        productos.append(producto)
        ventas.append(venta)
        dias_semana.append(dia)
        meses.append(mes)

df = pd.DataFrame({
    'fecha': fechas,
    'producto': productos,
    'dia_semana': dias_semana,
    'mes': meses,
    'ventas': ventas
})
df.to_csv('../dataset/ventas_panaderia.csv', index=False)
print("Dataset generado: dataset/ventas_panaderia.csv")