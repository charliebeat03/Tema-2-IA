import csv
import random
from datetime import datetime, timedelta

random.seed(42)
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
        # Ruido aleatorio (distribución normal simulada)
        ruido = sum(random.gauss(0, 0.1) for _ in range(1))
        venta = max(0, int(base * factor_dia * factor_mes * (1 + ruido)))
        
        fechas.append(fecha.strftime('%Y-%m-%d'))
        productos.append(producto)
        ventas.append(venta)
        dias_semana.append(dia)
        meses.append(mes)

# Crear directorio si no existe
import os
os.makedirs('../dataset', exist_ok=True)

# Guardar como CSV
with open('../dataset/ventas_panaderia.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['fecha', 'producto', 'dia_semana', 'mes', 'ventas'])
    for i in range(len(fechas)):
        writer.writerow([fechas[i], productos[i], dias_semana[i], meses[i], ventas[i]])

print("Dataset generado: dataset/ventas_panaderia.csv")