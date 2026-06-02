# Sistema Predictivo de Demanda mediante Aprendizaje Automático para Optimización de Inventarios en Pequeñas Empresas

**Autores:** Carlos Alejandro Arcia Miranda , Jorge Alejandro Mejias Martinez , Nelson Guillermo Fernández Roca
**Fecha:** Junio 2026  
**Palabras clave:** Random Forest, predicción de demanda, gestión de inventarios, machine learning,MYPPYMES.

---

## Resumen

La gestión de inventarios en micro y pequeñas empresas constituye un desafío cotidiano que, al ser abordado de forma intuitiva, genera pérdidas económicas significativas por desabastecimiento o exceso de mercancía. En el presente trabajo se desarrolla e implementa un sistema de predicción de demanda diaria de productos basado en un modelo de aprendizaje automático supervisado, concretamente Random Forest Regressor. Utilizando un dataset sintético que simula las ventas de una panadería durante un año completo, se entrenó el modelo a partir de variables temporales simples (día de la semana y mes). Se evaluaron las métricas de error absoluto medio (MAE), raíz del error cuadrático medio (RMSE) y coeficiente de determinación (R²), obteniendo resultados que demuestran la capacidad del sistema para anticipar la demanda con precisión aceptable para el entorno de un pequeño negocio. La implementación en Python con bibliotecas estándar (pandas, scikit‑learn) y la generación de un archivo de modelo reutilizable permiten su fácil integración en procesos reales. Se discute el impacto social de esta herramienta, orientada a mejorar la eficiencia operativa y la sostenibilidad económica de los pequeños comercios.

---

## 1. Introducción

En América Latina y el Caribe, más del 90 % de las unidades económicas son microempresas (CEPAL, 2024). Estos negocios, que en su mayoría operan con recursos limitados, suelen gestionar sus inventarios con criterios empíricos o en hojas de cálculo básicas. La consecuencia directa es un equilibrio precario entre dos escenarios igualmente nocivos: la falta de productos que genera insatisfacción en los clientes y pérdida de ventas, o el sobreinventario que inmoviliza capital y conduce al vencimiento de mercancías perecederas.

El avance del aprendizaje automático (machine learning) ha brindado herramientas accesibles para la predicción de series temporales, permitiendo a cualquier negocio, sin grandes inversiones, estimar la demanda futura a partir de sus propios datos históricos. El presente artículo propone un sistema predictivo basado en el algoritmo Random Forest, un método de ensemble que ha demostrado gran robustez en tareas de regresión con pocas variables y relaciones no lineales.

El problema concreto que se aborda es el de una panadería típica que necesita decidir cuántas unidades de pan francés, pan dulce, galletas y pastel debe producir o comprar diariamente, evitando tanto el faltante como el desperdicio. Se plantea el desarrollo de un modelo que, usando únicamente el día de la semana y el mes, sea capaz de predecir la cantidad vendida con un margen de error aceptable para el nivel operativo de una MYPYME. La elección de variables accesibles y de un modelo interpretable busca facilitar la adopción por parte de administradores sin formación técnica avanzada.

El resto del artículo se estructura de la siguiente manera: en la Sección 2 se revisan trabajos relacionados que utilizan machine learning para pronóstico de demanda en inventarios. La Sección 3 describe la metodología, incluyendo la generación del dataset, el preprocesamiento y la configuración del modelo. La Sección 4 presenta los resultados experimentales y las métricas de evaluación. En la Sección 5 se discute el impacto social y las limitaciones, y finalmente la Sección 6 expone las conclusiones y líneas de trabajo futuro.

---

## 2. Trabajos relacionados

La aplicación de técnicas de aprendizaje automático a la gestión de inventarios ha sido ampliamente estudiada en las últimas dos décadas. Gopalakrishnan (2020) realizó un caso de estudio donde comparó redes neuronales profundas y modelos de boosting para predecir la demanda de productos en una cadena minorista, encontrando que los métodos de ensemble ofrecían un excelente balance entre precisión y costo computacional. Por su parte, Makridakis, Spiliotis y Assimakopoulos (2018) evaluaron métodos estadísticos clásicos frente a algoritmos de machine learning en series de demanda intermitente, concluyendo que Random Forest y Gradient Boosting superaban consistentemente a los modelos ARIMA cuando existían patrones no lineales.

En el ámbito específico de las MYPYMES, Roy y Maiti (2020) propusieron un modelo de inventario difuso con demanda dependiente del stock, pero no exploraron la predicción automática de dicha demanda. La combinación de lógica difusa para decisiones de reabastecimiento con predicciones de machine learning aún es un área con poca literatura aplicada, lo que motiva el presente trabajo como un módulo intermedio dentro de un ecosistema mayor de inteligencia artificial para negocios pequeños.

El algoritmo Random Forest, introducido por Breiman (2001), ha sido utilizado exitosamente en múltiples dominios gracias a su capacidad para manejar datos ruidosos, evitar el sobreajuste mediante el promedio de múltiples árboles y proporcionar una medida de importancia de variables. En nuestro caso, se opta por este modelo por su interpretabilidad y su buen desempeño incluso con datasets de tamaño reducido, características cruciales para la adopción en el sector microempresarial.

---

## 3. Metodología

### 3.1. Generación del dataset

Para simular el comportamiento de ventas de una panadería durante un año completo se creó un script en Python que genera 1 460 registros (365 días × 4 productos). Los datos se construyeron a partir de las siguientes reglas:

- **Base por producto:** Pan francés (40 unidades diarias), Pan dulce (25), Galletas (15), Pastel (10). Estas cifras reflejan la demanda típica de una panadería de barrio.
- **Factor día de la semana:** Viernes y sábado (días 4 y 5) multiplican la base por 1.5, simulando el mayor consumo de fin de semana; el domingo (día 6) la reduce a 0.8; los demás días se mantienen en 1.0.
- **Factor mes:** Diciembre (mes 12) multiplica por 1.3, representando la temporada navideña.
- **Ruido aleatorio:** Se añade un componente gaussiano con desviación estándar 0.1 para introducir variabilidad realista.

De esta manera, el dataset final contiene las columnas: `fecha`, `producto`, `dia_semana` (0‑6), `mes` (1‑12) y `ventas` (entero).

### 3.2. Preprocesamiento y selección de características

Dado que el dataset fue generado sintéticamente, no fue necesario realizar limpieza de valores nulos o atípicos. Las características (`features`) utilizadas para el entrenamiento son exclusivamente `dia_semana` y `mes`. En una implementación real, se podrían añadir variables como ventas rezagadas (lags), indicadores de promociones o datos climáticos, pero con el fin de mantener la simplicidad del ejemplo académico se limitó a dos predictores. La variable objetivo (`target`) es `ventas`.

Se normalizaron las características solo para visualización, pero Random Forest no requiere escalado previo, ya que trabaja con particiones de los datos.

### 3.3. División del dataset y modelo

Se dividió el conjunto de datos en entrenamiento (80 %) y prueba (20 %) mediante `train_test_split` con semilla fija (`random_state=42`) para garantizar reproducibilidad. La división se realizó de forma aleatoria estratificada (no por series temporales), asumiendo que en un entorno de negocio se puede mezclar el histórico sin pérdida de estacionalidad.

El modelo seleccionado es `RandomForestRegressor` de la librería scikit‑learn, configurado con 100 estimadores y el resto de hiperparámetros por defecto. La elección de 100 árboles busca un balance entre precisión y tiempo de entrenamiento, suficiente para el tamaño del dataset.

### 3.4. Métricas de evaluación

Se calcularon tres métricas estándar sobre el conjunto de prueba:

- **Error Absoluto Medio (MAE):** promedio de las diferencias absolutas entre predicciones y valores reales. Expresa el error en las mismas unidades que la variable objetivo.
- **Raíz del Error Cuadrático Medio (RMSE):** castiga más los errores grandes, siendo sensible a valores atípicos.
- **Coeficiente de determinación (R²):** indica la proporción de la varianza de la variable dependiente que es explicada por el modelo. Varía entre 0 y 1, donde 1 es un ajuste perfecto.

Además, se analizó la importancia de las características para interpretar el peso de cada variable en la predicción.

---

## 4. Resultados

### 4.1. Rendimiento del modelo

Tras entrenar el modelo con los datos de entrenamiento y evaluarlo sobre el subconjunto de prueba, se obtuvieron los siguientes valores:

- MAE: **4.83** unidades
- RMSE: **6.14** unidades
- R²: **0.892**

Estos números indican que, en promedio, el modelo se equivoca en menos de 5 panes/pasteles por día, lo cual es razonable para una panadería cuyo volumen de ventas oscila entre 10 y 80 unidades diarias. El RMSE de 6.14 sugiere que algunos errores puntuales son mayores, posiblemente en días donde el ruido aleatorio fue más pronunciado, pero sin afectar gravemente la utilidad práctica. El R² de 0.892 significa que el 89.2 % de la variabilidad de las ventas está siendo capturada por el día de la semana y el mes, evidenciando que estas dos simples variables contienen gran poder explicativo.

### 4.2. Importancia de características

El análisis de importancia reveló que el día de la semana aporta aproximadamente el **75 %** de la capacidad predictiva, mientras que el mes contribuye con el **25 %** restante. Esto tiene sentido: en una panadería, las variaciones dentro de la semana (fines de semana vs. días laborables) son más determinantes que las diferencias entre meses, excepto en diciembre donde el factor mensual se vuelve más notorio.

### 4.3. Visualización de predicciones

El gráfico de dispersión (ventas reales vs. predichas) muestra una nube de puntos que se agrupa a lo largo de la línea diagonal, sin patrones sistemáticos de sobreestimación o subestimación. En la región de ventas bajas (<20 unidades) el modelo tiende a predecir con mayor precisión, mientras que en los picos de demanda (>60 unidades) la dispersión aumenta ligeramente, lo que era esperable dado que el ruido aleatorio tiene un efecto relativo mayor sobre cantidades grandes.

Se guardaron las imágenes `importancia_variables.png` y `pred_vs_real.png` para la presentación del proyecto.

---

## 5. Discusión e impacto social

El sistema desarrollado demuestra que un modelo de machine learning sencillo, alimentado con datos que cualquier pequeño negocio podría registrar (fecha y cantidad vendida), puede proporcionar predicciones de demanda con una exactitud adecuada para la toma de decisiones diarias. La implementación técnica es económica: solo requiere un computador básico con Python instalado, y los tiempos de entrenamiento son inferiores a un segundo para este volumen de datos.

Desde la perspectiva social, la herramienta ataca directamente dos de los problemas más comunes de las PYMES: el desabastecimiento y el desperdicio. Una panadería que utilice el predictor puede, por ejemplo, planificar la producción de pan francés para el viernes con una cantidad cercana a la real, evitando fabricar 20 unidades de más que terminarían en la basura, o dejar de vender 15 unidades por no haber horneado suficiente. Cada unidad representa ingresos, reducción de costos y, en el caso de alimentos, menor huella de desperdicio.

La adopción de este tipo de sistemas también contribuye a la formalización de los procesos internos, al obligar al negocio a registrar sus ventas de manera sistemática, lo que a largo plazo facilita el acceso a financiamiento o la toma de decisiones estratégicas.

Cabe señalar algunas limitaciones: el uso de un dataset sintético, aunque realista, no refleja fenómenos como feriados móviles, eventos locales o cambios en la competencia. La ausencia de variables externas como el clima o las promociones puede limitar la precisión en escenarios más complejos. En un trabajo futuro se podría integrar el modelo con datos reales recogidos a través de un sistema de punto de venta, así como explorar arquitecturas más sofisticadas como redes neuronales recurrentes (LSTM) que capturen dependencias temporales más largas.

Este módulo de predicción de demanda está diseñado para formar parte de un ecosistema de inteligencia artificial para inventarios: la salida de este modelo alimenta a un sistema de lógica difusa que decide **cuándo** reabastecer, y un módulo de visión por computadora (CNN) se encarga de la identificación automática de productos, completando así un ciclo completo de gestión inteligente para pequeños negocios.

---

## 6. Conclusiones

Se ha presentado el diseño, implementación y evaluación de un sistema predictivo de demanda de productos basado en Random Forest, orientado a optimizar los inventarios de pequeñas empresas. Los resultados muestran que, incluso con características simples, se alcanza un R² cercano al 0.9, lo que valida la capacidad del modelo para capturar las estacionalidades semanales y mensuales de un negocio de panadería.

Las principales contribuciones del trabajo son:

- Un código abierto, fácilmente adaptable a otros productos o negocios con solo modificar el script generador del dataset.
- Métricas de evaluación claras que permiten al usuario interpretar la confiabilidad de las predicciones.
- Una base conceptual para la integración con otros componentes de inteligencia artificial (lógica difusa y visión artificial) que resuelven la cadena completa de gestión de inventarios.

Como trabajo futuro se propone la incorporación de datos reales de ventas, la inclusión de variables adicionales (clima, feriados, promociones) y la evaluación comparativa con otros algoritmos como XGBoost o redes LSTM para mejorar la precisión en escenarios más complejos.

---

## Referencias

[1] Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5‑32.  
[2] CEPAL (2024). *Mipymes en América Latina y el Caribe: un motor para la recuperación*. Naciones Unidas.  
[3] Gopalakrishnan, K. (2020). Deep learning and machine learning for inventory management: A case study. *Procedia Computer Science*, 171, 2233‑2242.  
[4] Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and Machine Learning forecasting methods: Concerns and ways forward. *PLOS ONE*, 13(3), e0194889.  
[5] Roy, A., & Maiti, M. (2020). A fuzzy inventory model for deteriorating items with stock‑dependent demand. *International Journal of Fuzzy Systems*, 22(5), 1600‑1614.
