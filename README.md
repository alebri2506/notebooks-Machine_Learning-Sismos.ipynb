# Análisis exploratorio sobre predicciones de Magnitud Sísmica en el Perú mediante Machine Learning

Modelo de Machine Learning basado en una Red Neuronal Artificial (Perceptrón Multicapa - MLP) para estimar la magnitud de eventos sísmicos en territorio peruano a partir de su ubicación geográfica, ventana temporal y clasificación de profundidad geológica.

---

## 1. Objetivo del Proyecto

El objetivo es responder a la interrogante:  
*¿Qué magnitud sísmica podría presentarse en una ubicación determinada del Perú, considerando el registro histórico de sismos en coordenadas y profundidades similares?*

A través de técnicas de regresión con Deep Learning, el sistema modela los patrones espacio-temporales y de subducción tectónica para predecir la variable continua de magnitud sísmica.

---

## 2. Dataset

Los datos provienen del **Catálogo Sísmico 1960 - 2023** elaborado por el **Instituto Geofísico del Perú (IGP)** y publicado en la plataforma de Datos Abiertos del Gobierno del Perú.

* **Volumen original:** 23,680 registros sísmicos registrados entre 1960 y 2023[cite: 4].
* **Atributos originales:** `ID`, `FECHA_UTC`, `HORA_UTC`, `LATITUD`, `LONGITUD`, `PROFUNDIDAD`, `MAGNITUD` y `FECHA_CORTE`.
* **Zona geográfica de estudio:** Coordenadas delimitadas al territorio peruano (Latitud: -18.0° a -3.0°; Longitud: -81.0° a -68.0°).

---

## 3. Preprocesamiento e Ingeniería de Características

1. **Depuración de Atributos:** Se eliminaron identificadores y metadatos administrativos (`ID`, `FECHA_CORTE`, `HORA_UTC`) que no aportan señal geológica.
2. **Transformación Temporal:** A partir de `FECHA_UTC` se derivaron las variables numéricas de `AÑO` y `MES`.
3. **Clasificación de Profundidad (`TIPO_PROFUNDIDAD`):** La profundidad focal en kilómetros se categorizó en tres niveles sismológicos estándar:
   * **Superficial:** Profundidad menor o igual a 60 km.
   * **Intermedio:** Profundidad entre 61 km y 300 km.
   * **Profundo:** Profundidad mayor a 300 km.
4. **Control de Calidad y Filtrado de Outliers:** 
   * Se acotaron los valores válidos de magnitud (rango de 0 a 9) y temporales (meses de 1 a 12, años válidos hasta 2025).
   * Tratamiento de valores faltantes mediante imputación de moda en variables categóricas y mediana en magnitud.
5. **Codificación y Escalado:**
   * Aplicación de *One-Hot Encoding* sobre la variable categórica `TIPO_PROFUNDIDAD`.
   * Estandarización de todas las variables predictoras (`LATITUD`, `LONGITUD`, `AÑO`, `MES` y columnas dummificadas) mediante `StandardScaler`.

---

## 4. Metodología y Arquitectura del Modelo

El problema se formuló como una **tarea de regresión supervisada** implementada en TensorFlow / Keras[cite: 4]:

* **Partición de Datos:** 80% para entrenamiento y 20% para evaluación y prueba (`test_size=0.2`, `random_state=42`).
* **Estructura de la Red Neuronal (Sequential):**
  * **Capa de Entrada y Oculta 1:** Capa densa de 64 neuronas con activación ReLU.
  * **Capa Oculta 2:** Capa densa de 32 neuronas con activación ReLU.
  * **Capa de Salida:** 1 neurona con activación lineal para la estimación directa de la magnitud.
* **Compilación:**
  * **Optimizador:** Adam.
  * **Función de Pérdida:** Error Cuadrático Medio (`MSE` / `loss='mse'`).
  * **Métrica de Desempeño:** Error Absoluto Medio (`MAE`).
* **Hiperparámetros de Entrenamiento:** 300 épocas de iteración y tamaño de lote (*batch size*) de 32.

---

## 5. Resultados

El modelo fue evaluado sobre el 20% de datos de prueba no vistos durante el entrenamiento, obteniendo los siguientes resultados:

| Métrica | Valor Obtenido |
| :--- | :--- |
| **Error Absoluto Medio (MAE)** | **0.25** |
| **Pérdida en Test (MSE)** | **0.1311**|

### Interpretación de Métricas:
* Un **MAE de 0.25** indica que, en promedio, las predicciones del modelo se desvían únicamente ±0.25 grados de magnitud respecto al valor real medido por el IGP.
* Las curvas de aprendizaje de MSE y MAE a lo largo de las 300 épocas mostraron convergencia estable entre el conjunto de entrenamiento y el de validación, sin evidencia de sobreajuste severo.

---

## 6. Módulo de Inferencia y Predicción

El proyecto incluye la función `predecir_magnitud`, la cual recibe coordenadas y parámetros de un evento y replica la canalización completa de preprocesamiento (One-Hot Encoding y normalización) antes de ejecutar el modelo:

```python
# Ejemplo de uso del modelo entrenado
magnitud_estimada = predecir_magnitud(
    latitud = -20.1,
    longitud = -77.0,
    año = 2024,
    mes = 7,
    profundidad = 'Intermedio',
    columnas_esperadas = columnas_esperadas,
    scaler = scaler,
    model = model
)

# Salida estimada obtenida: 6.27
