# Análisis Completo del Modelo de Clasificación de Spam

## Resumen 

Este reporte presenta un análisis del modelo de clasificación de spam implementado con Regresión Logística. El análisis incluye evaluación de bias, varianza, ajuste del modelo y aplicación de técnicas de regularización para optimizar el rendimiento.

## 1. División de Datos

- **Conjunto de Entrenamiento**: 3342 muestras (60%)
- **Conjunto de Prueba**: 1115 muestras (20%)
- **Conjunto de Validación**: 1115 muestras (20%)
- **Total de Features**: 5000 
- **Distribución de Clases**: 747 spam (13.4%), 4825 ham (86.6%)

## 2. Diagnóstico de Bias/Sesgo

### Métricas de Bias:
- **Tasa de Falsos Positivos (FPR)**: 0.0000
- **Tasa de Falsos Negativos (FNR)**: 0.3733
- **Probabilidad promedio para Spam**: 0.5288
- **Probabilidad promedio para Ham**: 0.0684
- **Score de Bias**: 0.4604
- **Nivel de Bias**: **ALTO**

### Interpretación:
El modelo presenta un nivel de bias ALTO, lo que indica que hay una diferencia significativa entre las probabilidades promedio asignadas a cada clase. Esto sugiere que el modelo tiene dificultades para calibrar correctamente las probabilidades de predicción, especialmente para la clase de spam.

## 3. Diagnóstico de Varianza

### Métricas de Varianza:
- **Scores de Validación Cruzada**: [0.9327, 0.9163, 0.9237, 0.9207, 0.9281]
- **Media**: 0.9243
- **Desviación Estándar**: 0.0057
- **Coeficiente de Variación**: 0.0062
- **Nivel de Varianza**: **BAJO**

### Interpretación:
El modelo presenta un nivel de varianza BAJO, lo que indica que es estable y consistente en sus predicciones. La baja variabilidad en los scores de validación cruzada sugiere que el modelo generaliza bien a diferentes subconjuntos de datos.

## 4. Diagnóstico de Ajuste

### Métricas de Ajuste:
- **Score en Entrenamiento**: 0.9611
- **Score en Validación**: 0.9498
- **Diferencia**: 0.0113
- **Nivel de Ajuste**: **FIT (Bien ajustado)**

### Interpretación:
El modelo está bien ajustado (FIT), mostrando un balance adecuado entre el rendimiento en entrenamiento y validación. La pequeña diferencia entre ambos scores indica que no hay sobreajuste ni subajuste significativo.

## 5. Aplicación de Regularización

### Optimización de Parámetros:
Se probaron diferentes valores del parámetro de regularización C:

- **C = 0.001**: Accuracy = 0.8655
- **C = 0.010**: Accuracy = 0.8655
- **C = 0.100**: Accuracy = 0.8655
- **C = 1.000**: Accuracy = 0.9498
- **C = 10.000**: Accuracy = 0.9776
- **C = 100.000**: Accuracy = 0.9740

**Mejor valor de C**: 10.0

### Resultados de la Regularización:

| Métrica | Baseline | Regularizado | Mejora |
|---------|----------|--------------|--------|
| Accuracy | 0.9498 | 0.9776 | +0.0278 |
| Precision | 1.0000 | 0.9845 | -0.0155 |
| Recall | 0.6267 | 0.8467 | +0.2200 |
| F1-Score | 0.7705 | 0.9104 | +0.1399 |
| ROC-AUC | 0.9861 | 0.9867 | +0.0006 |

## 6. Análisis de Mejoras

### Mejoras Significativas:
- **Accuracy**: Mejoró en +2.78%
- **Recall**: Mejoró significativamente en +22.00%
- **F1-Score**: Mejoró en +13.99%

### Interpretación de las Mejoras:
La regularización logró un balance más adecuado entre precision y recall, reduciendo el número de falsos negativos (spam no detectado) mientras mantenía una alta precisión. Esto es especialmente importante en aplicaciones de filtrado de spam donde es crucial detectar la mayor cantidad posible de mensajes spam.

## 7. Conclusiones y Recomendaciones

### Hallazgos Principales:
1. **Bias Alto**: El modelo presenta dificultades en la calibración de probabilidades, especialmente para la clase de spam.
2. **Varianza Baja**: El modelo es estable y consistente en sus predicciones.
3. **Ajuste Adecuado**: No hay evidencia de sobreajuste.
4. **Mejora con Regularización**: La regularización mejoró significativamente el rendimiento, especialmente en Recall y F1-Score.


## 9. Código de Análisis

El análisis completo se puede ejecutar usando el script `ejecutar_analisis.py` que incluye:

```python
# División Train/Test/Validation
X_temp, X_test, y_temp, y_test = train_test_split(
    df['clean_message'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# Análisis de bias
spam_probs = y_val_proba[y_val == 1]
ham_probs = y_val_proba[y_val == 0]
bias_score = abs(np.mean(spam_probs) - 0.5) + abs(np.mean(ham_probs) - 0.5)

# Análisis de varianza
cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='accuracy')
variance_score = np.std(cv_scores) / np.mean(cv_scores)

# Regularización
C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
best_C = 10.0  # Mejor valor encontrado
```

---
