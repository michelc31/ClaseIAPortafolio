# Clasificador de Spam - Regresión Lineal

Sistema de clasificación de spam usando Regresión Logística con scikit-learn.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python spam_classifier.py
```

El programa:
1. Entrena un modelo de Regresión Lineal
2. Muestra métricas de rendimiento
3. Genera gráficas de resultados
4. Guarda el modelo entrenado
5. Pregunta si quieres abrir la interfaz gráfica

## Interfaz Gráfica

- Ingresa mensajes para clasificar
- Botón "Entrenar" para entrenar nuevo modelo
- Ejemplos predefinidos para probar

## Archivos

- `spam.csv` - Dataset de mensajes
- `spam_classifier.py` - Código principal
- `requirements.txt` - Dependencias
- `modelo.pkl` - Modelo entrenado (se genera automáticamente)
- `resultados.png` - Gráficas de resultados

## Algoritmo

- **Regresión Logística** con scikit-learn
- **TF-IDF** para vectorización de texto
- **Preprocesamiento** básico de texto
- **Métricas**: Accuracy, Precision, Recall, F1-Score

## Resultados Típicos

- Accuracy: ~95%
- Precision: ~90%
- Recall: ~85%
- F1-Score: ~87%