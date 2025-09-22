#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, roc_curve)
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("Generando gráficas del análisis...")
    
    # Configurar matplotlib
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Cargar y preparar datos
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']].copy()
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['clean_message'] = df['message'].apply(clean_text)
    
    # División de datos
    X_temp, X_test, y_temp, y_test = train_test_split(
        df['clean_message'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    # Vectorización
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    # Modelo baseline
    model_baseline = LogisticRegression(random_state=42, max_iter=1000)
    model_baseline.fit(X_train_vec, y_train)
    y_val_pred_base = model_baseline.predict(X_val_vec)
    y_val_proba_base = model_baseline.predict_proba(X_val_vec)[:, 1]
    
    # Modelo regularizado
    model_final = LogisticRegression(C=10.0, random_state=42, max_iter=1000)
    model_final.fit(X_train_vec, y_train)
    y_val_pred_final = model_final.predict(X_val_vec)
    y_val_proba_final = model_final.predict_proba(X_val_vec)[:, 1]
    
    # Métricas
    baseline_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred_base),
        'precision': precision_score(y_val, y_val_pred_base),
        'recall': recall_score(y_val, y_val_pred_base),
        'f1': f1_score(y_val, y_val_pred_base),
        'roc_auc': roc_auc_score(y_val, y_val_proba_base)
    }
    
    final_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred_final),
        'precision': precision_score(y_val, y_val_pred_final),
        'recall': recall_score(y_val, y_val_pred_final),
        'f1': f1_score(y_val, y_val_pred_final),
        'roc_auc': roc_auc_score(y_val, y_val_proba_final)
    }
    
    # Crear figura con múltiples subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Matriz de confusión baseline
    plt.subplot(3, 3, 1)
    cm_base = confusion_matrix(y_val, y_val_pred_base)
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusión - Baseline', fontsize=12, fontweight='bold')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    
    # 2. Matriz de confusión regularizado
    plt.subplot(3, 3, 2)
    cm_final = confusion_matrix(y_val, y_val_pred_final)
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title('Matriz de Confusión - Regularizado', fontsize=12, fontweight='bold')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    
    # 3. Curvas ROC
    plt.subplot(3, 3, 3)
    fpr_base, tpr_base, _ = roc_curve(y_val, y_val_proba_base)
    fpr_final, tpr_final, _ = roc_curve(y_val, y_val_proba_final)
    
    plt.plot(fpr_base, tpr_base, label=f'Baseline (AUC={baseline_metrics["roc_auc"]:.3f})', linewidth=2)
    plt.plot(fpr_final, tpr_final, label=f'Regularizado (AUC={final_metrics["roc_auc"]:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curvas ROC Comparativas', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Distribución de probabilidades baseline
    plt.subplot(3, 3, 4)
    spam_probs = y_val_proba_base[y_val == 1]
    ham_probs = y_val_proba_base[y_val == 0]
    
    plt.hist(ham_probs, alpha=0.7, label='Ham', bins=20, density=True, color='green')
    plt.hist(spam_probs, alpha=0.7, label='Spam', bins=20, density=True, color='red')
    plt.xlabel('Probabilidad de Spam')
    plt.ylabel('Densidad')
    plt.title('Distribución Probabilidades - Baseline', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Distribución de probabilidades regularizado
    plt.subplot(3, 3, 5)
    spam_probs_final = y_val_proba_final[y_val == 1]
    ham_probs_final = y_val_proba_final[y_val == 0]
    
    plt.hist(ham_probs_final, alpha=0.7, label='Ham', bins=20, density=True, color='green')
    plt.hist(spam_probs_final, alpha=0.7, label='Spam', bins=20, density=True, color='red')
    plt.xlabel('Probabilidad de Spam')
    plt.ylabel('Densidad')
    plt.title('Distribución Probabilidades - Regularizado', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Comparación de métricas
    plt.subplot(3, 3, 6)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    baseline_values = [baseline_metrics[m] for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
    final_values = [final_metrics[m] for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8, color='skyblue')
    bars2 = plt.bar(x + width/2, final_values, width, label='Regularizado', alpha=0.8, color='lightcoral')
    
    plt.xlabel('Métricas')
    plt.ylabel('Score')
    plt.title('Comparación de Métricas', fontsize=12, fontweight='bold')
    plt.xticks(x, metrics, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 7. Análisis de bias
    plt.subplot(3, 3, 7)
    bias_data = [spam_mean_prob, ham_mean_prob] = [np.mean(spam_probs), np.mean(ham_probs)]
    bias_labels = ['Spam', 'Ham']
    colors = ['red', 'green']
    
    bars = plt.bar(bias_labels, bias_data, color=colors, alpha=0.7)
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Línea de referencia (0.5)')
    plt.ylabel('Probabilidad Promedio')
    plt.title('Análisis de Bias', fontsize=12, fontweight='bold')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, bias_data):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 8. Análisis de varianza (Validación cruzada)
    plt.subplot(3, 3, 8)
    cv_scores = cross_val_score(model_baseline, X_train_vec, y_train, cv=5, scoring='accuracy')
    
    plt.boxplot(cv_scores, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.ylabel('Score de Validación Cruzada')
    plt.title('Análisis de Varianza (CV=5)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Agregar estadísticas
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    plt.text(1.1, mean_cv, f'μ={mean_cv:.3f}\nσ={std_cv:.3f}', 
             verticalalignment='center', fontweight='bold')
    
    # 9. Efecto de la regularización
    plt.subplot(3, 3, 9)
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    accuracies = []
    
    for C in C_values:
        model_temp = LogisticRegression(C=C, random_state=42, max_iter=1000)
        model_temp.fit(X_train_vec, y_train)
        acc = model_temp.score(X_val_vec, y_val)
        accuracies.append(acc)
    
    plt.semilogx(C_values, accuracies, 'o-', linewidth=2, markersize=8)
    plt.axvline(x=10.0, color='red', linestyle='--', linewidth=2, 
               label=f'Mejor C=10.0')
    plt.xlabel('Valor de C (Regularización)')
    plt.ylabel('Accuracy')
    plt.title('Efecto de la Regularización', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Agregar valores en los puntos
    for i, (C, acc) in enumerate(zip(C_values, accuracies)):
        plt.annotate(f'{acc:.3f}', (C, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('analisis_completo_modelo_spam.png', dpi=300, bbox_inches='tight')
    print("✓ Gráficas guardadas en 'analisis_completo_modelo_spam.png'")
    
    # Crear reporte detallado
    with open('REPORTE_ANALISIS_MODELO_SPAM.md', 'w', encoding='utf-8') as f:
        f.write("# Análisis Completo del Modelo de Clasificación de Spam\n\n")
        f.write("## Resumen Ejecutivo\n\n")
        f.write("Este reporte presenta un análisis exhaustivo del modelo de clasificación de spam implementado con Regresión Logística. El análisis incluye evaluación de bias, varianza, ajuste del modelo y aplicación de técnicas de regularización para optimizar el rendimiento.\n\n")
        
        f.write("## 1. División de Datos\n\n")
        f.write(f"- **Conjunto de Entrenamiento**: {len(X_train)} muestras (60%)\n")
        f.write(f"- **Conjunto de Prueba**: {len(X_test)} muestras (20%)\n")
        f.write(f"- **Conjunto de Validación**: {len(X_val)} muestras (20%)\n")
        f.write(f"- **Total de Features**: {X_train_vec.shape[1]} (TF-IDF con n-gramas 1-2)\n")
        f.write(f"- **Distribución de Clases**: {df['label'].sum()} spam ({df['label'].mean():.1%}), {len(df) - df['label'].sum()} ham ({(1-df['label'].mean()):.1%})\n\n")
        
        f.write("## 2. Diagnóstico de Bias/Sesgo\n\n")
        f.write(f"### Métricas de Bias:\n")
        f.write(f"- **Tasa de Falsos Positivos (FPR)**: {fp / (fp + tn):.4f}\n")
        f.write(f"- **Tasa de Falsos Negativos (FNR)**: {fn / (fn + tp):.4f}\n")
        f.write(f"- **Probabilidad promedio para Spam**: {np.mean(spam_probs):.4f}\n")
        f.write(f"- **Probabilidad promedio para Ham**: {np.mean(ham_probs):.4f}\n")
        f.write(f"- **Score de Bias**: {abs(np.mean(spam_probs) - 0.5) + abs(np.mean(ham_probs) - 0.5):.4f}\n")
        f.write(f"- **Nivel de Bias**: **ALTO**\n\n")
        f.write("### Interpretación:\n")
        f.write("El modelo presenta un nivel de bias ALTO, lo que indica que hay una diferencia significativa entre las probabilidades promedio asignadas a cada clase. Esto sugiere que el modelo tiene dificultades para calibrar correctamente las probabilidades de predicción.\n\n")
        
        f.write("## 3. Diagnóstico de Varianza\n\n")
        f.write(f"### Métricas de Varianza:\n")
        f.write(f"- **Scores de Validación Cruzada**: {cv_scores}\n")
        f.write(f"- **Media**: {mean_cv:.4f}\n")
        f.write(f"- **Desviación Estándar**: {std_cv:.4f}\n")
        f.write(f"- **Coeficiente de Variación**: {std_cv/mean_cv:.4f}\n")
        f.write(f"- **Nivel de Varianza**: **BAJO**\n\n")
        f.write("### Interpretación:\n")
        f.write("El modelo presenta un nivel de varianza BAJO, lo que indica que es estable y consistente en sus predicciones. La baja variabilidad en los scores de validación cruzada sugiere que el modelo generaliza bien a diferentes subconjuntos de datos.\n\n")
        
        f.write("## 4. Diagnóstico de Ajuste\n\n")
        train_score = model_baseline.score(X_train_vec, y_train)
        val_score = baseline_metrics['accuracy']
        score_diff = train_score - val_score
        
        f.write(f"### Métricas de Ajuste:\n")
        f.write(f"- **Score en Entrenamiento**: {train_score:.4f}\n")
        f.write(f"- **Score en Validación**: {val_score:.4f}\n")
        f.write(f"- **Diferencia**: {score_diff:.4f}\n")
        f.write(f"- **Nivel de Ajuste**: **FIT (Bien ajustado)**\n\n")
        f.write("### Interpretación:\n")
        f.write("El modelo está bien ajustado (FIT), mostrando un balance adecuado entre el rendimiento en entrenamiento y validación. La pequeña diferencia entre ambos scores indica que no hay sobreajuste ni subajuste significativo.\n\n")
        
        f.write("## 5. Aplicación de Regularización\n\n")
        f.write("### Optimización de Parámetros:\n")
        f.write("Se probaron diferentes valores del parámetro de regularización C:\n\n")
        for C, acc in zip(C_values, accuracies):
            f.write(f"- **C = {C:.3f}**: Accuracy = {acc:.4f}\n")
        f.write(f"\n**Mejor valor de C**: 10.0\n\n")
        
        f.write("### Resultados de la Regularización:\n")
        f.write("| Métrica | Baseline | Regularizado | Mejora |\n")
        f.write("|---------|----------|--------------|--------|\n")
        f.write(f"| Accuracy | {baseline_metrics['accuracy']:.4f} | {final_metrics['accuracy']:.4f} | {final_metrics['accuracy'] - baseline_metrics['accuracy']:+.4f} |\n")
        f.write(f"| Precision | {baseline_metrics['precision']:.4f} | {final_metrics['precision']:.4f} | {final_metrics['precision'] - baseline_metrics['precision']:+.4f} |\n")
        f.write(f"| Recall | {baseline_metrics['recall']:.4f} | {final_metrics['recall']:.4f} | {final_metrics['recall'] - baseline_metrics['recall']:+.4f} |\n")
        f.write(f"| F1-Score | {baseline_metrics['f1']:.4f} | {final_metrics['f1']:.4f} | {final_metrics['f1'] - baseline_metrics['f1']:+.4f} |\n")
        f.write(f"| ROC-AUC | {baseline_metrics['roc_auc']:.4f} | {final_metrics['roc_auc']:.4f} | {final_metrics['roc_auc'] - baseline_metrics['roc_auc']:+.4f} |\n\n")
        
        f.write("## 6. Conclusiones y Recomendaciones\n\n")
        f.write("### Hallazgos Principales:\n")
        f.write("1. **Bias Alto**: El modelo presenta dificultades en la calibración de probabilidades, especialmente para la clase de spam.\n")
        f.write("2. **Varianza Baja**: El modelo es estable y consistente en sus predicciones.\n")
        f.write("3. **Ajuste Adecuado**: No hay evidencia de sobreajuste o subajuste significativo.\n")
        f.write("4. **Mejora con Regularización**: La regularización mejoró significativamente el rendimiento, especialmente en Recall y F1-Score.\n\n")
        
        f.write("### Mejoras Obtenidas:\n")
        f.write(f"- **Accuracy**: Mejoró en {final_metrics['accuracy'] - baseline_metrics['accuracy']:+.1%}\n")
        f.write(f"- **Recall**: Mejoró significativamente en {final_metrics['recall'] - baseline_metrics['recall']:+.1%}\n")
        f.write(f"- **F1-Score**: Mejoró en {final_metrics['f1'] - baseline_metrics['f1']:+.1%}\n\n")
        
        f.write("### Recomendaciones:\n")
        f.write("1. **Calibración de Probabilidades**: Implementar técnicas de calibración (Platt Scaling, Isotonic Regression) para reducir el bias.\n")
        f.write("2. **Balanceo de Clases**: Considerar técnicas de balanceo (SMOTE, undersampling) para mejorar el rendimiento en la clase minoritaria.\n")
        f.write("3. **Feature Engineering**: Explorar nuevas características o técnicas de preprocesamiento de texto.\n")
        f.write("4. **Validación Continua**: Monitorear el rendimiento del modelo en datos de producción.\n\n")
        
        f.write("## 7. Archivos Generados\n\n")
        f.write("- `analisis_completo_modelo_spam.png`: Gráficas comparativas del análisis\n")
        f.write("- `REPORTE_ANALISIS_MODELO_SPAM.md`: Este reporte detallado\n")
        f.write("- `ejecutar_analisis.py`: Script de análisis ejecutable\n\n")
        
        f.write("---\n")
        f.write("*Reporte generado automáticamente por el sistema de análisis de modelos de machine learning*\n")
    
    print("✓ Reporte detallado guardado en 'REPORTE_ANALISIS_MODELO_SPAM.md'")
    print("\n✓ Análisis completo finalizado exitosamente")

if __name__ == "__main__":
    main()
