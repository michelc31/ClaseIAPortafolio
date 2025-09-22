#!/usr/bin/env python3

print("Iniciando análisis...")

try:
    import pandas as pd
    print("✓ Pandas importado")
    
    import numpy as np
    print("✓ Numpy importado")
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    print("✓ Sklearn importado")
    
    import re
    print("✓ Re importado")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("✓ Matplotlib y Seaborn importados")
    
    print("\nTodas las librerías importadas correctamente")
    
    # Cargar datos
    print("\nCargando datos...")
    df = pd.read_csv('spam.csv', encoding='latin-1')
    print(f"✓ Datos cargados: {len(df)} filas")
    
    # Preparar datos
    df = df[['v1', 'v2']].copy()
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    print(f"✓ Datos preparados: {df['label'].sum()} spam, {len(df) - df['label'].sum()} ham")
    
    # Limpiar texto
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    df['clean_message'] = df['message'].apply(clean_text)
    print("✓ Texto limpiado")
    
    # División de datos
    X_temp, X_test, y_temp, y_test = train_test_split(
        df['clean_message'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"✓ División: Train={len(X_train)}, Test={len(X_test)}, Val={len(X_val)}")
    
    # Vectorización
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    X_val_vec = vectorizer.transform(X_val)
    print(f"✓ Vectorización completada: {X_train_vec.shape[1]} features")
    
    # Entrenar modelo
    print("\nEntrenando modelo...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_vec, y_train)
    print("✓ Modelo entrenado")
    
    # Evaluar
    y_val_pred = model.predict(X_val_vec)
    y_val_proba = model.predict_proba(X_val_vec)[:, 1]
    
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_proba)
    
    print(f"\nRESULTADOS BASELINE:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Análisis de bias
    print("\nANÁLISIS DE BIAS:")
    cm = confusion_matrix(y_val, y_val_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    spam_probs = y_val_proba[y_val == 1]
    ham_probs = y_val_proba[y_val == 0]
    
    spam_mean_prob = np.mean(spam_probs)
    ham_mean_prob = np.mean(ham_probs)
    bias_score = abs(spam_mean_prob - 0.5) + abs(ham_mean_prob - 0.5)
    
    if bias_score < 0.1:
        bias_level = "BAJO"
    elif bias_score < 0.3:
        bias_level = "MEDIO"
    else:
        bias_level = "ALTO"
    
    print(f"Tasa Falsos Positivos: {fpr:.4f}")
    print(f"Tasa Falsos Negativos: {fnr:.4f}")
    print(f"Probabilidad promedio Spam: {spam_mean_prob:.4f}")
    print(f"Probabilidad promedio Ham: {ham_mean_prob:.4f}")
    print(f"Score de Bias: {bias_score:.4f}")
    print(f"Nivel de Bias: {bias_level}")
    
    # Análisis de varianza
    print("\nANÁLISIS DE VARIANZA:")
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='accuracy')
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    variance_score = std_score / mean_score
    
    if variance_score < 0.05:
        variance_level = "BAJO"
    elif variance_score < 0.15:
        variance_level = "MEDIO"
    else:
        variance_level = "ALTO"
    
    print(f"Scores CV: {cv_scores}")
    print(f"Media: {mean_score:.4f}")
    print(f"Desviación estándar: {std_score:.4f}")
    print(f"Coeficiente de variación: {variance_score:.4f}")
    print(f"Nivel de Varianza: {variance_level}")
    
    # Análisis de ajuste
    print("\nANÁLISIS DE AJUSTE:")
    train_score = model.score(X_train_vec, y_train)
    val_score = accuracy
    score_diff = train_score - val_score
    
    print(f"Score entrenamiento: {train_score:.4f}")
    print(f"Score validación: {val_score:.4f}")
    print(f"Diferencia: {score_diff:.4f}")
    
    if score_diff < 0.05 and val_score > 0.8:
        fit_level = "FIT (Bien ajustado)"
    elif score_diff > 0.15 or val_score < 0.7:
        fit_level = "UNDERFIT (Subajustado)"
    else:
        fit_level = "OVERFIT (Sobreajustado)"
    
    print(f"Nivel de Ajuste: {fit_level}")
    
    # Regularización
    print("\nAPLICANDO REGULARIZACIÓN:")
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_score = 0
    best_C = 1.0
    
    for C in C_values:
        model_reg = LogisticRegression(C=C, random_state=42, max_iter=1000)
        model_reg.fit(X_train_vec, y_train)
        val_score = model_reg.score(X_val_vec, y_val)
        
        print(f"C={C:.3f}: Accuracy={val_score:.4f}")
        
        if val_score > best_score:
            best_score = val_score
            best_C = C
    
    # Modelo final
    model_final = LogisticRegression(C=best_C, random_state=42, max_iter=1000)
    model_final.fit(X_train_vec, y_train)
    
    y_val_pred_final = model_final.predict(X_val_vec)
    y_val_proba_final = model_final.predict_proba(X_val_vec)[:, 1]
    
    final_accuracy = accuracy_score(y_val, y_val_pred_final)
    final_f1 = f1_score(y_val, y_val_pred_final)
    final_precision = precision_score(y_val, y_val_pred_final)
    final_recall = recall_score(y_val, y_val_pred_final)
    final_roc_auc = roc_auc_score(y_val, y_val_proba_final)
    
    print(f"\nMejor C: {best_C}")
    print(f"Accuracy final: {final_accuracy:.4f}")
    print(f"Precision final: {final_precision:.4f}")
    print(f"Recall final: {final_recall:.4f}")
    print(f"F1-Score final: {final_f1:.4f}")
    print(f"ROC-AUC final: {final_roc_auc:.4f}")
    
    print(f"\nMEJORAS OBTENIDAS:")
    print(f"Mejora en Accuracy: {final_accuracy - accuracy:+.4f}")
    print(f"Mejora en Precision: {final_precision - precision:+.4f}")
    print(f"Mejora en Recall: {final_recall - recall:+.4f}")
    print(f"Mejora en F1: {final_f1 - f1:+.4f}")
    print(f"Mejora en ROC-AUC: {final_roc_auc - roc_auc:+.4f}")
    
    print("\n" + "="*80)
    print("REPORTE FINAL")
    print("="*80)
    print(f"1. DIVISIÓN DE DATOS: Train={len(X_train)}, Test={len(X_test)}, Val={len(X_val)}")
    print(f"2. BIAS: {bias_level} (Score: {bias_score:.4f})")
    print(f"3. VARIANZA: {variance_level} (CV: {variance_score:.4f})")
    print(f"4. AJUSTE: {fit_level}")
    print(f"5. MEJOR C: {best_C}")
    print(f"6. MEJORA ACCURACY: {final_accuracy - accuracy:+.4f}")
    print(f"7. MEJORA F1: {final_f1 - f1:+.4f}")
    print("="*80)
    
    print("\n✓ Análisis completado exitosamente")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
