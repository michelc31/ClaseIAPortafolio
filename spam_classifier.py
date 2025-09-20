#!/usr/bin/env python3


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import re
import pickle
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import warnings
warnings.filterwarnings('ignore')

class SpamClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.accuracy = 0
        
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def load_data(self, filepath):
        print("Cargando datos...")
        df = pd.read_csv(filepath, encoding='latin-1')
        df = df[['v1', 'v2']].copy()
        df.columns = ['label', 'message']
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        
        print(f"Total mensajes: {len(df)}")
        print(f"Spam: {df['label'].sum()}, Ham: {len(df) - df['label'].sum()}")
        return df
    
    def prepare_data(self, df):
        print("Preparando datos...")
        df['clean_message'] = df['message'].apply(self.clean_text)
        
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(df['clean_message'])
        y = df['label']
        
        print(f"Features: {X.shape[1]}")
        return X, y
    
    def train(self, X, y):
        print("Entrenando modelo...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {self.accuracy:.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1: {f1_score(y_test, y_pred):.4f}")
        
        return X_test, y_test, y_pred
    
    def plot_results(self, X_test, y_test, y_pred):
        print("Generando graficas...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Matriz de confusion
        cm = confusion_matrix(y_test, y_pred)
        axes[0,0].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[0,0].set_title('Matriz de Confusion')
        axes[0,0].set_xlabel('Prediccion')
        axes[0,0].set_ylabel('Real')
        
        # ROC
        y_proba = self.model.predict_proba(X_test)[:, 1]
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        axes[0,1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
        axes[0,1].plot([0, 1], [0, 1], 'k--')
        axes[0,1].set_title('Curva ROC')
        axes[0,1].legend()
        
        # Distribucion de probabilidades
        spam_probs = y_proba[y_test == 1]
        ham_probs = y_proba[y_test == 0]
        axes[1,0].hist(ham_probs, alpha=0.7, label='Ham', bins=20)
        axes[1,0].hist(spam_probs, alpha=0.7, label='Spam', bins=20)
        axes[1,0].set_title('Distribucion de Probabilidades')
        axes[1,0].legend()
        
        # Metricas
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        values = [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), 
                 recall_score(y_test, y_pred), f1_score(y_test, y_pred)]
        axes[1,1].bar(metrics, values)
        axes[1,1].set_title('Metricas')
        
        plt.tight_layout()
        plt.savefig('resultados.png')
        plt.show()
    
    def predict(self, message):
        if self.model is None:
            return None
        
        clean_msg = self.clean_text(message)
        msg_vector = self.vectorizer.transform([clean_msg])
        pred = self.model.predict(msg_vector)[0]
        prob = self.model.predict_proba(msg_vector)[0]
        
        return {
            'prediction': 'Spam' if pred == 1 else 'Ham',
            'confidence': max(prob),
            'spam_prob': prob[1],
            'ham_prob': prob[0]
        }
    
    def save_model(self, filename='modelo.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump({'model': self.model, 'vectorizer': self.vectorizer, 'accuracy': self.accuracy}, f)
        print(f"Modelo guardado en {filename}")
    
    def load_model(self, filename='modelo.pkl'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.vectorizer = data['vectorizer']
        self.accuracy = data['accuracy']
        print(f"Modelo cargado desde {filename}")

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Spam")
        self.root.geometry("600x500")
        
        self.classifier = SpamClassifier()
        self.model_loaded = False
        
        self.create_widgets()
        self.try_load_model()
    
    def create_widgets(self):
        # Titulo
        title = tk.Label(self.root, text="Clasificador de Spam", font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Area de texto
        tk.Label(self.root, text="Mensaje:").pack(anchor='w', padx=20)
        self.text_area = scrolledtext.ScrolledText(self.root, height=6, width=70)
        self.text_area.pack(padx=20, pady=5)
        
        # Botones
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        self.classify_btn = tk.Button(btn_frame, text="Clasificar", command=self.classify, state='disabled')
        self.classify_btn.pack(side='left', padx=5)
        
        tk.Button(btn_frame, text="Limpiar", command=self.clear).pack(side='left', padx=5)
        tk.Button(btn_frame, text="Entrenar", command=self.train).pack(side='left', padx=5)
        
        # Resultado
        self.result_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)
        
        # Info del modelo
        self.model_info = tk.Label(self.root, text="Modelo no cargado", fg="red")
        self.model_info.pack()
        
        # Ejemplos
        examples_frame = tk.Frame(self.root)
        examples_frame.pack(pady=10)
        
        examples = [
            ("Spam", "Congratulations! You've won $1000!"),
            ("Ham", "Hey, how are you doing?"),
            ("Spam", "URGENT: Your account will be closed!")
        ]
        
        for label, msg in examples:
            tk.Button(examples_frame, text=label, 
                     command=lambda m=msg: self.text_area.insert('1.0', m)).pack(side='left', padx=2)
    
    def try_load_model(self):
        try:
            self.classifier.load_model()
            self.model_loaded = True
            self.classify_btn.config(state='normal')
            self.model_info.config(text=f"Modelo cargado - Accuracy: {self.classifier.accuracy:.3f}", fg="green")
        except:
            pass
    
    def classify(self):
        if not self.model_loaded:
            messagebox.showerror("Error", "No hay modelo cargado")
            return
        
        message = self.text_area.get('1.0', tk.END).strip()
        if not message:
            messagebox.showwarning("Advertencia", "Ingresa un mensaje")
            return
        
        result = self.classifier.predict(message)
        if result:
            color = "red" if result['prediction'] == 'Spam' else "green"
            self.result_label.config(text=f"Prediccion: {result['prediction']} (Confianza: {result['confidence']:.2%})", fg=color)
    
    def clear(self):
        self.text_area.delete('1.0', tk.END)
        self.result_label.config(text="")
    
    def train(self):
        try:
            messagebox.showinfo("Info", "Entrenando modelo... esto puede tomar un momento")
            df = self.classifier.load_data('spam.csv')
            X, y = self.classifier.prepare_data(df)
            self.classifier.train(X, y)
            self.classifier.save_model()
            self.model_loaded = True
            self.classify_btn.config(state='normal')
            self.model_info.config(text=f"Modelo entrenado - Accuracy: {self.classifier.accuracy:.3f}", fg="green")
            messagebox.showinfo("Exito", "Modelo entrenado correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")

def main():
    print("="*50)
    print("CLASIFICADOR DE SPAM - REGRESION LINEAL")
    print("="*50)
    
    classifier = SpamClassifier()
    
    # Cargar y entrenar
    df = classifier.load_data('spam.csv')
    X, y = classifier.prepare_data(df)
    X_test, y_test, y_pred = classifier.train(X, y)
    
    # Graficas
    classifier.plot_results(X_test, y_test, y_pred)
    
    # Guardar modelo
    classifier.save_model()
    
    # Ejemplos
    print("\nEjemplos:")
    test_msgs = [
        "Congratulations! You've won $1000!",
        "Hey, how are you doing?",
        "URGENT: Your account will be closed!"
    ]
    
    for msg in test_msgs:
        result = classifier.predict(msg)
        print(f"'{msg}' -> {result['prediction']} ({result['confidence']:.2%})")
    
    print("\n¿Abrir interfaz grafica? (s/n): ", end="")
    if input().lower().startswith('s'):
        root = tk.Tk()
        app = GUI(root)
        root.mainloop()

if __name__ == "__main__":
    main()