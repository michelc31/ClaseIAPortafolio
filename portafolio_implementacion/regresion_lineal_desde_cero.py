#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import random

class RegresionLineal:
    def __init__(self, learning_rate=0.01, max_iter=1000, tolerance=1e-6, regularization=None, alpha=0.01):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.regularization = regularization
        self.alpha = alpha
        
        self.weights = None
        self.bias = 0
        self.cost_history = []
        self.is_fitted = False
        
    def _add_intercept(self, X):
        m = X.shape[0]
        intercept = np.ones((m, 1))
        return np.concatenate((intercept, X), axis=1)
    
    def _compute_cost(self, X, y, weights):
        m = len(y)
        predictions = np.dot(X, weights)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        
        if self.regularization == 'l1':
            cost += self.alpha * np.sum(np.abs(weights[1:]))
        elif self.regularization == 'l2':
            cost += self.alpha * np.sum(weights[1:] ** 2)
            
        return cost
    
    def _compute_gradients(self, X, y, weights):
        m = len(y)
        predictions = np.dot(X, weights)
        error = predictions - y
        
        gradients = (1 / m) * np.dot(X.T, error)
        
        if self.regularization == 'l1':
            gradients[1:] += self.alpha * np.sign(weights[1:])
        elif self.regularization == 'l2':
            gradients[1:] += 2 * self.alpha * weights[1:]
            
        return gradients
    
    def fit(self, X, y, verbose=False):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).flatten()
        
        if X.shape[0] != len(y):
            raise ValueError("X e y deben tener el mismo número de muestras")
        
        X_with_intercept = self._add_intercept(X)
        
        np.random.seed(42)
        self.weights = np.random.normal(0, 0.01, X_with_intercept.shape[1])
        
        initial_cost = self._compute_cost(X_with_intercept, y, self.weights)
        self.cost_history = [initial_cost]
        
        if verbose:
            print(f"Costo inicial: {initial_cost:.6f}")
            print("Iniciando entrenamiento...")
        
        for i in range(self.max_iter):
            gradients = self._compute_gradients(X_with_intercept, y, self.weights)
            self.weights -= self.learning_rate * gradients
            
            current_cost = self._compute_cost(X_with_intercept, y, self.weights)
            self.cost_history.append(current_cost)
            
            if len(self.cost_history) > 1:
                cost_change = abs(self.cost_history[-1] - self.cost_history[-2])
                if cost_change < self.tolerance:
                    if verbose:
                        print(f"Convergencia alcanzada en iteración {i+1}")
                    break
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Iteración {i+1}: Costo = {current_cost:.6f}")
        
        self.is_fitted = True
        
        if verbose:
            print(f"Entrenamiento completado en {len(self.cost_history)} iteraciones")
            print(f"Costo final: {self.cost_history[-1]:.6f}")
            print(f"Pesos finales: {self.weights}")
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        X = np.array(X, dtype=float)
        X_with_intercept = self._add_intercept(X)
        return np.dot(X_with_intercept, self.weights)
    
    def get_weights(self):
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")
        return self.weights[1:], self.weights[0]
    
    def get_cost_history(self):
        return self.cost_history

class ValidacionCruzada:
    def __init__(self, k_folds=5):
        self.k_folds = k_folds
    
    def _split_data(self, X, y):
        n_samples = len(X)
        indices = list(range(n_samples))
        random.shuffle(indices)
        
        fold_size = n_samples // self.k_folds
        folds = []
        
        for i in range(self.k_folds):
            start = i * fold_size
            end = start + fold_size if i < self.k_folds - 1 else n_samples
            fold_indices = indices[start:end]
            folds.append(fold_indices)
        
        return folds
    
    def cross_validate(self, model_class, X, y, **model_params):
        folds = self._split_data(X, y)
        scores = []
        
        for i, test_indices in enumerate(folds):
            train_indices = []
            for j, fold in enumerate(folds):
                if j != i:
                    train_indices.extend(fold)
            
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]
            
            model = model_class(**model_params)
            model.fit(X_train, y_train, verbose=False)
            
            y_pred = model.predict(X_test)
            mse = np.mean((y_test - y_pred) ** 2)
            scores.append(mse)
        
        return np.mean(scores), np.std(scores)

def generar_datos_sinteticos(n_samples=100, n_features=1, noise=10, random_state=42):
    np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    true_bias = np.random.randn()
    y = np.dot(X, true_weights) + true_bias + np.random.normal(0, noise, n_samples)
    
    return X, y, true_weights, true_bias

def calcular_metricas(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def visualizar_resultados(model, X, y, title="Regresión Lineal"):
    print(f"\nGenerando visualizaciones para: {title}")
    
    y_pred = model.predict(X)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].plot(model.cost_history, 'b-', linewidth=2)
    axes[0, 0].set_title('Curva de Costo durante el Entrenamiento')
    axes[0, 0].set_xlabel('Iteración')
    axes[0, 0].set_ylabel('Costo (MSE)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(y, y_pred, alpha=0.6, color='blue')
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Línea perfecta')
    axes[0, 1].set_xlabel('Valores Reales')
    axes[0, 1].set_ylabel('Predicciones')
    axes[0, 1].set_title('Predicciones vs Valores Reales')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    residuals = y - y_pred
    axes[1, 0].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Predicciones')
    axes[1, 0].set_ylabel('Residuos')
    axes[1, 0].set_title('Análisis de Residuos')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('Residuos')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].set_title('Distribución de Residuos')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_analisis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    metricas = calcular_metricas(y, y_pred)
    print(f"MSE: {metricas['MSE']:.4f}")
    print(f"RMSE: {metricas['RMSE']:.4f}")
    print(f"MAE: {metricas['MAE']:.4f}")
    print(f"R²: {metricas['R2']:.4f}")
    
    return metricas

def comparar_regularizaciones(X, y):
    print("\n" + "="*60)
    print("COMPARACIÓN DE REGULARIZACIONES")
    print("="*60)
    
    n_samples = len(X)
    n_train = int(0.8 * n_samples)
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    configs = [
        {"name": "Sin Regularización", "regularization": None, "alpha": 0},
        {"name": "Ridge (L2) - Baja", "regularization": "l2", "alpha": 0.01},
        {"name": "Ridge (L2) - Alta", "regularization": "l2", "alpha": 0.1},
        {"name": "Lasso (L1) - Baja", "regularization": "l1", "alpha": 0.01},
        {"name": "Lasso (L1) - Alta", "regularization": "l1", "alpha": 0.1}
    ]
    
    results = []
    
    for config in configs:
        print(f"\nProbando: {config['name']}")
        
        # Crear y entrenar modelo
        model = RegresionLineal(
            learning_rate=0.01,
            max_iter=1000,
            regularization=config['regularization'],
            alpha=config['alpha']
        )
        
        model.fit(X_train, y_train, verbose=False)
        
        # Hacer predicciones
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        metricas = calcular_metricas(y_test, y_pred)
        
        # Validación cruzada
        cv = ValidacionCruzada(k_folds=5)
        cv_mean, cv_std = cv.cross_validate(RegresionLineal, X_train, y_train, 
                                          learning_rate=0.01, max_iter=1000,
                                          regularization=config['regularization'], 
                                          alpha=config['alpha'])
        
        results.append({
            'Modelo': config['name'],
            'MSE': metricas['MSE'],
            'R²': metricas['R2'],
            'MAE': metricas['MAE'],
            'CV_MSE_Mean': cv_mean,
            'CV_MSE_Std': cv_std
        })
        
        print(f"  MSE: {metricas['MSE']:.4f}")
        print(f"  R²: {metricas['R2']:.4f}")
        print(f"  MAE: {metricas['MAE']:.4f}")
        print(f"  CV MSE: {cv_mean:.4f} ± {cv_std:.4f}")
    
    return results

def demostrar_gradiente_descendente():
    print("\n" + "="*60)
    print("DEMOSTRACIÓN DE GRADIENTE DESCENDENTE")
    print("="*60)
    
    X, y, true_weights, true_bias = generar_datos_sinteticos(n_samples=50, n_features=1, noise=5)
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    
    plt.figure(figsize=(15, 10))
    
    for i, lr in enumerate(learning_rates):
        plt.subplot(2, 2, i+1)
        
        # Entrenar modelo
        model = RegresionLineal(learning_rate=lr, max_iter=1000)
        model.fit(X, y, verbose=False)
        
        # Graficar curva de costo
        plt.plot(model.cost_history, 'b-', linewidth=2)
        plt.title(f'Tasa de Aprendizaje: {lr}')
        plt.xlabel('Iteración')
        plt.ylabel('Costo')
        plt.grid(True, alpha=0.3)
        
        # Mostrar información de convergencia
        final_cost = model.cost_history[-1]
        iterations = len(model.cost_history)
        plt.text(0.6, 0.8, f'Costo final: {final_cost:.4f}\nIteraciones: {iterations}', 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('gradiente_descendente_comparacion.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("="*80)
    print("IMPLEMENTACIÓN DE REGRESIÓN LINEAL DESDE CERO")
    print("(Sin librerías de Machine Learning)")
    print("="*80)
    
    print("\n1. REGRESIÓN LINEAL SIMPLE")
    print("-" * 40)
    X_simple, y_simple, true_weights, true_bias = generar_datos_sinteticos(
        n_samples=100, n_features=1, noise=10
    )
    
    print(f"Pesos verdaderos: {true_weights}")
    print(f"Bias verdadero: {true_bias:.4f}")
    
    model_simple = RegresionLineal(learning_rate=0.01, max_iter=1000)
    model_simple.fit(X_simple, y_simple, verbose=True)
    
    learned_weights, learned_bias = model_simple.get_weights()
    print(f"Pesos aprendidos: {learned_weights}")
    print(f"Bias aprendido: {learned_bias:.4f}")
    
    visualizar_resultados(model_simple, X_simple, y_simple, "Regresión Lineal Simple")
    
    print("\n2. REGRESIÓN LINEAL MÚLTIPLE")
    print("-" * 40)
    X_multiple, y_multiple, true_weights_multi, true_bias_multi = generar_datos_sinteticos(
        n_samples=200, n_features=3, noise=15
    )
    
    print(f"Pesos verdaderos: {true_weights_multi}")
    print(f"Bias verdadero: {true_bias_multi:.4f}")
    
    model_multiple = RegresionLineal(learning_rate=0.01, max_iter=1000)
    model_multiple.fit(X_multiple, y_multiple, verbose=True)
    
    learned_weights_multi, learned_bias_multi = model_multiple.get_weights()
    print(f"Pesos aprendidos: {learned_weights_multi}")
    print(f"Bias aprendido: {learned_bias_multi:.4f}")
    
    visualizar_resultados(model_multiple, X_multiple, y_multiple, "Regresión Lineal Múltiple")
    
    demostrar_gradiente_descendente()
    
    print("\n3. COMPARACIÓN DE REGULARIZACIONES")
    print("-" * 40)
    X_regularization, y_regularization, _, _ = generar_datos_sinteticos(
        n_samples=150, n_features=5, noise=20
    )
    
    results = comparar_regularizaciones(X_regularization, y_regularization)
    
    print("\nResumen de Resultados:")
    print("-" * 80)
    print(f"{'Modelo':<20} {'MSE':<10} {'R²':<10} {'MAE':<10} {'CV MSE':<15}")
    print("-" * 80)
    for result in results:
        print(f"{result['Modelo']:<20} {result['MSE']:<10.4f} {result['R²']:<10.4f} "
              f"{result['MAE']:<10.4f} {result['CV_MSE_Mean']:.4f}±{result['CV_MSE_Std']:.4f}")
    
    print("\n4. ANÁLISIS DE CONVERGENCIA")
    print("-" * 40)
    
    configs = [
        {"lr": 0.001, "iter": 2000, "name": "LR Baja, Muchas Iteraciones"},
        {"lr": 0.01, "iter": 1000, "name": "LR Media, Iteraciones Medias"},
        {"lr": 0.1, "iter": 500, "name": "LR Alta, Pocas Iteraciones"}
    ]
    
    plt.figure(figsize=(15, 5))
    
    for i, config in enumerate(configs):
        plt.subplot(1, 3, i+1)
        
        model = RegresionLineal(learning_rate=config["lr"], max_iter=config["iter"])
        model.fit(X_simple, y_simple, verbose=False)
        
        plt.plot(model.cost_history, 'b-', linewidth=2)
        plt.title(config["name"])
        plt.xlabel('Iteración')
        plt.ylabel('Costo')
        plt.grid(True, alpha=0.3)
        
        # Mostrar información
        final_cost = model.cost_history[-1]
        iterations = len(model.cost_history)
        plt.text(0.6, 0.8, f'Costo: {final_cost:.4f}\nIter: {iterations}', 
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('analisis_convergencia.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
    print("Archivos generados:")
    print("- regresion_lineal_simple_analisis.png")
    print("- regresion_lineal_múltiple_analisis.png")
    print("- gradiente_descendente_comparacion.png")
    print("- analisis_convergencia.png")
    print("="*80)
    print("Análisis completado")

if __name__ == "__main__":
    main()