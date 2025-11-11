# supervised_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import time

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_preprocess_data(file_path):
    """Carga y preprocesa los datos"""
    
    # Selección de características
    features = [
        'age', 'overall_rating', 'potential', 'international_reputation(1-5)',
        'weak_foot(1-5)', 'skill_moves(1-5)', 'crossing', 'finishing', 
        'short_passing', 'dribbling', 'ball_control', 'acceleration',
        'sprint_speed', 'stamina', 'strength'
    ]
    
    # Limpieza de datos
    df_clean = df[features + ['value_euro']].dropna()
    
    return df_clean, features

def evaluate_model(y_true, y_pred, model_name):
    """Evalúa el modelo y retorna métricas"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'Modelo': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

def plot_feature_importance(importance_df, top_n=10):
    """Grafica la importancia de características"""
    plt.figure(figsize=(10, 6))
    top_features = importance_df.head(top_n)
    
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importancia')
    plt.title(f'Top {top_n} Características Más Importantes', fontsize=20)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Cargar datos
    df, features = load_and_preprocess_data('fifa_players.csv')
    
    # Preparar datos
    X = df[features]
    y = df['value_euro']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Modelos a evaluar
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        'Regresión Lineal': LinearRegression()
    }
    
    # Entrenar y evaluar modelos
    results = []
    training_times = {}
    
    for name, model in models.items():
        print(f"Entrenando {name}...")
        
        # Medir tiempo de entrenamiento
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        training_times[name] = training_time
        
        # Predecir
        y_pred = model.predict(X_test)
        
        # Evaluar
        metrics = evaluate_model(y_test, y_pred, name)
        metrics['Tiempo Entrenamiento'] = training_time
        results.append(metrics)
        
        print(f"{name} - MAE: {metrics['MAE']:,.2f} €, R²: {metrics['R2']:.3f}")
    
    # Resultados en DataFrame
    results_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("RESULTADOS COMPARATIVOS")
    print("="*50)
    print(results_df.round(4))
    
    # Importancia de características (del mejor modelo)
    best_model = models['Random Forest']
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nIMPORTANCIA DE CARACTERÍSTICAS:")
        print(importance_df)
        
        # Graficar importancia
        plot_feature_importance(importance_df)
    
    # Gráfico de predicciones vs reales
    plt.figure(figsize=(10, 6))
    y_pred_best = best_model.predict(X_test)
    plt.scatter(y_test, y_pred_best, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valor Real (€)', fontsize=20)
    plt.ylabel('Valor Predicho (€)', fontsize=20)
    plt.title('Predicciones vs Valores Reales - Random Forest', fontsize=20)
    plt.tight_layout()
    plt.savefig('predicciones_vs_reales.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df, importance_df

if __name__ == "__main__":
    results, importance = main()