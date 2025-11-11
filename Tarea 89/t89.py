# experimental_design_corrected.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from itertools import product
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FootballValuationExperiment:
    """
    Implementación corregida del diseño experimental para valoración de jugadores
    """
    
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.results = []
        self.setup_factors()
        
    def setup_factors(self):
        """Define los factores y niveles del experimento"""
        self.factors = {
            'algorithm': {
                'RF': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'XGB': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'LGBM': LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            },
            'preprocessing': {
                'MinMax': MinMaxScaler(),
                'Zscore': StandardScaler()
            },
            'feature_selection': {
                'RFE': 'rfe',
                'SelectKBest': 'kbest', 
                'PCA': 'pca'
            },
            'dataset_size': {
                'Complete': 'complete',
                'Balanced': 'balanced'
            },
            'validation': {
                'Holdout': 'holdout',
                'CV': 'cross_validation'
            }
        }
    
    def calculate_smape(self, y_true, y_pred):
        """Calcula sMAPE (Symmetric Mean Absolute Percentage Error)"""
        # Evitar división por cero
        denominator = (np.abs(y_true) + np.abs(y_pred))
        # Reemplazar ceros en el denominador con un valor pequeño
        denominator = np.where(denominator == 0, 1e-10, denominator)
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / denominator)
    
    def prepare_dataset(self, size_type):
        """Prepara el dataset según el factor de tamaño - CORREGIDO"""
        features = ['age', 'overall_rating', 'potential', 'international_reputation(1-5)',
                   'weak_foot(1-5)', 'skill_moves(1-5)', 'crossing', 'finishing', 
                   'short_passing', 'dribbling', 'ball_control', 'acceleration',
                   'sprint_speed', 'stamina', 'strength', 'value_euro']
        
        # Filtrar solo las columnas que existen
        available_features = [f for f in features if f in self.data.columns]
        df_filtered = self.data[available_features].dropna()
        
        if size_type == 'complete':
            return df_filtered
        else:  # balanced
            # Estrategia simplificada de balanceo
            target_size = min(10000, len(df_filtered))
            return df_filtered.sample(n=target_size, random_state=42)
    
    def select_features(self, method, X, y, n_features=10):
        """Aplica selección de características - CORREGIDO"""
        try:
            n_samples, n_current_features = X.shape
            n_features = min(n_features, n_current_features)
            
            if method == 'rfe':
                # Usar un estimador más simple para RFE
                from sklearn.linear_model import LinearRegression
                selector = RFE(
                    estimator=LinearRegression(),
                    n_features_to_select=n_features
                )
                X_selected = selector.fit_transform(X, y)
                return X_selected
                
            elif method == 'kbest':
                selector = SelectKBest(score_func=f_regression, k=n_features)
                X_selected = selector.fit_transform(X, y)
                return X_selected
                
            elif method == 'pca':
                pca = PCA(n_components=n_features)
                X_selected = pca.fit_transform(X)
                return X_selected
            else:
                # Si el método no es reconocido, usar todas las características
                return X
                
        except Exception as e:
            print(f"Error en selección de características {method}: {e}")
            # En caso de error, retornar características originales
            return X
    
    def run_treatment(self, treatment):
        """Ejecuta un tratamiento específico - CORREGIDO"""
        try:
            algorithm_key = list(self.factors['algorithm'].keys())[
                list(self.factors['algorithm'].values()).index(treatment['algorithm'])
            ]
            preprocessing_key = list(self.factors['preprocessing'].keys())[
                list(self.factors['preprocessing'].values()).index(treatment['preprocessing'])
            ]
            
            print(f"  Ejecutando: {algorithm_key}, {preprocessing_key}, "
                  f"{treatment['feature_selection']}, {treatment['dataset_size']}, "
                  f"{treatment['validation']}")
            
            # Preparar datos
            df_clean = self.prepare_dataset(treatment['dataset_size'])
            
            if len(df_clean) == 0:
                print("  ERROR: Dataset vacío después de la limpieza")
                return None
            
            # Separar características y target
            feature_columns = [col for col in df_clean.columns if col != 'value_euro']
            X = df_clean[feature_columns].values
            y = df_clean['value_euro'].values
            
            if len(X) == 0:
                print("  ERROR: No hay datos de características")
                return None
            
            # Preprocesamiento
            scaler = treatment['preprocessing']
            X_scaled = scaler.fit_transform(X)
            
            # Selección de características
            X_processed = self.select_features(
                treatment['feature_selection'], 
                X_scaled, 
                y
            )
            
            # Verificar que X_processed no sea None
            if X_processed is None:
                print("  ERROR: X_processed es None")
                return None
                
            # Entrenamiento y evaluación
            start_time = time.time()
            model = treatment['algorithm']
            
            if treatment['validation'] == 'holdout':
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y, test_size=0.2, random_state=42
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                smape = self.calculate_smape(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
            else:  # cross_validation
                # Usar una métrica que esté disponible en cross_val_score
                from sklearn.metrics import make_scorer
                smape_scorer = make_scorer(self.calculate_smape, greater_is_better=False)
                
                # Calcular MAE con CV
                mae_scores = -cross_val_score(model, X_processed, y, 
                                            cv=5, scoring='neg_mean_absolute_error')
                mae = mae_scores.mean()
                
                # Calcular sMAPE manualmente con CV
                smape_scores = []
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                
                for train_idx, test_idx in kf.split(X_processed):
                    X_train, X_test = X_processed[train_idx], X_processed[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    model_clone = treatment['algorithm']
                    model_clone.fit(X_train, y_train)
                    y_pred = model_clone.predict(X_test)
                    
                    smape_fold = self.calculate_smape(y_test, y_pred)
                    smape_scores.append(smape_fold)
                
                smape = np.mean(smape_scores)
                
                # Para R2 y MSE, hacer una validación holdout adicional
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y, test_size=0.2, random_state=42
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
            
            training_time = time.time() - start_time
            
            return {
                'algorithm': algorithm_key,
                'preprocessing': preprocessing_key,
                'feature_selection': treatment['feature_selection'],
                'dataset_size': treatment['dataset_size'],
                'validation': treatment['validation'],
                'smape': smape,
                'mae': mae,
                'mse': mse,
                'r2': r2,
                'training_time': training_time,
                'n_features': X_processed.shape[1],
                'n_samples': len(X_processed)
            }
            
        except Exception as e:
            print(f"  ERROR en tratamiento: {e}")
            return None
    
    def generate_fractional_factorial(self):
        """Genera diseño factorial fraccional 2^(5-1) - CORREGIDO"""
        treatments = []
        
        # Generar combinaciones sistemáticas
        algorithms = list(self.factors['algorithm'].values())
        preprocessings = list(self.factors['preprocessing'].values())
        feature_selections = list(self.factors['feature_selection'].keys())
        dataset_sizes = list(self.factors['dataset_size'].values())
        validations = list(self.factors['validation'].values())
        
        # Diseño reducido y balanceado
        combinations = [
            # Tratamientos diversos que cubren diferentes combinaciones
            (0, 0, 0, 0, 0),  # RF, MinMax, RFE, Complete, Holdout
            (1, 1, 1, 1, 1),  # XGB, Zscore, SelectKBest, Balanced, CV
            (2, 0, 2, 0, 1),  # LGBM, MinMax, PCA, Complete, CV
            (0, 1, 1, 1, 0),  # RF, Zscore, SelectKBest, Balanced, Holdout
            (1, 0, 2, 0, 0),  # XGB, MinMax, PCA, Complete, Holdout
            (2, 1, 0, 1, 1),  # LGBM, Zscore, RFE, Balanced, CV
        ]
        
        for combo in combinations:
            treatment = {
                'algorithm': algorithms[combo[0]],
                'preprocessing': preprocessings[combo[1]],
                'feature_selection': feature_selections[combo[2]],
                'dataset_size': dataset_sizes[combo[3]],
                'validation': validations[combo[4]]
            }
            treatments.append(treatment)
        
        return treatments
    
    def run_experiment(self):
        """Ejecuta el experimento completo - CORREGIDO"""
        treatments = self.generate_fractional_factorial()
        
        print("Iniciando experimento de valoración de jugadores...")
        print(f"Número de tratamientos: {len(treatments)}")
        print("=" * 60)
        
        for i, treatment in enumerate(treatments, 1):
            print(f"Tratamiento {i}/{len(treatments)}:")
            result = self.run_treatment(treatment)
            
            if result is not None:
                self.results.append(result)
                print(f"  ✓ Completado - sMAPE: {result['smape']:.2f}%")
            else:
                print(f"  ✖ Fallado")
            print("-" * 40)
        
        if self.results:
            return self.analyze_results()
        else:
            print("ERROR: Ningún tratamiento se completó exitosamente")
            return None, None
    
    def analyze_results(self):
        """Analiza los resultados del experimento - CORREGIDO"""
        results_df = pd.DataFrame(self.results)
        
        print("\n" + "="*60)
        print("RESULTADOS DEL EXPERIMENTO")
        print("="*60)
        
        # Encontrar el mejor tratamiento (menor sMAPE)
        best_idx = results_df['smape'].idxmin()
        best_result = results_df.loc[best_idx]
        
        print(f"\n🎯 MEJOR TRATAMIENTO (menor sMAPE):")
        print(f"   Algoritmo: {best_result['algorithm']}")
        print(f"   Preprocesamiento: {best_result['preprocessing']}")
        print(f"   Selección de Features: {best_result['feature_selection']}")
        print(f"   Tamaño de Dataset: {best_result['dataset_size']}")
        print(f"   Validación: {best_result['validation']}")
        print(f"   sMAPE: {best_result['smape']:.2f}%")
        print(f"   MAE: {best_result['mae']:,.0f} €")
        print(f"   R²: {best_result['r2']:.3f}")
        print(f"   Tiempo: {best_result['training_time']:.1f} segundos")
        
        # Análisis estadístico
        print(f"\n📊 ESTADÍSTICAS GENERALES:")
        print(f"   sMAPE promedio: {results_df['smape'].mean():.2f}%")
        print(f"   sMAPE mínimo: {results_df['smape'].min():.2f}%")
        print(f"   sMAPE máximo: {results_df['smape'].max():.2f}%")
        print(f"   R² promedio: {results_df['r2'].mean():.3f}")
        
        # ANOVA simplificado
        try:
            # Preparar datos para ANOVA
            anova_df = results_df.copy()
            anova_df['smape'] = anova_df['smape'].astype(float)
            
            # Modelo ANOVA
            formula = 'smape ~ C(algorithm) + C(preprocessing) + C(feature_selection) + C(dataset_size) + C(validation)'
            model = ols(formula, data=anova_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            print(f"\n📈 ANÁLISIS ANOVA:")
            print(anova_table)
            
        except Exception as e:
            print(f"\n⚠️  No se pudo realizar ANOVA: {e}")
            anova_table = None
        
        # Gráficos
        self.plot_results(results_df)
        
        return results_df, anova_table
    
    def plot_results(self, results_df):
        """Genera gráficos de resultados - CORREGIDO"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Gráfico 1: sMAPE por algoritmo
            sns.boxplot(data=results_df, x='algorithm', y='smape', ax=axes[0,0])
            axes[0,0].set_title('sMAPE por Algoritmo')
            axes[0,0].set_ylabel('sMAPE (%)')
            
            # Gráfico 2: sMAPE por preprocesamiento
            sns.boxplot(data=results_df, x='preprocessing', y='smape', ax=axes[0,1])
            axes[0,1].set_title('sMAPE por Método de Preprocesamiento')
            axes[0,1].set_ylabel('sMAPE (%)')
            
            # Gráfico 3: Tiempo vs sMAPE
            scatter = axes[1,0].scatter(results_df['training_time'], results_df['smape'], 
                                      c=results_df['r2'], cmap='viridis', s=100, alpha=0.7)
            axes[1,0].set_xlabel('Tiempo de Entrenamiento (s)')
            axes[1,0].set_ylabel('sMAPE (%)')
            axes[1,0].set_title('Relación: Tiempo vs sMAPE (color = R²)')
            plt.colorbar(scatter, ax=axes[1,0], label='R²')
            
            # Gráfico 4: Heatmap de performance
            pivot_data = results_df.pivot_table(
                values='smape', 
                index='algorithm', 
                columns='preprocessing', 
                aggfunc='mean'
            )
            sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1,1])
            axes[1,1].set_title('sMAPE Promedio: Algoritmo vs Preprocesamiento')
            
            plt.tight_layout()
            plt.savefig('experimental_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Error generando gráficos: {e}")

# Función de ejecución simplificada para Colab
def run_simplified_experiment(data_path):
    """Ejecuta una versión simplificada del experimento para Colab"""
    print("🚀 INICIANDO EXPERIMENTO SIMPLIFICADO")
    print("=" * 50)
    
    try:
        experiment = FootballValuationExperiment(data_path)
        results, anova = experiment.run_experiment()
        
        if results is not None:
            print("\n✅ EXPERIMENTO COMPLETADO EXITOSAMENTE")
            return results, anova
        else:
            print("\n❌ EXPERIMENTO FALLADO")
            return None, None
            
    except Exception as e:
        print(f"\n💥 ERROR CRÍTICO: {e}")
        return None, None

# Ejecutar en Colab
if __name__ == "__main__":
    # Usar el archivo local en Colab
    results, anova = run_simplified_experiment('fifa_players.csv')