# ==============================================================================
# PROYECTO MAESTRÍA EN IA - UNIVERSIDAD CÉSAR VALLEJO
# APLICACIÓN DE MACHINE LEARNING PARA PREDECIR EL RENDIMIENTO ACADÉMICO EN ESTUDIANTES DE PRIMARIA
# Autores: Arbieto Ramos Julio Vidal & Bances Ramírez Katherine del Milagro
# Entorno: Google Colab / Python 3.11 / Scikit-Learn 1.5.1
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib

# 1. CARGA DE DATOS
# Subir archivo dataset_estudiantes_ugel04.csv a Colab
df = pd.read_csv('dataset_estudiantes_ugel04.csv')
print(f"Dataset cargado con {df.shape[0]} estudiantes y {df.shape[1]} columnas.\n")

# 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
print("Distribución de Estudiantes por Grado:")
print(df['Grado'].value_counts())
print("\nDistribución por Nivel de Riesgo Predictivo:")
print(df['Nivel_Riesgo'].value_counts())

# 3. PREPARACIÓN DE DATOS & CODIFICACIÓN DE VARIABLES (CRISP-DM Fase 3)
df_model = df.copy()
df_model['Participacion_Num'] = df_model['Participación_Académica'].map({'Bajo': 1, 'Medio': 2, 'Alto': 3})
df_model['Acompanamiento_Num'] = df_model['Acompañamiento_Familiar'].map({'Bajo': 1, 'Medio': 2, 'Alto': 3})
df_model['Sexo_Num'] = df_model['Sexo'].map({'Masculino': 1, 'Femenino': 0})
df_model['Grado_Num'] = df_model['Grado'].map({'4to Grado': 4, '6to Grado': 6})

feature_cols = [
    'Edad', 'Sexo_Num', 'Grado_Num', 'Asistencia_Pct', 'Rendimiento_Previo',
    'Evaluación_Diagnóstica', 'Participacion_Num', 'Cumplimiento_Tareas_Pct', 'Acompanamiento_Num'
]

X = df_model[feature_cols]
y = df_model['Requiere_Refuerzo']

# 4. DIVISION DE DATOS: 80% ENTRENAMIENTO / 20% PRUEBA (Tesis Pág. 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"\nDatos Entrenamiento: {X_train.shape[0]} muestras | Datos Prueba: {X_test.shape[0]} muestras")

# 5. MODELADO CON RANDOM FOREST & 10-FOLD CROSS VALIDATION (CRISP-DM Fase 4)
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

# Validacion Cruzada 10-Fold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_validate(rf_classifier, X_train, y_train, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])

print("\n--- RESULTADOS DE VALIDACIÓN CRUZADA (10-FOLD CV) ---")
print(f"Accuracy Promedio CV : {cv_scores['test_accuracy'].mean():.4f}")
print(f"Recall Promedio CV   : {cv_scores['test_recall'].mean():.4f}")
print(f"F1-Score Promedio CV : {cv_scores['test_f1'].mean():.4f}")
print(f"AUC-ROC Promedio CV  : {cv_scores['test_roc_auc'].mean():.4f}")

# 6. EVALUACIÓN DEL MODELO EN CONJUNTO DE PRUEBA (CRISP-DM Fase 5)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
y_proba = rf_classifier.predict_proba(X_test)[:, 1]

print("\n--- MÉTRICAS EN CONJUNTO DE PRUEBA (20% TEST SET) ---")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC   : {roc_auc_score(y_test, y_proba):.4f}")

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Refuerzo (0)', 'Requiere Refuerzo (1)']))

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Refuerzo', 'Requiere Refuerzo'], yticklabels=['No Refuerzo', 'Requiere Refuerzo'])
plt.title('Matriz de Confusión - Random Forest')
plt.xlabel('Predicción del Modelo')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.savefig('matriz_confusion_rf.png', dpi=300)
plt.show()

# Importancia de Variables (Feature Importance)
feature_names = ['Edad', 'Sexo', 'Grado', 'Asistencia (%)', 'Rendimiento Previo',
                 'Evaluación Diagnóstica', 'Participación Académica', 'Cumplimiento Tareas (%)', 'Acompañamiento Familiar']
importances = pd.Series(rf_classifier.feature_importances_, index=feature_names).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='#1B365D')
plt.title('Importancia de Variables en Predicción de Rendimiento (Feature Importance)')
plt.xlabel('Nivel de Importancia Relativa')
plt.tight_layout()
plt.savefig('importancia_variables_rf.png', dpi=300)
plt.show()

# 7. GUARDAR MODELO ENTRENADO
joblib.dump(rf_classifier, 'modelo_random_forest.joblib')
print("Modelo guardado exitosamente como 'modelo_random_forest.joblib'.")
