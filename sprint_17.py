#!/usr/bin/env python
# coding: utf-8


# # Sprint 17 Proyecto final

# # **1. Descripción del proyecto**

# Al operador de telecomunicaciones Interconnect le gustaría poder pronosticar su tasa de cancelación de clientes. Si se descubre que un usuario o usuaria planea irse, se le ofrecerán códigos promocionales y opciones de planes especiales. El equipo de marketing de Interconnect ha recopilado algunos de los datos personales de sus clientes, incluyendo información sobre sus planes y contratos.

# 1. Objetivo Principal
#
# Pronosticar la tasa de cancelación de clientes de Interconnect para identificar usuarios propensos a abandonar el servicio y ofrecerles incentivos adecuados.
#
# 2. Métricas de Evaluación
#
# Característica objetivo: la columna `'EndDate'` es igual a `'No'`.
#
# Métrica principal: AUC-ROC.
#
# Métrica Adicional: Exactitud (Accuracy)
#
# Métricas que sirven para desbalanceo: Recall-Sensibilidad, Average precision, F1 y Accuray.
#
# Estas métricas se tendrán en cuenta, ya que AUC-ROC no funciona con desbalanceo y puede dar un valor erróneo debido a esto
#

# ## 1.1 Plan de Trabajo

# 1. Importar Librerías Necesarias
# 2. Cargar los Datos
# 3. Exploración Inicial de los Datos
# 4. Limpieza y Preprocesamiento de los Datos
# 5. Fusión de los Datos
# 6. Examinar el equilibrio de clases.
# 7. Análisis estadístico de datos
# 8. Ingeniería de Características (Feature Engineering)
# 9. Preparación para el Modelado
# 10. Mejora la calidad del modelo. (submuestreo y sobremuestreo)
# 11. Entrenamiento de Modelos
# - Regresión Logística
# - Decision Tree
# - Random Forest
# - Gradient Boosting Machines (CatBoost)
# - XGBoost
# - Support Vector Machine (SVM)
# 12. Evaluación del Modelo
# 13. Selección del Mejor Modelo. Utilizar conjuntos de entrenamiento y validación para encontrar el mejor modelo y el mejor conjunto de parámetros.Entrenar diferentes modelos en los conjuntos de entrenamiento y validación.
# 14. Comprobar la calidad del modelo usando el conjunto de prueba.
#

# # **2. Importar datos y librerías**

# ## 2.1 Importar las librerías

# In[136]:


# 1. Librerías estándar de Python
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from google.colab import drive
from catboost import CatBoostClassifier
from boruta import BorutaPy
import math

# 2. Librerías de terceros
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

# 3. Scikit-learn (modelos y métricas)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  # Fixed typo in module name
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

# 4. Librerías adicionales (Boruta, SHAP, CatBoost)
get_ipython().system('pip install boruta')
get_ipython().system('pip install shap')
get_ipython().system('pip install catboost')


# ## 2.2 Importar datos

# In[137]:


drive.mount('/content/drive')


# In[138]:


ruta_phone = '/content/drive/My Drive/proyecto_final_tripleten/phone.csv'
ruta_personal = '/content/drive/My Drive/proyecto_final_tripleten/personal.csv'
ruta_contract = '/content/drive/My Drive/proyecto_final_tripleten/contract.csv'
ruta_internet = '/content/drive/My Drive/proyecto_final_tripleten/internet.csv'


# In[139]:


df_phone = pd.read_csv(ruta_phone)
df_personal = pd.read_csv(ruta_personal)
df_contract = pd.read_csv(ruta_contract)
df_internet = pd.read_csv(ruta_internet)

# ## 2.3 Información Inicial de los Datos

# In[140]:


print('Phone Data')
display(df_phone.sample(15))
print()
print('Personal Data')
display(df_personal.sample(15))
print()
print('Contract Data')
display(df_contract.sample(15))
print()
print('Internet Data')
display(df_internet.sample(15))


# In[141]:


print('Phone Data')
print(df_phone.info())
print()
print('Personal Data')
print(df_personal.info())
print()
print('Contract Data')
print(df_contract.info())
print()
print('Internet Data')
print(df_internet.info())


# ## **3. Análisis Exploratorio de Datos (EDA)**

# Limpieza y Preprocesamiento de los Datos

# ## 3.1 correción del tipo de datos

# In[142]:


# Contract Data

df_contract['TotalCharges'] = pd.to_numeric(
    df_contract['TotalCharges'], errors='coerce')


# In[143]:


print('Contract Data')
print(df_contract.info())

# ## 3.2 Manejo de Valores Nulos y Duplicados

# In[144]:


# Phone Data

print('Phone Data')
print('Valores nulos:')
print(df_phone.isnull().sum())
print()
print('Valores duplicados:', df_phone.duplicated().sum())
print()


# In[145]:


# Personal Data

print('Personal Data')
print('Valores nulos:')
print(df_personal.isnull().sum())
print()
print('Valores duplicados:', df_personal.duplicated().sum())
print()


# In[146]:


# Contract Data

print('Contract Data')
print('Valores nulos:')
print(df_contract.isnull().sum())
print()
print('Valores duplicados:', df_contract.duplicated().sum())
print()


# In[147]:


df_contract['TotalCharges'] = df_contract['TotalCharges'].fillna(0)
print(df_contract.isnull().sum())


# In[148]:


# Internet Data

print('Internet Data')
print('Valores nulos:')
print(df_internet.isnull().sum())
print()
print('Valores duplicados:', df_internet.duplicated().sum())
print()

# ## 3.3 Integración de Datos

# In[149]:


# Fusionar Phone y Personal Data
df = df_phone.merge(df_personal, on='customerID', how='left')

# Fusionar con Contract Data
df = df.merge(df_contract, on='customerID', how='left')

# Fusionar con Internet Data
df = df.merge(df_internet, on='customerID', how='left')

# Verificar el tamaño del DataFrame fusionado
print("Tamaño del DataFrame fusionado:", df.shape)

df.drop_duplicates(inplace=True)

print(df)


# ## 3.4 Definición de la Variable Objetivo

# 1: Indica que el cliente ha cancelado (cuando 'EndDate' contiene una fecha).
#
# 0: Indica que el cliente no ha cancelado (cuando 'EndDate' es 'No').

# In[150]:


df['Churn'] = df['EndDate'].apply(lambda x: 1 if x != 'No' else 0)


# ## 3.5 Eliminar Columnas innecesarias

# In[151]:


df.drop(['customerID', 'EndDate'], axis=1, inplace=True)


# In[152]:


print(df.sample(15))

# ## 3.6 Examina el equilibrio de clases.

# In[153]:


# Examinar las clases

class_distribution = df['Churn'].value_counts()
print("Distribución de clases:")
print(class_distribution)


# In[154]:


class_distribution.plot(
    kind='bar', x=class_distribution.index, y=class_distribution.values)

plt.show()


# Distribución de clases:
#
# - 0 (No ha cancelado): 4,662 clientes (lo que representa aproximadamente el 66.2% de los datos).
#
# - 1 (Ha cancelado): 1,699 clientes (aproximadamente el 33.8% de los datos).
#
# La distribución de la clase está desbalanceada, con más clientes que no han cancelado (66.2%) que aquellos que sí han cancelado (33.8%). Lo cual se debe de corregir antes de empezar a modelar.


# # **6.  Análisis estadístico de datos**

# In[155]:


print(df.sample(20))
print()
print(df.isnull().sum())
print()
print(df.info())


# In[156]:


cols_to_fill = [
    'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies'
]

for col in cols_to_fill:
    df[col].fillna('No', inplace=True)

print(df.isnull().sum())


# In[157]:


df['InternetService'] = df['InternetService'].str.replace(' ', '_').str.lower()

# Verificar los cambios
print(df['InternetService'].unique())


# In[158]:


print(df.sample(15))


# In[159]:


print(df['InternetService'].unique())

# ## Correlaciones entre la duración del contrato y la cancelación

# In[160]:


# Filtrar los datos para incluir solo a los que han cancelado (Churn == 1)
churned_users = df[df['Churn'] == 1]

# Crear una tabla de frecuencia para la duración del contrato
contract_churn = churned_users.groupby('Type')['Churn'].count().reset_index()
contract_churn['Churn'] = contract_churn['Churn'] / churned_users.shape[0]

print(contract_churn['Churn'])

# Crear un gráfico de barras
plt.figure(figsize=(8, 6))
sns.barplot(x='Type', y='Churn', data=contract_churn, palette='Blues_d')
plt.title('Tasa de Cancelación por Duración del Contrato (Usuarios que Cancelaron)')
plt.xlabel('Duración del Contrato')
plt.ylabel('Proporción de Cancelaciones')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# El gráfico buscaba obtener la correlación de la duración del contrato con la tasa de cancelación de los usuarios, esto con el fin de observar qué tipo de ususarios son más propensos a cancelar.
#
# Esto fue lo que se observó:
#
# - Contratos Mes a Mes:
#
# Se observa una correlación positiva muy alta (0.88) entre los contratos mes a mes y la tasa de cancelación.
#
# Esto indica que los usuarios con este tipo de contrato son significativamente más propensos a cancelar sus servicios.
#
# Este tipo de usuarios son de alto riesgo y se deben de considerar estrategias para lograr su retención a largo plazo, como la oferta de códigos promocionales y planes especiales por parte del equipo de marketing.
#
# - Contratos de Un Año:
#
# La tasa de cancelación para los contratos de un año es baja (0.09), lo que sugiere que estos clientes son más estables en comparación con los de contratos mes a mes.
#
# - Contrataos de Dos años
#
# La correlación es prácticamente nula (0.03), indicando que los usuarios con contratos de dos años tienen la menor tasa de cancelación. Esto refleja la efectividad de los contratos a largo plazo en la retención de clientes.


# ## Prueba de Hipótesis

# In[161]:


df['InternetService'] = df['InternetService'].apply(
    lambda x: 'Yes' if str(x).strip().lower() in ['dsl', 'fiber_optic'] else 'No')

print(df['InternetService'].unique())


# In[162]:


# Prueba las hipótesis

# Hipótesis nula = La tasa de cancelación de los usuarios que contrataron solo internet y de los usuarios que contrataron solo teléfono fijo es igual
# Hipótesis alternativa = La tasa de cancelación de los usuarios que contrataron solo internet y de los usuarios que contrataron solo teléfono fijo es diferente
# Prueba estadística = Prueba de Chi-cuadrado
# alpha = 0.05


# Filtrar usuarios que han cancelado
churned_users = df[df['Churn'] == 1]

# Obtener usuarios con solo servicio de teléfono y solo internet
total_only_phone_users = df[(df['MultipleLines'] == 'Yes') & (
    df['InternetService'] == 'No')]
total_only_internet_users = df[(df['MultipleLines'] == 'No') & (
    df['InternetService'] == 'Yes')]

print(f"Número total de usuarios con solo teléfono: {
      len(total_only_phone_users)}")
print(f"Número total de usuarios con solo internet: {
      len(total_only_internet_users)}")
print(f"Número total de usuarios que han cancelado: {len(churned_users)}")

# Crear la tabla de contingencia
contingency_table = pd.crosstab(
    df['InternetService'], df['Churn'], margins=False)

# Realizar la prueba de Chi-cuadrado
chi2_stat, p_val, dof, expected = st.chi2_contingency(contingency_table)

# Mostrar los resultados
print("Tabla de contingencia para la prueba de Chi-cuadrado:")
print(contingency_table)
print(f"Valor p (Chi-cuadrado): {p_val}")

alpha = 0.05
if p_val < alpha:
    print("Rechazamos la hipótesis nula: La tasa de cancelación es diferente entre los usuarios de solo internet y solo teléfono.")
else:
    print("No podemos rechazar la hipótesis nula: No hay diferencia significativa en la tasa de cancelación entre los usuarios de solo internet y solo teléfono.")


# El objetivo de era determinar si la tasa de cancelación de los usuarios que contrataron solo internet y de los usuarios que contrataron solo teléfono es igual o diferente, utilizando una prueba de hipótesis basada en la prueba de Chi-cuadrado

# La elección de la hipótesis nula se basó en la idea de que no hay diferencias entre los grupos, y se estableció una significancia alfa de 0.05, que es un valor comúnmente utilizado como estándar en la mayoría de las pruebas de hipótesis.
#
# Hipótesis:
#
# - Hipótesis nula (H₀): No hay diferencia significativa en la tasa de cancelación entre los usuarios que contrataron solo internet y los usuarios que contrataron solo teléfono.
#
# - Hipótesis alternativa (H₁): La tasa de cancelación de los usuarios que contrataron solo internet es diferente a la de los usuarios que contrataron solo teléfono.


# Dado que el valor p (7.90e-85) es muy pequeño, mucho menor que el nivel de significancia (0.05), podemos rechazar la hipótesis nula.
#
# Esto significa que existen diferencias significativas en la tasa de cancelación entre los usuarios que tienen solo internet y los usuarios que tienen solo teléfono.
#
# La tasa de cancelación es significativamente mayor en los usuarios que tienen solo internet en comparación con aquellos que tienen solo teléfono. Este hallazgo es importante, ya que sugiere que los clientes que solo adquieren un servicio (especialmente internet) tienen una mayor probabilidad de cancelar su suscripción en comparación con aquellos que tienen otros servicios como el de teléfono.


# # **3. Ingeniería de Características (Feature Engineering)**

# ## 3.1 Seleccionar Características

# In[163]:


features = df.drop(columns=['Churn'])
target = df['Churn']


# ## 3.2 Segmentación de los Datos en el Conjunto de Entrenamiento, uno de Validación y uno de Prueba.


# División del conjunto de datos en dos: conjunto de prueba (20%) y el resto (conjunto de validación + conjunto de entrenamiento) (80%)
features_80, features_test, target_80, target_test = train_test_split(
    features, target, test_size=0.20, random_state=12345)

# División del conjunto de datos restante en conjunto de entrenamiento (60%) y conjunto de validación (20%)
features_train, features_valid, target_train, target_valid = train_test_split(
    features_80, target_80, test_size=0.25, random_state=12345)

# Codificación de variables categóricas (One-Hot Encoding)
features_train = pd.get_dummies(features_train)
features_valid = pd.get_dummies(features_valid)
features_test = pd.get_dummies(features_test)

features_valid = features_valid.reindex(
    columns=features_train.columns, fill_value=0)
features_test = features_test.reindex(
    columns=features_train.columns, fill_value=0)


# In[165]:


# Entrenar Boruta y seleccionar características
boruta_selector.fit(features_train.values, target_train.values)
features_selected = features_train.columns[boruta_selector.support_]

# Aplicar las características seleccionadas a los datasets de entrenamiento, validación y prueba
features_train_boruta = features_train[features_selected]
features_valid_boruta = features_valid[features_selected]
features_test_boruta = features_test[features_selected]

print("Características seleccionadas:", features_selected)


# Escalar datos

scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train_boruta)
features_valid_scaled = scaler.transform(features_valid_boruta)
features_test_scaled = scaler.transform(features_test_boruta)


# # **4. Mejora la calidad del modelo**

# ## 4.1 corregir el desequilibrio de clases.

# In[167]:


# Balanceo de Clases

get_ipython().system('pip install -q imblearn')

smote = SMOTE(random_state=42)
features_train_smote, target_train_smote = smote.fit_resample(
    features_train_scaled, target_train)


# In[168]:


smote = SMOTE(random_state=42)
features_train_smote, target_train_smote = smote.fit_resample(
    features_train_scaled, target_train)

# Verificar la nueva distribución de clases después de aplicar SMOTE
print("Distribución de clases antes de SMOTE:", Counter(target_train))
print("Distribución de clases después de SMOTE:", Counter(target_train_smote))


# # **5.** **Modelado**

# Al ser un proyecto de clasificación, vamos a utilizar modelos de clasificación como la Regresión logistica, arbol de desición, Random Forest,  Gradient Boosting Machines (CatBoost) y Redes Neuronales o SVM

# ## 5.1 Modelos

# In[169]:


get_ipython().system('pip install xgboost')

models = {
    'Logistic Regression': LogisticRegression(random_state=54321, solver='liblinear'),
    'Decision Tree': DecisionTreeClassifier(random_state=54321),
    'Random Forest': RandomForestClassifier(random_state=54321),
    'CatBoost': CatBoostClassifier(random_state=54321, verbose=0),
    'XGBoost': XGBClassifier(random_state=54321),
    'SVM': SVC(random_state=54321, probability=True)
}


# ## 5.2 Variación de hiperparametros

# In[170]:


param_distributions = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'max_iter': [100, 200, 300]
    },
    'Decision Tree': {
        'max_depth': [3, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'max_iter': [1000, 2000]
    }
}


# ## 5.3 Evaluación de los modelos

# In[171]:


for name, model in models.items():
    print(f"Evaluando {name} con SMOTE:")

    # RandomizedSearchCV para encontrar los mejores hiperparámetros
    search_algorithm = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions.get(name, {}),
        n_iter=10,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        random_state=42
    )

    search_algorithm.fit(features_train_smote, target_train_smote)

    best_model = search_algorithm.best_estimator_
    predictions_valid = best_model.predict(features_valid_scaled)
    probas_valid = best_model.predict_proba(features_valid_scaled)[:, 1]

    # Evaluación con AUC, F1 y Accuracy
    auc_roc = roc_auc_score(target_valid, probas_valid)
    f1 = f1_score(target_valid, predictions_valid)
    accuracy = accuracy_score(target_valid, predictions_valid)

    print(f"Mejor modelo para {name}: {best_model}")
    print(f"AUC-ROC: {auc_roc:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")


# ## 5.4 Mejor modelo y métricas

# In[172]:


best_model_final = best_model
predictions_test = best_model_final.predict(features_test_scaled)
probas_test = best_model_final.predict_proba(features_test_scaled)[:, 1]

auc_roc_test = roc_auc_score(target_test, probas_test)
f1_test = f1_score(target_test, predictions_test)
accuracy_test = accuracy_score(target_test, predictions_test)


# In[173]:


print(f"Evaluación en test: AUC-ROC: {auc_roc_test:.4f}, F1: {
      f1_test:.4f}, Accuracy: {accuracy_test:.4f}")
print(best_model_final)


# Según los resultados el mejor modelo de entramiento fue el SVM con SVC con los siguientes hiperparametros: C=10, max_iter=2000, probability=True, random_state=54321.
#
# El modelo es lo suficientemente completo para predecir cancelaciones con una precisión general alta (accuracy 75%) y una buena capacidad de discriminación (AUC-ROC 0.81). Esto significa que la empresa Interconnect podría implementar dicho modelo para identificar clientes en riesgo de cancelar.
#
#

# Por otro lado las métricas utilizadas indican lo siguiente:
#
# - El AUC-ROC mide la capacidad del modelo para distinguir entre las clases positivas y negativas (en este caso, clientes que cancelan y los que no).
#
#   Un AUC-ROC de 0.8121 indica una buena capacidad del modelo para discriminar entre usuarios que van a cancelar y los que no. Cuanto más cerca de 1, mejor. Un AUC-ROC de 0.8121 indica que el modelo tiene un buen rendimiento en la clasificación de clientes propensos a cancelar. En conclusión hay una probabilidad del 81.21% de que el modelo clasifique correctamente a un cliente que cancela por encima de uno que no.
#
# - F1 Score es una métrica que combina precisión y recall en una única puntuación. Es útil cuando las clases están desbalanceadas.
#
#   Un F1 de 0.6163 indica un buen balance entre la detección de cancelaciones y la cantidad de falsos positivos.
#
# - Accuracy (Precisión) es la proporción de predicciones correctas sobre el total de predicciones.
#
#   Un accuracy de 0.7526 da la precisión general del modelo, es decir, qué porcentaje de las predicciones fueron correctas. Por lo que el 75.26% de las predicciones del modelo (clientes que cancelan o no) son correctas.
