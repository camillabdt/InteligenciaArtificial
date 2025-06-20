# train_models.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Carregar os dados
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

# Verificar se há valores NaN nas features e labels
print("Valores NaN nas features de treino:")
print(X_train.isna().sum())

print("Valores NaN nas labels de treino:")
print(y_train.isna().sum())

# Preencher valores NaN nas features com a média
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Verificar se ainda há NaN após o preenchimento
print("Valores NaN nas features de treino após imputação:")
print(pd.DataFrame(X_train_imputed).isna().sum())

# Inicializar os modelos
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'MLP': MLPClassifier(max_iter=1000, random_state=42, learning_rate_init=0.001, solver='adam'),  # Ajustes no MLP
    'KNN': KNeighborsClassifier()
}

# Avaliar o desempenho de cada modelo
results = {}

for model_name, model in models.items():
    model.fit(X_train_imputed, y_train.values.ravel())  # Usando ravel() para garantir que y seja 1D
    y_pred = model.predict(X_test_imputed)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

# Criar um DataFrame para armazenar os resultados
results_df = pd.DataFrame(results).T

# Salvar os resultados em um arquivo CSV
results_df.to_csv('model_results.csv', index=True)

# Exibir a tabela de forma bonitinha com Matplotlib
fig, ax = plt.subplots(figsize=(8, 4))  # Tamanho da figura
ax.axis('off')  # Desabilitar os eixos

# Exibir a tabela no gráfico
tbl = ax.table(cellText=results_df.values,
               colLabels=results_df.columns,
               rowLabels=results_df.index,
               loc='center',
               cellLoc='center',  # Alinhar o texto das células ao centro
               colColours=["#f1f1f1"] * results_df.shape[1],  # Cor das colunas
               cellColours=[["#ffffff"] * results_df.shape[1]] * results_df.shape[0])  # Cor das células

# Melhorar o estilo da tabela
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.5, 1.5)  # Aumentar o tamanho da tabela

# Salvar a tabela gerada
plt.title("Desempenho dos Modelos", fontsize=14)
plt.savefig('tabela_resultados_bonita.png', dpi=300, bbox_inches='tight')
plt.show()

print("Modelos treinados e tabela de resultados gerada com sucesso!")
