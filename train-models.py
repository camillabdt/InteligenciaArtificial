# train_models.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# Carregar os dados
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

# Preencher valores ausentes com a média
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Inicializar os modelos
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'MLP': MLPClassifier(max_iter=1000, random_state=42, learning_rate_init=0.001, solver='adam'),
    'KNN': KNeighborsClassifier()
}

# Avaliar o desempenho dos modelos
results = {}

for model_name, model in models.items():
    model.fit(X_train_imputed, y_train.values.ravel())
    y_pred = model.predict(X_test_imputed)

    results[model_name] = {
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1-Score': round(f1_score(y_test, y_pred), 4)
    }

# Criar DataFrame com os resultados
results_df = pd.DataFrame(results).T
results_df.index.name = "Modelo"
results_df.to_csv('model_results.csv')

# Criar visualização com destaque
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.axis('off')

highlight_color = "#d1ffd6"
default_color = "#ffffff"
best_model = results_df["Accuracy"].idxmax()

# Cores das células (destaca o melhor)
cell_colors = []
for idx, row in results_df.iterrows():
    color = [highlight_color if idx == best_model else default_color] * len(row)
    cell_colors.append(color)

# Tabela visual
tbl = ax.table(cellText=results_df.values,
               colLabels=results_df.columns,
               rowLabels=results_df.index,
               loc='center',
               cellLoc='center',
               cellColours=cell_colors,
               colColours=["#f1f1f1"] * len(results_df.columns))

tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.3, 1.3)

plt.title("Desempenho dos Modelos de Classificação", fontsize=14)
plt.savefig('tabela_postrain.png', dpi=300, bbox_inches='tight')
plt.show()

print("Modelos treinados e tabela visual gerada com sucesso!")
