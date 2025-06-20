import pandas as pd
from sklearn.model_selection import train_test_split

# Carregar o dataset
df = pd.read_csv('Fakenews-dataset-final.csv')

# Selecionar as features e o target
X = df.drop('Classe', axis=1)  # 'Classe' é a coluna de rótulos (verdadeira/falsa)
y = df['Classe']

# Divisão do dataset em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Dados carregados e salvos com sucesso!")
