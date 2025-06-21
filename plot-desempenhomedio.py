import pandas as pd
import matplotlib.pyplot as plt

# Resultados médios dos modelos (exemplo, substitua pelos seus valores reais)
media_resultados = {
    'Random Forest': 0.985,
    'XGBoost': 0.975,
    'SVM': 0.96,
    'MLP': 0.961,
    'KNN': 0.948
}

# Dicionário de cores para cada modelo
cores = {
    'Random Forest': '#1f77b4',  # azul
    'XGBoost': '#ff7f0e',        # laranja
    'SVM': '#2ca02c',            # verde
    'MLP': '#d62728',            # vermelho
    'KNN': '#9467bd'             # roxo
}

# Converter para DataFrame
df = pd.DataFrame.from_dict(media_resultados, orient='index', columns=['Média'])
df = df.sort_values(by='Média', ascending=True)

# Criar gráfico
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(df.index, df['Média'], color=[cores[modelo] for modelo in df.index])

# Título e eixos
plt.title('Desempenho Médio dos Modelos de Classificação')
plt.xlabel('Média das Métricas (Accuracy, Precision, Recall, F1-Score)')

# Adicionar legenda personalizada
from matplotlib.patches import Patch
legenda = [Patch(facecolor=cor, label=modelo) for modelo, cor in cores.items()]
ax.legend(handles=legenda, title='Classificadores')

plt.tight_layout()
plt.savefig('grafico_media_modelos_com_legenda.png', dpi=300)
plt.show()
