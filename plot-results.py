# plot_results.py

import pandas as pd
import matplotlib.pyplot as plt

# Carregar os resultados dos modelos
results_df = pd.read_csv('model_results.csv')

# Gráfico de barras para as métricas de desempenho
results_df.set_index('Modelo', inplace=True)
results_df.plot(kind='bar', figsize=(10, 6))
plt.title('Comparação de Desempenho dos Modelos')
plt.ylabel('Score')
plt.xlabel('Modelo')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('comparacao_modelos.png')  # Salvar o gráfico como uma imagem
plt.show()

print("Gráfico de comparação de modelos gerado com sucesso!")
