import pandas as pd
import matplotlib.pyplot as plt

results_df = pd.read_csv('model_results.csv')

# Verificar e ajustar nome da coluna
if 'Modelo' not in results_df.columns:
    print("Colunas disponíveis:", results_df.columns)
    raise ValueError("A coluna 'Modelo' não foi encontrada. Verifique o nome correto.")

results_df.set_index('Modelo', inplace=True)
results_df.plot(kind='bar', figsize=(10, 6))
plt.title('Comparação de Desempenho dos Modelos')
plt.ylabel('Score')
plt.xlabel('Modelo')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('comparacao_modelos.png')
plt.show()
