# save_final_table.py

import pandas as pd
import matplotlib.pyplot as plt

# Carregar os resultados finais de acurácia
results_df = pd.read_csv('model_results.csv')

# Resultados finais para comparar os modelos
resultados = {
    "Modelo": results_df['Modelo'].tolist(),
    "Acurácia final (%)": [round(100 * acc, 1) for acc in results_df['Acurácia']]
}

df_resultados = pd.DataFrame(resultados)

# Exibir tabela com acurácias finais
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('off')
tbl = ax.table(cellText=df_resultados.values, colLabels=df_resultados.columns, loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
plt.title("Acurácias Finais dos Modelos", pad=20)
plt.savefig("tabela_acuracias_finais.png", dpi=300, bbox_inches='tight')  # Salvar a tabela como imagem
plt.show()

print("Tabela de acurácias finais gerada com sucesso!")
