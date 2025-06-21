import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Recarregar os dados
df = pd.read_csv('model_results.csv')

# Converter valores para porcentagem
df_percent = df.copy()
for col in df.columns[1:]:
    df_percent[col] = df[col] * 100

# Plotar tabela com destaque visual
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('off')

# Criar tabela com colormap
table = ax.table(
    cellText=df_percent.round(2).values,
    colLabels=df_percent.columns,
    cellLoc='center',
    loc='center',
    colColours=sns.color_palette("Set2", n_colors=len(df_percent.columns))
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.title("Desempenho dos Classificadores (%)", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig("tabelanova.png", dpi=300)
plt.show()
