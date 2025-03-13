import matplotlib.pyplot as plt


def plot_feature_importances(model, features, output_path='feature_importances.png'):
    importances = model.feature_importances_
    colors = plt.cm.viridis(importances)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(features, importances, color=colors, edgecolor='black')
    plt.xlabel("Importancia", fontsize=15)
    plt.ylabel("Característica", fontsize=15)
    plt.title("Importancia de las Características para predecir Friend Zone",
              fontsize=18, fontweight='bold')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    max_idx = importances.argmax()
    plt.annotate(f'Máxima importancia: {importances[max_idx]:.2f}',
                 xy=(importances[max_idx], max_idx),
                 xytext=(importances[max_idx] + 0.05, max_idx),
                 arrowprops=dict(facecolor='red', arrowstyle='->'),
                 fontsize=12,
                 color='red',
                 ha='center')

    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
