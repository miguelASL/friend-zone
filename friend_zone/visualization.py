import matplotlib.pyplot as plt


def plot_feature_importances(model, features):
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
                 xy=(importances[max_idx], features[max_idx]),
                 xytext=(importances[max_idx]-0.2, max_idx+0.5),
                 arrowprops=dict(facecolor='red', arrowstyle='->'),
                 fontsize=12,
                 color='red',
                 ha='center')

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
