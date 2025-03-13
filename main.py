import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import os
from friend_zone.data import load_data
from friend_zone.preprocessing import preprocess_data
from friend_zone.model import train_model, evaluate_model
from friend_zone.visualization import plot_feature_importances
import matplotlib.pyplot as plt


def calculate_probability(model, sample):
    """Calcular la probabilidad de estar en la friend zone."""
    probas = model.predict_proba(sample)
    return probas[0][1]


def run_analysis(user_data, frame):
    # Cargar datos
    df = load_data()

    # Preprocesar datos
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)

    # Entrenar el modelo
    model = train_model(X_train, y_train, feature_names)

    # Evaluar el modelo
    evaluate_model(model, X_test, y_test)

    # Visualizar importancias de las características
    output_path = 'feature_importances.png'
    plot_feature_importances(model, feature_names, output_path)

    # Mostrar la gráfica
    img = plt.imread(output_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Preparar los datos del usuario
    sample = pd.DataFrame([user_data], columns=feature_names)

    # Calcular probabilidad
    probability = calculate_probability(model, sample)

    # Mostrar resultados
    result_msg = (
        f"La probabilidad de estar en la friend zone según los datos ingresados es: {probability:.2%}"
    )
    messagebox.showinfo("Resultado del Análisis", result_msg)


def main():
    root = tk.Tk()
    root.title("🔍 Análisis Predictivo: Friend Zone")

    frame = tk.Frame(root, padx=20, pady=20)
    frame.pack(padx=10, pady=10)

    # Crear entradas para cada característica
    labels = [
        "Tiempo invertido en mensajes",
        "Invitaciones rechazadas",
        "Confianza en compartir secretos",
        "Interacciones sociales",
    ]

    entries = {}
    for label_text in labels:
        row = tk.Frame(frame)
        label = tk.Label(row, text=label_text, width=30, anchor='w')
        entry = tk.Entry(row)
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        label.pack(side=tk.LEFT)
        entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries[label_text] = entry

    def on_submit():
        try:
            # Leer datos ingresados por el usuario
            user_data = [
                float(entries["Tiempo invertido en mensajes"].get()),
                float(entries["Invitaciones rechazadas"].get()),
                float(entries["Confianza en compartir secretos"].get()),
                float(entries["Interacciones sociales"].get()),
            ]
            run_analysis(user_data, frame)
        except ValueError:
            messagebox.showerror(
                "Error", "Por favor, ingresa valores numéricos válidos en todos los campos."
            )

    # Botón para ejecutar el análisis
    run_button = tk.Button(frame, text="Ejecutar Análisis", command=on_submit)
    run_button.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
