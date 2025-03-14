import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
import pandas as pd
import numpy as np
import os
from friend_zone.data import load_data
from friend_zone.preprocessing import preprocess_data
from friend_zone.model import train_model, evaluate_model
from friend_zone.visualization import plot_feature_importances
import random
import matplotlib.pyplot as plt


def calculate_probability(model, sample):
    """Calcular la probabilidad de estar en la friend zone."""
    probas = model.predict_proba(sample)
    return probas[0][1]


def save_results(results):
    """Guardar los resultados en un archivo."""
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[
                                             ("Text files", "*.txt"), ("All files", "*.*")])
    if file_path:
        with open(file_path, 'w') as file:
            file.write(results)


def run_analysis(user_data, result_text, recommendation_text):
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
        f"La probabilidad de estar en la friend zone según los datos ingresados es: {probability:.2%}\n"
    )
    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, result_msg)
    result_text.config(state=tk.DISABLED)

    # Mostrar recomendaciones
    recommendation_msg = "Recomendaciones:\n"
    if probability > 0.5:
        recommendations = [
            "- Intenta reducir el tiempo invertido en mensajes.\n",
            "- Aumenta las interacciones sociales en persona.\n",
            "- Considera hablar directamente sobre tus sentimientos.\n",
            "- Evita ser demasiado disponible.\n"
        ]
    else:
        recommendations = [
            "- Continúa con tus esfuerzos actuales.\n",
            "- Mantén una comunicación abierta y honesta.\n",
            "- Sigue fortaleciendo la amistad.\n",
            "- Asegúrate de tener intereses comunes.\n"
        ]

    recommendation_msg += random.choice(recommendations)
    recommendation_text.config(state=tk.NORMAL)
    recommendation_text.delete(1.0, tk.END)
    recommendation_text.insert(tk.END, recommendation_msg)
    recommendation_text.config(state=tk.DISABLED)

    messagebox.showinfo("Resultado del Análisis",
                        "Análisis completado con éxito. Revisa los resultados y recomendaciones.")


def main():
    root = tk.Tk()
    root.title("🔍 Análisis Predictivo: Friend Zone")
    root.geometry("700x800")

    style = ttk.Style()
    style.configure("TFrame", background="#f0f0f0")
    style.configure("TLabel", background="#f0f0f0", font=("Helvetica", 12))
    style.configure("TButton", font=("Helvetica", 12), padding=10)
    style.configure("TText", font=("Helvetica", 12))

    frame = ttk.Frame(root, padding="20")
    frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    title_label = ttk.Label(
        frame, text="🔍 Análisis Predictivo: Friend Zone", font=("Helvetica", 16, "bold"))
    title_label.pack(pady=10)

    subtitle_label = ttk.Label(
        frame, text="Ingrese los datos de esta semana para realizar el análisis", font=("Helvetica", 12))
    subtitle_label.pack(pady=5)

    # Crear entradas para cada característica
    labels = [
        "Tiempo invertido en mensajes",
        "Invitaciones rechazadas",
        "Confianza en compartir secretos",
        "Interacciones sociales",
    ]

    entries = {}
    for label_text in labels:
        row = ttk.Frame(frame)
        label = ttk.Label(row, text=label_text, width=30, anchor='w')
        entry = ttk.Entry(row)
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        label.pack(side=tk.LEFT)
        entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries[label_text] = entry

    result_text = tk.Text(frame, height=5, width=50,
                          state=tk.DISABLED, font=("Helvetica", 12))
    result_text.pack(pady=10)

    recommendation_text = tk.Text(
        frame, height=5, width=50, state=tk.DISABLED, font=("Helvetica", 12))
    recommendation_text.pack(pady=10)

    def on_submit():
        try:
            # Leer datos ingresados por el usuario
            user_data = [
                float(entries["Tiempo invertido en mensajes"].get()),
                float(entries["Invitaciones rechazadas"].get()),
                float(entries["Confianza en compartir secretos"].get()),
                float(entries["Interacciones sociales"].get()),
            ]
            run_analysis(user_data, result_text, recommendation_text)
        except ValueError:
            messagebox.showerror(
                "Error", "Por favor, ingresa valores numéricos válidos en todos los campos."
            )

    def on_save():
        results = result_text.get(1.0, tk.END)
        save_results(results)

    # Botón para ejecutar el análisis
    run_button = ttk.Button(frame, text="Ejecutar Análisis", command=on_submit)
    run_button.pack(pady=10)

    # Botón para guardar los resultados
    save_button = ttk.Button(frame, text="Guardar Resultados", command=on_save)
    save_button.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
