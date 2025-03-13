# 🔍 Análisis Predictivo: ¿Estás en la "Friend Zone"?

> Este proyecto realiza un análisis predictivo usando un conjunto de datos ficticio para determinar si alguien está en la "friend zone" basándose en diferentes características.

## 📌 Contenido

-   [Características analizadas](#características-analizadas)
-   [Tecnologías utilizadas](#tecnologías-utilizadas)
-   [Cómo ejecutar el proyecto](#cómo-ejecutar-el-proyecto)
-   [Resultados](#resultados)
-   [Librerías necesarias](#librerías-necesarias)

## 🌟 Características analizadas

Las siguientes son las características que se tomaron en cuenta para este análisis:

1. **Tiempo invertido en mensajes** - Cantidad de tiempo dedicado a enviar mensajes.
2. **Invitaciones rechazadas** - Número de invitaciones rechazadas.
3. **Confianza en compartir secretos** - Nivel de confianza para compartir secretos.
4. **Interacciones sociales** - Número de interacciones sociales.
5. **Friend Zone** - Indicador binario de si está en la friend zone (0: No, 1: Sí).

## 💻 Tecnologías utilizadas

El proyecto fue construido usando las siguientes herramientas y bibliotecas:

-   **Python** - Lenguaje de programación.
-   **pandas** - Biblioteca para el análisis de datos.
-   **matplotlib** - Biblioteca para visualizaciones.
-   **scikit-learn** - Biblioteca para modelado y aprendizaje automático.
-   **Django** - Framework para el desarrollo web.

## 🚀 Cómo ejecutar el proyecto

Sigue estos pasos para ejecutar el proyecto localmente:

1. Asegúrate de tener instaladas las bibliotecas necesarias (`pandas`, `numpy`, `scikit-learn`, `matplotlib`).
2. Clona el repositorio a tu máquina local.
3. Abre tu terminal y navega al directorio del proyecto.
4. Ejecuta los siguientes comandos para aplicar las migraciones y ejecutar el servidor:
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    python manage.py runserver
    ```
5. Abre tu navegador y navega a `http://127.0.0.1:8000/` para ver la aplicación en funcionamiento.

## 📊 Resultados

Al ejecutar el programa, obtendrás un reporte de clasificación que detalla el rendimiento del modelo. Además, se generará una visualización que muestra la importancia de cada característica en la predicción.

## 📦 Librerías necesarias

Asegúrate de tener instaladas las siguientes librerías para ejecutar el proyecto:

-   **pandas**
-   **numpy**
-   **scikit-learn**
-   **matplotlib**
-   **Django**

Puedes instalar todas las librerías necesarias ejecutando el siguiente comando:

```bash
pip install -r requirements.txt
```
