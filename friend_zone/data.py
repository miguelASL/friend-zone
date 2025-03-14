import pandas as pd
import numpy as np


def load_data():
    """Cargar y preparar el DataFrame con características específicas."""
    data = {
        "Tiempo invertido en mensajes en minutos": np.random.randint(1, 100, 100),
        "Invitaciones rechazadas": np.random.randint(0, 10, 100),
        "Confianza en compartir secretos": np.random.uniform(0, 1, 100),
        "Interacciones sociales": np.random.randint(1, 50, 100),
        "Friend Zone": np.random.choice([0, 1], 100, p=[0.7, 0.3]),
    }
    df = pd.DataFrame(data)
    return df
