import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def analizar_perfiles_musicales(df):

    df_calc = df.copy()

    # Seleccionar variables
    X = df_calc[[
        "tiempo_pop",
        "tiempo_rock",
        "tiempo_reggaeton"
    ]]

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    kmeans = KMeans(
        n_clusters=2,
        random_state=0,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)

    # Agregar clusters
    df_calc["cluster"] = clusters

    etiquetas_cluster = {}

    # Analizar clusters
    for cluster_id in np.unique(clusters):

        grupo = df_calc[df_calc["cluster"] == cluster_id]

        promedios = grupo[[
            "tiempo_pop",
            "tiempo_rock",
            "tiempo_reggaeton"
        ]].mean()

        max_val = promedios.max()
        min_val = promedios.min()

        if max_val - min_val > 20:
            etiquetas_cluster[cluster_id] = "especializado"
        else:
            etiquetas_cluster[cluster_id] = "diverso"

    # Resultado final
    resultado = {}

    for _, row in df_calc.iterrows():
        resultado[int(row["usuario_id"])] = etiquetas_cluster[row["cluster"]]

    return resultado
