import pandas as pd
import numpy as np


def eliminar_outliers(df, columnas):

    mascara_outlier = pd.Series(False, index=df.index)

    reporte_filas = []

    for col in columnas:

        serie = df[col].dropna()

        q1, q3 = np.percentile(serie, [25, 75])

        iqr = q3 - q1

        lim_inf = q1 - 1.5 * iqr
        lim_sup = q3 + 1.5 * iqr

        es_outlier = (
            (df[col] < lim_inf) |
            (df[col] > lim_sup)
        )

        mascara_outlier |= es_outlier

        reporte_filas.append({
            'columna': col,
            'q1': round(q1, 4),
            'q3': round(q3, 4),
            'iqr': round(iqr, 4),
            'limite_inf': round(lim_inf, 4),
            'limite_sup': round(lim_sup, 4),
            'outliers_eliminados': int(es_outlier.sum()),
        })

    df_limpio = df[~mascara_outlier].reset_index(drop=True)

    reporte = pd.DataFrame(reporte_filas).set_index('columna')

    return df_limpio, reporte
