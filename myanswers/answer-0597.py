import pandas as pd

def detectar_olas_calor(df, umbral):
    df = df.copy()

    df['fecha'] = pd.to_datetime(df['fecha'])

    df['dia_semana'] = df['fecha'].dt.day_name()

    df = df[df['temperatura'] > umbral]

    return df
