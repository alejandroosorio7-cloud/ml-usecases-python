from sklearn.cluster import KMeans

def agrupar_pacientes(df, n_clusters):

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10
    )

    labels = kmeans.fit_predict(df)

    return labels
