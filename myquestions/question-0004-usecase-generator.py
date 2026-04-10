import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generar_caso_de_uso_aplicar_pca():
    n = random.randint(20, 50)
    m = random.randint(3, 6)

    df = pd.DataFrame(np.random.randn(n, m), columns=[f"x{i}" for i in range(m)])

    n_componentes = random.randint(2, m)

    input_data = {"df": df.copy(), "n_componentes": n_componentes}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=n_componentes)
    X_pca = pca.fit_transform(X_scaled)

    return input_data, (X_pca, pca.explained_variance_ratio_)
