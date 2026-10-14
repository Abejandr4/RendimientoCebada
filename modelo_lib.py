import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Y = "RENDIMIENTO_T_HA"
K_SHRINK, K_VEC = 5, 5


def prior_mun(dtr, dte):
    """Media por municipio suavizada hacia la media global. Excluye la propia parcela en dtr."""
    ytr = dtr[Y].to_numpy()
    mtr, mte = dtr["Municipio"].to_numpy(), dte["Municipio"].to_numpy()
    g = ytr.mean()
    suma = pd.Series(ytr).groupby(mtr).sum()
    cnt = pd.Series(ytr).groupby(mtr).count()
    p_te = (pd.Series(mte).map(suma).fillna(0).to_numpy() + K_SHRINK * g) / \
           (pd.Series(mte).map(cnt).fillna(0).to_numpy() + K_SHRINK)
    s_i, n_i = pd.Series(mtr).map(suma).to_numpy(), pd.Series(mtr).map(cnt).to_numpy()
    p_tr = (s_i - ytr + K_SHRINK * g) / (n_i - 1 + K_SHRINK)
    return p_tr, p_te


def prior_vec(dtr, dte):
    """Media de las K_VEC parcelas de entrenamiento más cercanas."""
    ctr, cte = dtr[["x_m", "y_m"]].to_numpy(), dte[["x_m", "y_m"]].to_numpy()
    ytr = dtr[Y].to_numpy()
    tree = cKDTree(ctr)
    p_te = ytr[tree.query(cte, k=K_VEC)[1]].mean(axis=1)
    p_tr = ytr[tree.query(ctr, k=K_VEC + 1)[1][:, 1:]].mean(axis=1)   # sin la propia parcela
    return p_tr, p_te


def prior_mix(dtr, dte):
    a, b = prior_mun(dtr, dte), prior_vec(dtr, dte)
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def nuevo_modelo():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         RidgeCV(alphas=np.logspace(-1, 3, 25)))


def ajustar(dtr, cols):
    """Entrena la corrección (ridge) sobre el residuo y - prior."""
    p_tr, _ = prior_mix(dtr, dtr.iloc[:1])
    return nuevo_modelo().fit(dtr[cols], dtr[Y].to_numpy() - p_tr)


def predecir(dtr, dte, cols):
    """Devuelve (predicción final, prior, corrección aprendida)."""
    p_tr, p_te = prior_mix(dtr, dte)
    m = nuevo_modelo().fit(dtr[cols], dtr[Y].to_numpy() - p_tr)
    corr = m.predict(dte[cols])
    return p_te + corr, p_te, corr