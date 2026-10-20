import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedKFold
from modelo_lib import Y, prior_mix, nuevo_modelo

# ---------- Datos (mismo armado que prueba_ols.py)
df = pd.read_parquet("df_master.parquet")
S2 = [c for c in df.columns if c.endswith("_s2") and not c.startswith("n_obs")]
fen = pd.read_csv("fenologia_features.csv")
fen["ciclo_sin_senescencia_s2"] = fen["sen50_doy_s2"].isna().astype(int)
FEN = [c for c in fen.columns if c not in ("ID_POLIGONO", "sen50_doy_s2", "dur50_dias_s2")]
df = df.merge(fen, on="ID_POLIGONO", how="left")
GEO = [c for c in ["area_ha_geom", "compacidad", "elong", "elev_mean", "pend_mean"] if c in df.columns]
TODAS = S2 + FEN + GEO

SIM_CLIMA = ["ETo_acum", "ETa_acum", "GDD_antesis", "GDD_llenado", "DAP_antesis_est"]
SIM_AGUA = ["Deficit_Hidrico_Total", "Deficit_ETc", "KS_prom_antesis", "KS_prom_llenado"]
SIM_BIOMASA = ["fPAR_integrado", "Biomasa_Antesis_Sim", "Biomasa_Total_Sim", "HI_penalizado", "s_sim"]
SIM_BANDERAS = ["ciclo_incompleto", "caida_fpar_no_alcanzada"]
BIO_ALL = SIM_CLIMA + SIM_AGUA + SIM_BIOMASA + SIM_BANDERAS

train = df[df["CONJUNTO"] == "ENTRENAMIENTO"].reset_index(drop=True)
coords = train[["x_m", "y_m"]].to_numpy()
splits = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=0).split(coords))

def grupos_modelo(cols):
    por = lambda pref: [c for c in cols if c.startswith(pref)]
    g = {
        "NDVI (nivel, pico, AUC, meses)": por("ndvi_"),
        "EVI": por("evi_"), "LAI": por("lai_"), "MSI": por("msi_"), "NDWI": por("ndwi_"),
        "Fenología (inicio, subida, días altos)": [c for c in ["gu25_doy_s2", "gu50_doy_s2", "pend_subida_s2",
                                                              "dias_sobre90_s2", "ciclo_sin_senescencia_s2"] if c in cols],
        "Geometría de la parcela": [c for c in ["area_ha_geom", "compacidad", "elong"] if c in cols],
        "Topografía (elevación, pendiente)": [c for c in ["elev_mean", "pend_mean"] if c in cols],
        "Simulador: clima y temperatura": [c for c in SIM_CLIMA if c in cols],
        "Simulador: agua y estrés": [c for c in SIM_AGUA if c in cols],
        "Simulador: biomasa y rendimiento": [c for c in SIM_BIOMASA if c in cols],
        "Simulador: banderas": [c for c in SIM_BANDERAS if c in cols],
    }
    return {k: v for k, v in g.items() if v}

def importancia(cols, conjuntos, n_perm=5, seed=0):
    """Aumento del RMSE al permutar (romper la relación de) cada conjunto de columnas en el fold de prueba."""
    rng = np.random.default_rng(seed)
    base, n = np.zeros(3), np.zeros(3)
    perm = {k: np.zeros(3) for k in conjuntos}
    for i, (tr, te) in enumerate(splits):
        r = i // 5                                          # repetición 0, 1 o 2
        dtr, dte = train.iloc[tr], train.iloc[te]
        p_tr, p_te = prior_mix(dtr, dte)
        m = nuevo_modelo().fit(dtr[cols], dtr[Y].to_numpy() - p_tr)
        yte = dte[Y].to_numpy()
        base[r] += ((yte - (p_te + m.predict(dte[cols]))) ** 2).sum(); n[r] += len(te)
        for k, cs in conjuntos.items():
            se = 0.0
            for _ in range(n_perm):
                X = dte[cols].copy()
                X[cs] = X[cs].to_numpy()[rng.permutation(len(X))]   # permuta las filas del grupo juntas
                se += ((yte - (p_te + m.predict(X))) ** 2).sum()
            perm[k][r] += se / n_perm
    rb = np.sqrt(base / n)
    d = pd.DataFrame({k: np.sqrt(perm[k] / n) - rb for k in conjuntos}).T
    return pd.DataFrame({"dRMSE_medio": d.mean(axis=1), "min_rep": d.min(axis=1), "max_rep": d.max(axis=1)}), rb.mean()

def mostrar(titulo, tabla, rmse_base, top=None):
    t = tabla.sort_values("dRMSE_medio", ascending=False)
    print(f"\n===== {titulo} (RMSE base {rmse_base:.4f}) =====")
    print((t.head(top) if top else t).round(4).to_string())

for nombre, cols in {"MODELO FINAL (34 variables)": TODAS,
                     "MODELO AMPLIADO (34 + simulador)": TODAS + BIO_ALL}.items():
    print(f"\n######## {nombre} ########", flush=True)
    t_g, rb = importancia(cols, grupos_modelo(cols))
    mostrar("Importancia por GRUPO", t_g, rb)
    t_v, rb = importancia(cols, {c: [c] for c in cols})
    mostrar("Importancia por VARIABLE", t_v, rb, top=15)

print("\nLectura: dRMSE > 0 significa que permutar la variable empeora el error (el modelo la usa); "
      "cerca de 0 o negativo, que no aporta. min_rep y max_rep son las tres repeticiones del CV: "
      "si el rango incluye 0, la importancia no se distingue de cero. "
      "Con alpha = 1000 se esperan valores muy pequeños (menos de 0.005 t/ha).")