import pandas as pd
import numpy as np

RUTA = "DataSet/DATASET_RETO_AGRO_2026/"
INICIO, FIN = pd.Timestamp("2025-04-01"), pd.Timestamp("2025-10-31")
NUBES = 40


raw = pd.read_csv(RUTA + "Conjunto_datos_BASICO_AgroCebada2026.csv", parse_dates=["fecha_captura"])
ids = pd.read_csv(RUTA + "ID_area_rendimiento_70_30_Reto_AgroCebada.csv")[["ID_POLIGONO"]]

base = raw[(raw.fecha_captura >= INICIO) & (raw.fecha_captura < FIN + pd.Timedelta(days=1))]
base = base[base.porcentaje_nubosidad.fillna(0) <= NUBES]
s2 = base[base.sensor == "Sentinel-2"].copy()
l8 = base[base.sensor == "Landsat"].copy()

# ---- DIAGNÓSTICO 1: ¿qué columnas trae cada sensor?
cand = [c for c in raw.columns if c.endswith("_promedio")]
disp = pd.DataFrame({"s2": s2[cand].notna().mean(), "l8": l8[cand].notna().mean()}).round(2)
print("\n=== Fracción de datos no vacíos por sensor ===")
print(disp.to_string())
print("Filas tras filtros -> S2:", len(s2), "| Landsat:", len(l8))

# Filtro de rango válido SOLO para S2 (Landsat no tiene ndvi_promedio)
s2 = s2[s2.ndvi_promedio.between(-1, 1)]

pref = ["ndvi_promedio", "evi_promedio", "savi_promedio", "lai_promedio", "fapar_promedio",
        "ndwi_promedio", "msi_promedio", "nddi_promedio", "crc_promedio", "ndti_promedio"]
idx_s2 = [c for c in pref if disp.loc[c, "s2"] > 0.9]
idx_l8 = [c for c in pref + ["vi6t_promedio"] if disp.loc[c, "l8"] > 0.5]
print("\nÍndices S2:", idx_s2)
print("Índices Landsat:", idx_l8)

# ---- Sentinel-2: serie regular + features
def serie_regular(df_s, cols, paso="5D", max_hueco=30):
    grid = pd.date_range(INICIO, FIN, freq=paso)
    def f(g):
        g = g.set_index("fecha_captura")[cols].groupby(level=0).mean()
        obs = g.index
        out = (g.reindex(g.index.union(grid))
                 .interpolate(method="time", limit_area="inside").reindex(grid))
        dist = np.array([np.abs((obs - t).days).min() for t in grid])
        out[dist > max_hueco] = np.nan
        out = out.rolling(3, center=True, min_periods=1).median()
        out["n_obs"] = len(obs)
        return out
    return (df_s.groupby("ID_POLIGONO").apply(f)
                .rename_axis(["ID_POLIGONO", "fecha"]).reset_index())

def auc(g, col):
    s = g.dropna(subset=[col])
    if len(s) < 2:
        return np.nan
    dias = (s["fecha"] - s["fecha"].iloc[0]).dt.days.to_numpy()
    y = s[col].to_numpy()
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(dias)))

def doy_pico(g, col):
    s = g.dropna(subset=[col])
    return s.loc[s[col].idxmax(), "fecha"].dayofyear if len(s) else np.nan

ser_s2 = serie_regular(s2, idx_s2)
ser_s2.to_csv("serie_s2.csv", index=False)          # para el dashboard

cols_f = [c for c in ["ndvi_promedio", "evi_promedio", "lai_promedio", "msi_promedio", "ndwi_promedio"]
          if c in idx_s2]
filas = []
for pid, g in ser_s2.groupby("ID_POLIGONO"):
    f = {"ID_POLIGONO": pid, "n_obs_s2": g["n_obs"].iloc[0]}
    for c in cols_f:
        n = c.replace("_promedio", "")
        f[f"{n}_p95_s2"] = g[c].quantile(.95)
        f[f"{n}_p05_s2"] = g[c].quantile(.05)
        f[f"{n}_mean_s2"] = g[c].mean()
    f["ndvi_doy_pico_s2"] = doy_pico(g, "ndvi_promedio")
    f["ndvi_auc_s2"] = auc(g, "ndvi_promedio")
    filas.append(f)
f_s2 = pd.DataFrame(filas)
intra = s2.groupby("ID_POLIGONO")["ndvi_std"].mean().rename("ndvi_std_intra_s2").reset_index()
f_s2 = f_s2.merge(intra, on="ID_POLIGONO", how="left")

# ---- Landsat: SIN interpolar, se agregan las observaciones reales
l8[["ID_POLIGONO", "fecha_captura"] + idx_l8].to_csv("obs_l8.csv", index=False)
g = l8.groupby("ID_POLIGONO")
f_l8 = pd.DataFrame({"n_obs_l8": g.size()})
for c in idx_l8:
    n = c.replace("_promedio", "")
    f_l8[f"{n}_p95_l8"] = g[c].quantile(.95)
    f_l8[f"{n}_mean_l8"] = g[c].mean()
f_l8 = f_l8.reset_index()

# ---- Unión (las 197 parcelas siempre quedan)
feats = ids.merge(f_s2, on="ID_POLIGONO", how="left").merge(f_l8, on="ID_POLIGONO", how="left")
feats.to_csv("features_satelitales.csv", index=False)

# ---- DIAGNÓSTICO 2
print("\n=== f_s2:", f_s2.shape, "| f_l8:", f_l8.shape, "| feats:", feats.shape, "(debe tener 197 filas)")
print("\nn_obs_s2:\n", feats["n_obs_s2"].describe().round(1).to_string())
print("\nn_obs_l8:\n", feats["n_obs_l8"].describe().round(1).to_string())
print("\nColumnas con más NaN:\n", feats.isna().mean().sort_values(ascending=False).head(8).round(2).to_string())