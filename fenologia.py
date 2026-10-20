import numpy as np, pandas as pd

serie = pd.read_csv("serie_s2.csv", parse_dates=["fecha"])

filas = []
for pid, g in serie.groupby("ID_POLIGONO"):
    g = g.sort_values("fecha")
    fila = {"ID_POLIGONO": pid}
    ok = g.dropna(subset=["ndvi_promedio"])
    if len(ok) >= 5:
        v = ok["ndvi_promedio"].to_numpy()
        doy = ok["fecha"].dt.dayofyear.to_numpy()
        ip = int(np.argmax(v))
        pico, base = v[ip], np.quantile(v, 0.05)
        amp = pico - base
        th25, th50, th90 = base + .25 * amp, base + .50 * amp, base + .90 * amp
        sube25 = np.where(v[:ip + 1] >= th25)[0]
        sube50 = np.where(v[:ip + 1] >= th50)[0]
        baja50 = np.where(v[ip:] <= th50)[0]
        gu25 = doy[sube25[0]] if len(sube25) else np.nan        # inicio del brote
        gu50 = doy[sube50[0]] if len(sube50) else np.nan        # mitad de la subida
        sen50 = doy[ip + baja50[0]] if len(baja50) else np.nan  # mitad de la senescencia
        fila.update(
            ndvi_amp=amp, gu25_doy=gu25, gu50_doy=gu50, sen50_doy=sen50,
            dur50_dias=sen50 - gu50,
            pend_subida=(pico - th25) / max(doy[ip] - gu25, 5),
            dias_sobre90=int((g["ndvi_promedio"] >= th90).sum()) * 5,   # malla de 5 días
        )
    for m in range(5, 10):                                       # NDVI medio mayo a septiembre
        fila[f"ndvi_mes{m}"] = g.loc[g["fecha"].dt.month == m, "ndvi_promedio"].mean()
    filas.append(fila)

feno = pd.DataFrame(filas)
feno = feno.rename(columns={c: c + "_s2" for c in feno.columns if c != "ID_POLIGONO"})
feno.to_csv("fenologia_features.csv", index=False)

# Diagnósticos
print("Forma:", feno.shape, "(deberia tener 197 filas)")
print("\nNaN por columna:\n", feno.isna().mean().round(2).to_string())
print("\n", feno.drop(columns="ID_POLIGONO").describe().T[["mean", "std", "min", "max"]].round(1).to_string())

df = pd.read_parquet("df_master.parquet")
tr = df[df.CONJUNTO == "ENTRENAMIENTO"].merge(feno, on="ID_POLIGONO")
res = tr["RENDIMIENTO_T_HA"] - tr.groupby("Municipio")["RENDIMIENTO_T_HA"].transform("mean")
rho = {}
for c in feno.columns.drop("ID_POLIGONO"):
    x = tr[c] - tr.groupby("Municipio")[c].transform("mean")
    rho[c] = x.corr(res, method="spearman")
rho = pd.Series(rho).dropna()
print("\nSpearman dentro del municipio (exploratorio):")
print(rho.reindex(rho.abs().sort_values(ascending=False).index).round(2).to_string())