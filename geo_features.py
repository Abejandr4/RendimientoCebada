import numpy as np, pandas as pd, geopandas as gpd
import rasterio
from rasterstats import zonal_stats, point_query
from scipy.spatial import cKDTree

RUTA = "DataSet/DATASET_RETO_AGRO_2026/"
SHP = RUTA + "Parcelas_Reto_AGC_CONJUNTO_70_30/Parcelas_Reto_AGC_CONJUNTO.shp"
ELEV = RUTA + "Reto_AgroCebada_Topografia_INEGI_CEM4/Reto_AgroCebada_Topografia_INEGI_CEM4/Reto_AgroCebada_Topografia_INEGI_CEM4/Elevacion_INEGI_CEM4_120m.tif"           # AJUSTA
PEND = RUTA + "Reto_AgroCebada_Topografia_INEGI_CEM4/Reto_AgroCebada_Topografia_INEGI_CEM4/Reto_AgroCebada_Topografia_INEGI_CEM4/Pendiente_INEGI_CEM4_120m_grados.tif"    # AJUSTA
ID_COL = "ID_POLIGON"                                        # AJUSTA tras ver el print de columnas
CRS_METRICO = 6372   # ITRF2008 / LCC de INEGI, buena para áreas en México

# ---------- 1. Carga y diagnóstico
gdf = gpd.read_file(SHP)
print("CRS:", gdf.crs)
print("Columnas:", gdf.columns.tolist())
print(gdf.drop(columns="geometry").head(3).to_string())
print(gdf.geom_type.value_counts().to_string())
assert ID_COL in gdf.columns, f"Cambia ID_COL; columnas: {gdf.columns.tolist()}"

# ---------- 2. Limpieza de geometrías
print("\nInválidas:", (~gdf.is_valid).sum(), "| vacías:", gdf.is_empty.sum(),
      "| IDs duplicados:", gdf[ID_COL].duplicated().sum())
gdf["geometry"] = gdf.geometry.make_valid()          # repara auto-intersecciones
gdf = gdf.explode(index_parts=False)                 # separa multipartes
gdf["_a"] = gdf.geometry.area
gdf = (gdf.sort_values("_a", ascending=False)        # conserva la parte más grande por ID
          .drop_duplicates(ID_COL).drop(columns="_a").sort_values(ID_COL).reset_index(drop=True))
gdf = gdf[gdf.geom_type == "Polygon"].reset_index(drop=True)
assert len(gdf) == 197, f"Se esperaban 197 polígonos y hay {len(gdf)}"

# ---------- 3. Features de forma y tamaño (en CRS métrico)
gm = gdf.to_crs(CRS_METRICO)
def elongacion(geom):
    r = geom.minimum_rotated_rectangle
    x, y = r.exterior.coords.xy
    l1, l2 = np.hypot(x[1]-x[0], y[1]-y[0]), np.hypot(x[2]-x[1], y[2]-y[1])
    return max(l1, l2) / max(min(l1, l2), 1e-9)

geo = pd.DataFrame({"ID_POLIGONO": gdf[ID_COL].values})

geo["Estado"] = gdf["Estado"].values
geo["Municipio"] = gdf["Municipio"].values
geo["area_ha_shp"] = gdf["área_ha"].values


geo["area_ha_geom"] = gm.area.values / 1e4
geo["perim_m"] = gm.length.values
geo["compacidad"] = 4 * np.pi * gm.area.values / gm.length.values**2   # 1 = círculo
geo["elong"] = gm.geometry.apply(elongacion).values
geo["n_vert"] = gm.geometry.apply(lambda g: len(g.exterior.coords)).values
c = gm.centroid
geo["x_m"], geo["y_m"] = c.x.values, c.y.values                         # SOLO para grupos de CV
c4 = c.to_crs(4326)
geo["lon"], geo["lat"] = c4.x.values, c4.y.values                       # SOLO para grupos / extraer clima

# ---------- 4. Topografía (zonal stats)
def zonal(path, pref):
    with rasterio.open(path) as src:
        crs, nodata = src.crs, src.nodata
    g = gdf.to_crs(crs)
    z = pd.DataFrame(zonal_stats(g, path, stats=["mean", "std", "min", "max", "count"],
                                 nodata=nodata, all_touched=True)).add_prefix(pref + "_")
    # respaldo: parcelas sin píxel válido -> valor en el centroide
    falta = z[pref + "_mean"].isna()
    if falta.any():
        pts = list(g.centroid[falta])
        z.loc[falta, pref + "_mean"] = point_query(pts, path)
    print(f"{pref}: parcelas sin dato = {z[pref + '_mean'].isna().sum()}, "
          f"píxeles/parcela mediana = {z[pref + '_count'].median():.0f}")
    return z

geo = pd.concat([geo, zonal(ELEV, "elev"), zonal(PEND, "pend")], axis=1)
geo["elev_rango"] = geo["elev_max"] - geo["elev_min"]

geo.to_csv("geo_features.csv", index=False)

# ---------- 5. Diagnósticos
print("\n", geo.drop(columns=["ID_POLIGONO"]).describe().T[["mean", "std", "min", "max"]].round(2).to_string())

par = pd.read_csv(RUTA + "ID_area_rendimiento_70_30_Reto_AgroCebada.csv")
print("\nColumnas del CSV de IDs:", par.columns.tolist())
m = par.merge(geo, on="ID_POLIGONO", how="left")
for col in par.columns:
    if "area" in col.lower() or "ha" == col.lower():
        print(f"corr({col}, area_ha_geom) =", round(m[col].corr(m["area_ha_geom"]), 3))

# ¿El split 70/30 es espacial? distancia al vecino de entrenamiento más cercano
tr = m[m.CONJUNTO == "ENTRENAMIENTO"][["x_m", "y_m"]].to_numpy()
te = m[m.CONJUNTO == "PREDICCION"][["x_m", "y_m"]].to_numpy()
d_te = cKDTree(tr).query(te, k=1)[0] / 1000
d_tr = cKDTree(tr).query(tr, k=2)[0][:, 1] / 1000
print(f"\nVecino de entrenamiento más cercano (km): PREDICCION mediana {np.median(d_te):.1f}"
      f" | ENTRENAMIENTO mediana {np.median(d_tr):.1f}")
print("geo_features.csv guardado:", geo.shape)

# ---------- 6. Diagnósticos de Estado/Municipio y chequeo de área
print("\nCorr(área_ha del shapefile, área calculada) =",
      round(geo["area_ha_shp"].corr(geo["area_ha_geom"]), 3))
print("Cultivo:", gdf["Cultivo"].value_counts().to_dict())

m = par.merge(geo[["ID_POLIGONO", "Estado", "Municipio"]], on="ID_POLIGONO", how="left")
print("\nParcelas por Estado:\n", m["Estado"].value_counts().to_string())
print("\nMunicipios distintos:", m["Municipio"].nunique())
print(m["Municipio"].value_counts().head(10).to_string())

if "RENDIMIENTO_T_HA" in m.columns:
    t = m[m.CONJUNTO == "ENTRENAMIENTO"]
    print("\nRendimiento por Estado (entrenamiento):")
    print(t.groupby("Estado")["RENDIMIENTO_T_HA"].agg(["count", "mean", "std"]).round(2).to_string())
    print("\nRendimiento por Municipio (los 10 con más parcelas):")
    g = t.groupby("Municipio")["RENDIMIENTO_T_HA"].agg(["count", "mean", "std"]).round(2)
    print(g.sort_values("count", ascending=False).head(10).to_string())

# ---------- 7. Chequeos adicionales
print("\nCultivo vs Estado:\n", pd.crosstab(gdf["Cultivo"], gdf["Estado"]).to_string())
print("\nCultivo vs CONJUNTO:\n", pd.crosstab(gdf["Cultivo"], gdf["CONJUNTO"]).to_string())
print("\nParcelas pequeñas o muy alargadas:")
print(geo.loc[(geo.elong > 8) | (geo.area_ha_geom < 0.5),
              ["ID_POLIGONO", "area_ha_geom", "elong", "compacidad"]].round(2).to_string(index=False))