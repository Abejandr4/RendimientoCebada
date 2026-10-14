import pandas as pd
raw = pd.read_csv("DataSet/DATASET_RETO_AGRO_2026/Conjunto_datos_BASICO_AgroCebada2026.csv", parse_dates=["fecha_captura"])
print(raw[raw.fecha_captura.dt.year == 2025].groupby("sensor")["fecha_captura"].agg(["min", "max"]))

ids = pd.read_csv("DataSet/DATASET_RETO_AGRO_2026/ID_area_rendimiento_70_30_Reto_AgroCebada.csv")
print(ids.groupby("CONJUNTO")["RENDIMIENTO_T_HA"].agg(["count", "size"]))   # PREDICCION debe dar count=0