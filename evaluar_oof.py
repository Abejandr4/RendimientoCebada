import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv("oof_train.csv")
e = d["oof_pred"] - d["real"]
rmse = np.sqrt((e ** 2).mean()); mae = e.abs().mean()
r2 = 1 - (e ** 2).sum() / ((d["real"] - d["real"].mean()) ** 2).sum()
print(f"n={len(d)} | RMSE {rmse:.3f} | MAE {mae:.3f} | R2 {r2:.3f} | sesgo medio {e.mean():+.3f} t/ha")
print(f"Referencia (predecir la media): RMSE {d['real'].std():.3f}")
print(f"Error relativo: RMSE {rmse / d['real'].mean():.1%} | MAE {mae / d['real'].mean():.1%}")

print("\nPor municipio:")
g = d.assign(err=e, abs_err=e.abs()).groupby("Municipio").agg(
    n=("real", "size"), real_media=("real", "mean"),
    sesgo=("err", "mean"), MAE=("abs_err", "mean"))
print(g.round(2).to_string())

print("\nCorrelación predicho vs. real dentro de cada municipio (n>=10):")
for m, s in d.groupby("Municipio"):
    if len(s) >= 10:
        print(f"  {m:15s} r = {s['real'].corr(s['oof_pred']):+.2f}  (n={len(s)})")

fig, ax = plt.subplots(figsize=(5, 5))
for m, s in d.groupby("Municipio"):
    ax.scatter(s["real"], s["oof_pred"], label=m, s=25, alpha=.8)
lim = [d[["real", "oof_pred"]].min().min() - .1, d[["real", "oof_pred"]].max().max() + .1]
ax.plot(lim, lim, "k--", lw=1)
ax.set(xlabel="Rendimiento real (t/ha)", ylabel="Predicho fuera de muestra (t/ha)",
       title=f"RMSE {rmse:.2f} | R² {r2:.2f}")
ax.legend(fontsize=6); fig.tight_layout(); fig.savefig("observado_vs_predicho.png", dpi=150)
print("\nGuardado: observado_vs_predicho.png")