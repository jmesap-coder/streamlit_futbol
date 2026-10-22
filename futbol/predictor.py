"""Carga los modelos entrenados en el notebook y hace las predicciones Monte Carlo."""
import json
import os

import joblib
import numpy as np
import pandas as pd

from .features import FEATURES, LABELS, construir_features
from .redes import RedesNumpy

N_GRUPOS_RF = 20   # los arboles del RF se agrupan en 20 sub-bosques para suavizar probabilidades

LIGAS = {
    'SP1': 'La Liga',
    'E0': 'Premier League',
    'I1': 'Serie A',
    'D1': 'Bundesliga',
    'F1': 'Ligue 1',
}


def sortear(P, rng):
    """P: (..., 3) probabilidades -> resultado 0/1/2 sorteado por inversion de la acumulada."""
    u = rng.random(P.shape[:-1])[..., None]
    return np.minimum((u > np.cumsum(P, axis=-1)).sum(axis=-1), 2)


class Predictor:
    def __init__(self, carpeta="modelos"):
        with open(os.path.join(carpeta, "config.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        self.w_rf = float(self.config["W_RF"])
        self.scaler = joblib.load(os.path.join(carpeta, "scaler.joblib"))
        self.rf = joblib.load(os.path.join(carpeta, "rf.joblib"))
        with open(os.path.join(carpeta, "elos.json"), encoding="utf-8") as f:
            self.elos = json.load(f)
        df = pd.read_csv(os.path.join(carpeta, "partidos.csv.gz"), parse_dates=["date"])
        self.df_hist = df.sort_values("date").reset_index(drop=True)

        # Las redes son opcionales: si no estan, la app funciona solo con el Random Forest
        ruta_redes = os.path.join(carpeta, "redes.npz")
        self.redes = RedesNumpy(ruta_redes if os.path.exists(ruta_redes) else None)
        if len(self.redes) == 0:
            self.w_rf = 1.0

    # ---------- datos ----------
    def equipos(self, liga=None, temporadas=2):
        """Equipos con partidos en las ultimas `temporadas` temporadas (evita clubes que ya no estan)."""
        df = self.df_hist
        corte = df["date"].max() - pd.Timedelta(days=365 * temporadas)
        df = df[df["date"] >= corte]
        if liga:
            df = df[df["tournament"] == liga]
        return sorted(set(df["home_team"]) | set(df["away_team"]))

    # ---------- modelos ----------
    def _probs_redes(self, x):
        """(n_redes, 3) probabilidades de cada red para una fila x ya escalada."""
        P = self.redes.probs(x)[:, 0, :]
        return P / P.sum(axis=-1, keepdims=True)

    def miembros(self, local, visitante):
        """Probabilidades de cada 'modelo posible' (redes + sub-bosques RF) y el peso de cada uno."""
        feat = construir_features(local, visitante, self.df_hist, self.elos)
        x = self.scaler.transform(np.array([[feat[f] for f in FEATURES]], dtype=np.float32))
        p_arb = np.array([t.predict_proba(x.astype(np.float32))[0] for t in self.rf.estimators_])
        grupos = np.array_split(np.arange(len(p_arb)), N_GRUPOS_RF)
        p_rf = np.array([p_arb[g].mean(axis=0) for g in grupos])
        if len(self.redes):
            p_nn = self._probs_redes(x)
            miembros = np.vstack([p_nn, p_rf])
            w = np.concatenate([np.full(len(p_nn), (1 - self.w_rf) / len(p_nn)),
                                np.full(len(p_rf), self.w_rf / len(p_rf))])
        else:
            miembros = p_rf
            w = np.full(len(p_rf), 1 / len(p_rf))
        return miembros, w, feat

    # ---------- Monte Carlo ----------
    def prediccion(self, local, visitante):
        """Probabilidades exactas (sin sorteo) de cada modelo y de la mezcla, mas las variables del partido."""
        miembros, w, feat = self.miembros(local, visitante)
        n_nn = len(self.redes)
        p_nn = miembros[:n_nn] if n_nn else None
        p_rf = miembros[n_nn:]
        return dict(features=feat, miembros=miembros, pesos=w, p_mezcla=w @ miembros,
                    p_nn=p_nn, p_rf=p_rf, n_arboles=len(self.rf.estimators_), n_redes=n_nn)

    def simular_partido(self, local, visitante, n_sim=20000, semilla=2026):
        rng = np.random.default_rng(semilla)
        miembros, w, feat = self.miembros(local, visitante)
        idx = rng.choice(len(w), size=n_sim, p=w)     # 1) cada simulacion elige un modelo posible
        P = miembros[idx]
        res = sortear(P, rng)                         # 2) y sortea el resultado con sus probabilidades
        freq = np.bincount(res, minlength=3) / n_sim
        ee = np.sqrt(freq * (1 - freq) / n_sim)       # error estandar de Monte Carlo
        lo, hi = np.percentile(P, [5, 95], axis=0)    # rango de desacuerdo entre modelos
        # convergencia: frecuencia acumulada tras n simulaciones
        cp = np.unique(np.geomspace(20, n_sim, 60).astype(int))
        acum = np.cumsum(np.eye(3)[res], axis=0)[cp - 1] / cp[:, None]
        conv = pd.DataFrame({"n": np.repeat(cp, 3), "resultado": np.tile([0, 1, 2], len(cp)),
                             "frecuencia": acum.ravel()})
        return dict(freq=freq, error_mc=ee, rango_lo=lo, rango_hi=hi, p_mezcla=w @ miembros,
                    ganador=LABELS[int(np.argmax(freq))], n_sim=n_sim, features=feat, convergencia=conv)

    # ---------- datos auxiliares para la interfaz ----------
    def historial(self, a, b, n=5):
        """Enfrentamientos directos entre a y b en el dataset."""
        df = self.df_hist
        m = df[((df.home_team == a) & (df.away_team == b)) | ((df.home_team == b) & (df.away_team == a))]
        gana_a = ((m.home_team == a) & (m.home_score > m.away_score)) | ((m.away_team == a) & (m.away_score > m.home_score))
        gana_b = ((m.home_team == b) & (m.home_score > m.away_score)) | ((m.away_team == b) & (m.away_score > m.home_score))
        emp = m.home_score == m.away_score
        ult = m.tail(n).iloc[::-1]
        return dict(
            total=len(m), gana_a=int(gana_a.sum()), gana_b=int(gana_b.sum()), empates=int(emp.sum()),
            desde=(m["date"].min().year if len(m) else None), hasta=(m["date"].max().year if len(m) else None),
            ultimos=[(r.date.strftime("%Y-%m"), r.home_team, int(r.home_score), int(r.away_score), r.away_team)
                     for r in ult.itertuples()])

    def importancias(self, top=12):
        imp = pd.Series(self.rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
        return imp.head(top)

    def ranking_elo(self, top=15):
        reci = set(self.equipos())
        elo = pd.Series({k: v for k, v in self.elos.items() if k in reci}).sort_values(ascending=False)
        return elo.head(top)

    def resumen_dataset(self):
        df = self.df_hist
        return dict(partidos=len(df), equipos=len(set(df.home_team) | set(df.away_team)),
                    desde=df["date"].min().year, hasta=df["date"].max().strftime("%Y-%m-%d"),
                    ligas=df["tournament"].nunique())

    def simular_liga(self, equipos, n_sim=20000, semilla=2026):
        """Liga todos contra todos (ida y vuelta). Una simulacion usa UN mismo modelo en todos
        los partidos, asi la incertidumbre del modelo se propaga de forma coherente."""
        rng = np.random.default_rng(semilla)
        partidos = [(l, v) for l in equipos for v in equipos if l != v]
        pools = []
        for l, v in partidos:
            miembros, w, _ = self.miembros(l, v)
            pools.append(miembros)
        pools = np.stack(pools)                       # (F, n_miembros, 3)
        idx = rng.choice(len(w), size=n_sim, p=w)
        res = sortear(pools[:, idx, :], rng)          # (F, n_sim)
        pos = {e: i for i, e in enumerate(equipos)}
        pts = np.zeros((len(equipos), n_sim))
        for f, (l, v) in enumerate(partidos):
            pts[pos[l]] += np.select([res[f] == 0, res[f] == 1], [3, 1], 0)
            pts[pos[v]] += np.select([res[f] == 2, res[f] == 1], [3, 1], 0)
        # Campeon = mas puntos; si hay empate en puntos se reparte la fraccion (sin sorteo adicional)
        lider = (pts == pts.max(axis=0))
        p_campeon = (lider / lider.sum(axis=0)).mean(axis=1)
        tabla = pd.DataFrame({
            "Equipo": equipos,
            "Puntos esperados": pts.mean(axis=1),
            "Puntos p5": np.percentile(pts, 5, axis=1),
            "Puntos p95": np.percentile(pts, 95, axis=1),
            "Prob. campeón": p_campeon,
        }).sort_values("Puntos esperados", ascending=False).reset_index(drop=True)
        return tabla
