import numpy as np
import pandas as pd
import streamlit as st

from futbol.features import LABELS
from futbol.predictor import LIGAS, Predictor

st.set_page_config(page_title="Predicción de partidos de clubes", page_icon="⚽", layout="wide")


@st.cache_resource(show_spinner="Cargando modelos...")
def cargar():
    return Predictor("modelos")


@st.cache_data(show_spinner="Simulando partido...")
def sim_partido(local, visitante, n_sim, semilla):
    return cargar().simular_partido(local, visitante, n_sim=n_sim, semilla=semilla)


@st.cache_data(show_spinner="Simulando liga...")
def sim_liga(equipos, n_sim, semilla):
    return cargar().simular_liga(list(equipos), n_sim=n_sim, semilla=semilla)


try:
    P = cargar()
except FileNotFoundError:
    st.error("No se encontraron los modelos en la carpeta `modelos/`. "
             "Ejecuta la celda de exportación del notebook y descomprime el .zip en la raíz del repositorio.")
    st.stop()

st.title("⚽ Predicción de partidos de clubes")
st.caption(f"Cinco grandes ligas europeas · datos hasta {P.config.get('fecha_datos', '—')} · "
           "Random Forest + redes neuronales · simulación Monte Carlo")
if len(P.redes) == 0:
    st.warning("Modo solo Random Forest: no se encontró `modelos/redes.npz`.")

with st.sidebar:
    st.header("Simulación")
    n_sim = st.select_slider("Número de simulaciones", [1000, 5000, 10000, 20000, 50000], value=20000)
    semilla = st.number_input("Semilla (misma semilla = mismo resultado)", value=2026, step=1)
    st.markdown("---")
    liga_txt = st.selectbox("Filtrar equipos por liga", ["Todas"] + list(LIGAS.values()))
    liga = None if liga_txt == "Todas" else [k for k, v in LIGAS.items() if v == liga_txt][0]

equipos = P.equipos(liga)
tab_partido, tab_liga, tab_modelo = st.tabs(["Partido", "Mini liga", "Sobre el modelo"])

# ---------------- Partido ----------------
with tab_partido:
    c1, c2 = st.columns(2)
    ixl = equipos.index("Real Madrid") if "Real Madrid" in equipos else 0
    ixv = equipos.index("Barcelona") if "Barcelona" in equipos else min(1, len(equipos) - 1)
    local = c1.selectbox("Equipo local", equipos, index=ixl)
    visitante = c2.selectbox("Equipo visitante", equipos, index=ixv)

    if local == visitante:
        st.info("Elige dos equipos distintos.")
    else:
        r = sim_partido(local, visitante, int(n_sim), int(semilla))
        st.subheader(f"{local} vs {visitante}")
        cols = st.columns(3)
        nombres = [f"{local} gana", "Empate", f"{visitante} gana"]
        for k in range(3):
            cols[k].metric(nombres[k], f"{r['freq'][k]:.1%}", f"± {1.96 * r['error_mc'][k]:.1%} (Monte Carlo)",
                           delta_color="off")
        st.bar_chart(pd.DataFrame({"Probabilidad": r["freq"]}, index=nombres), horizontal=True)
        st.write(f"**Resultado más frecuente en las simulaciones:** {r['ganador']}")

        rango = pd.DataFrame({
            "Resultado": nombres,
            "Prob. media": [f"{v:.1%}" for v in r["freq"]],
            "Rango entre modelos (5%-95%)": [f"{a:.1%} – {b:.1%}" for a, b in zip(r["rango_lo"], r["rango_hi"])],
        }).set_index("Resultado")
        st.table(rango)
        with st.expander("¿Cómo se calcula?"):
            st.markdown(
                "En cada simulación se elige al azar un **modelo posible** (las redes neuronales y "
                "sub-bosques del Random Forest, ponderados según su peso en la mezcla) y con sus "
                "probabilidades se **sortea el resultado**. La frecuencia converge a la probabilidad "
                "de la mezcla; el *± Monte Carlo* es el error por usar un número finito de simulaciones "
                "y el *rango entre modelos* muestra cuánto discrepan los modelos entre sí.")

# ---------------- Mini liga ----------------
with tab_liga:
    st.write("Liga todos contra todos (ida y vuelta) entre los equipos elegidos. "
             "Dentro de cada simulación se usa un mismo modelo para todos los partidos.")
    por_defecto = [e for e in ["Real Madrid", "Barcelona", "Ath Madrid", "Sevilla"] if e in equipos]
    elegidos = st.multiselect("Equipos (de 3 a 6)", equipos, default=por_defecto, max_selections=6)
    if len(elegidos) < 3:
        st.info("Elige al menos 3 equipos.")
    elif st.button("Simular liga", type="primary"):
        tabla = sim_liga(tuple(elegidos), int(n_sim), int(semilla))
        st.dataframe(tabla.style.format({"Puntos esperados": "{:.1f}", "Puntos p5": "{:.0f}",
                                         "Puntos p95": "{:.0f}", "Prob. campeón": "{:.1%}"}),
                     hide_index=True, width="stretch")
        st.bar_chart(tabla.set_index("Equipo")["Prob. campeón"])

# ---------------- Sobre el modelo ----------------
with tab_modelo:
    st.markdown(
        f"- **Datos:** partidos de las 5 grandes ligas europeas (hasta {P.config.get('fecha_datos', '—')}).\n"
        "- **Variables:** ELO dinámico, forma reciente, goles a favor y en contra, rachas y cara a cara.\n"
        f"- **Modelos:** {len(P.redes)} redes neuronales residuales + Random Forest de {len(P.rf.estimators_)} árboles. "
        f"Peso del Random Forest en la mezcla: {P.w_rf:.2f}.")
    met = P.config.get("metricas_test")
    if met:
        st.subheader("Desempeño en el conjunto de prueba")
        df_met = pd.DataFrame(met).set_index("modelo")
        st.dataframe(df_met.style.format("{:.4f}"), width="stretch")
        st.caption("Menor es mejor en log_loss, brier y RPS. Las referencias (frecuencias y solo ELO) "
                   "muestran cuánto aportan realmente los modelos.")
