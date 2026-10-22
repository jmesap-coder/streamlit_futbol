import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from futbol import ui
from futbol.features import FEATURES
from futbol.predictor import LIGAS, Predictor

st.set_page_config(page_title="Predictor Fútbol Pro", page_icon="⚽", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(ui.CSS, unsafe_allow_html=True)


def md(html):
    st.markdown(html, unsafe_allow_html=True)


# ---------------- carga y caché ----------------
@st.cache_resource(show_spinner="Cargando modelos...")
def cargar():
    return Predictor("modelos")


@st.cache_data(show_spinner=False)
def pred(local, visitante):
    return cargar().prediccion(local, visitante)


@st.cache_data(show_spinner="Simulando partidos...")
def sim(local, visitante, n, semilla):
    return cargar().simular_partido(local, visitante, n_sim=n, semilla=semilla)


@st.cache_data(show_spinner="Simulando torneo...")
def torneo(equipos, n, semilla):
    return cargar().simular_liga(list(equipos), n_sim=n, semilla=semilla)


@st.cache_data(show_spinner=False)
def hist(a, b):
    return cargar().historial(a, b)


try:
    P = cargar()
except FileNotFoundError:
    st.error("No se encontraron los modelos en la carpeta `modelos/`. "
             "Ejecuta la celda de exportación del notebook y descomprime el .zip en la raíz del repositorio.")
    st.stop()

PRESETS = [("El Clásico", "La Liga", "Real Madrid", "Barcelona"),
           ("Derbi de Mánchester", "Premier League", "Man City", "Man United"),
           ("Derbi de Italia", "Serie A", "Inter", "Juventus"),
           ("Der Klassiker", "Bundesliga", "Bayern Munich", "Dortmund"),
           ("Le Classique", "Ligue 1", "Paris SG", "Marseille"),
           ("Clásico inglés", "Premier League", "Liverpool", "Arsenal")]


def fijar_partido(local, visitante):
    st.session_state["liga"] = "Todas"
    st.session_state["local"] = local
    st.session_state["visitante"] = visitante


def altair_base(chart):
    return (chart.configure_view(strokeWidth=0).configure(background="transparent")
            .configure_axis(labelColor="#94a3b8", titleColor="#94a3b8", gridColor="#16213a", domainColor="#1e2b47")
            .configure_legend(labelColor="#cbd5e1", titleColor="#94a3b8"))


md(ui.header(P.config.get("fecha_datos", "—")))
barra = st.empty()
t_pred, t_mod, t_mc, t_torneo, t_data, t_met = st.tabs(
    ["Predicción", "Modelos paralelos", "Monte Carlo", "Torneo", "Dataset", "Metodología"])

# =====================================================================
# PREDICCIÓN
# =====================================================================
with t_pred:
    md(ui.hero("Arquitectura de predicción dual y estocástica",
               "Predicción paralela: red neuronal + Random Forest",
               "Compara en paralelo una red neuronal residual y un Random Forest, combinados en un ensamble "
               "cuyo peso se calibra en validación, y explora la incertidumbre con simulación Monte Carlo."))
    if len(P.redes) == 0:
        st.warning("Modo solo Random Forest: no se encontró `modelos/redes.npz`.")

    todos = P.equipos()
    presets = [p for p in PRESETS if p[2] in todos and p[3] in todos]
    md(ui.seccion("Enfrentamientos clásicos preconfigurados", "Datos de las 5 grandes ligas europeas"))
    for col, (nom, lg, l, v) in zip(st.columns(len(presets)), presets):
        col.button(f"{nom} · {lg}", key=f"pre_{nom}", on_click=fijar_partido, args=(l, v), width="stretch")

    st.session_state.setdefault("liga", "Todas")
    md(ui.seccion("Filtrar por liga"))
    liga_sel = st.segmented_control("Filtrar por liga", ["Todas"] + list(LIGAS.values()), key="liga",
                                    label_visibility="collapsed") or "Todas"
    codigo = None if liga_sel == "Todas" else [k for k, v in LIGAS.items() if v == liga_sel][0]
    equipos = P.equipos(codigo)
    for k, d in (("local", "Real Madrid"), ("visitante", "Barcelona")):
        if st.session_state.get(k) not in equipos:
            st.session_state[k] = d if d in equipos else equipos[0]
    if st.session_state["local"] == st.session_state["visitante"]:
        st.session_state["visitante"] = next(e for e in equipos if e != st.session_state["local"])

    st.write("")
    c1, c2 = st.columns(2)
    with c1, st.container(key="card_local"):
        md('<div class="eyebrow" style="--c:#10b981">Equipo local (cancha propia)</div>')
        local = st.selectbox("Equipo local", equipos, key="local", label_visibility="collapsed")
        ph_local = st.empty()
    with c2, st.container(key="card_visit"):
        md('<div class="eyebrow" style="--c:#06b6d4">Equipo visitante</div>')
        visitante = st.selectbox("Equipo visitante", equipos, key="visitante", label_visibility="collapsed")
        ph_visit = st.empty()

    if local == visitante:
        st.info("Elige dos equipos distintos.")
        st.stop()

    pr = pred(local, visitante)
    f = pr["features"]
    ph_local.markdown(ui.stats_equipo([
        ("ELO", f"{f['elo_local']:.0f}"),
        ("Victorias últ. 5 en casa", f"{f['victorias_ultimos_5_local']:.0%}"),
        ("Goles a favor (prom. 5)", f"{f['promedio_goles_ultimos_5_local']:.2f}"),
        ("Goles en contra (prom. 5)", f"{f['promedio_goles_recibidos_ultimos_5_local']:.2f}"),
        ("Forma ponderada", f"{f['forma_ponderada_local']:.2f}"),
        ("Racha de victorias", f"{f['racha_victorias_local']:.0f}")]), unsafe_allow_html=True)
    ph_visit.markdown(ui.stats_equipo([
        ("ELO", f"{f['elo_visitante']:.0f}"),
        ("Victorias últ. 5 de visita", f"{f['victorias_ultimos_5_visit']:.0%}"),
        ("Goles a favor (prom. 5)", f"{f['promedio_goles_ultimos_5_visit']:.2f}"),
        ("Goles en contra (prom. 5)", f"{f['promedio_goles_recibidos_ultimos_5_visit']:.2f}"),
        ("Forma ponderada", f"{f['forma_ponderada_visit']:.2f}"),
        ("Racha de victorias", f"{f['racha_victorias_visit']:.0f}")]), unsafe_allow_html=True)

    mezcla = pr["p_mezcla"]
    k_max = int(np.argmax(mezcla))
    nombres = [f"Local ({local})", "Empate", f"Visitante ({visitante})"]
    veredicto = ["Local gana", "Empate", "Visitante gana"][k_max]
    barra.markdown(ui.barra_matchup(local, visitante, veredicto, mezcla[k_max], ui.COLORES[k_max]),
                   unsafe_allow_html=True)

    # historial real
    h = hist(local, visitante)
    st.write("")
    if h["total"]:
        resumen = (f'<span style="color:{ui.COL_L}">{h["gana_a"]} {ui.esc(local)}</span> · '
                   f'<span style="color:{ui.COL_E}">{h["empates"]} empates</span> · '
                   f'<span style="color:{ui.COL_V}">{h["gana_b"]} {ui.esc(visitante)}</span>')
        filas = [f"{d} · <b>{ui.esc(hl)} {gl} - {ga} {ui.esc(av)}</b>" for d, hl, gl, ga, av in h["ultimos"]]
        md(ui.limpiar(f"""<div class="card"><div class="mc-h"><span style="color:#fff">Historial real en el dataset
        ({h['total']} partidos, {h['desde']}–{h['hasta']})</span><span class="mono" style="font-size:.84rem">{resumen}</span></div>
        <div style="color:#7385a6;font-size:.8rem;margin-top:10px">Últimos enfrentamientos</div>{ui.chips_partidos(filas)}</div>"""))
    else:
        md(ui.limpiar('<div class="card"><span style="color:#8294b3">Estos equipos no se han enfrentado en el dataset.</span></div>'))

    # consenso
    desc = (f"Mezcla de {pr['n_redes']} redes neuronales y {pr['n_arboles']} árboles de decisión "
            f"(peso del Random Forest: {P.w_rf:.0%})." if pr["n_redes"]
            else f"Random Forest de {pr['n_arboles']} árboles.")
    md(ui.limpiar(f"""<div class="card"><div class="mc-h"><span style="color:#c3cee3;font-size:.95rem;letter-spacing:.03em">
    CONSENSO DEL ENSAMBLE PARALELO</span><span class="pill-v">Veredicto determinista: {veredicto} ({mezcla[k_max]:.0%})</span></div>
    <div style="color:#8294b3;font-size:.86rem;margin-top:6px">{desc}</div>{ui.barra_apilada(mezcla, nombres)}</div>"""))

    i1, i2 = st.columns(2)
    i1.markdown(ui.limpiar("""<div class="card"><b style="color:#fff">Desglose red neuronal vs Random Forest</b>
    <div style="color:#8294b3;font-size:.86rem;margin-top:6px">Compara las probabilidades de cada modelo, su
    acuerdo y las variables que más pesan en el bosque. Ve a la pestaña <b>Modelos paralelos</b>.</div></div>"""),
                unsafe_allow_html=True)
    i2.markdown(ui.limpiar("""<div class="card"><b style="color:#fff">Del determinismo a Monte Carlo</b>
    <div style="color:#8294b3;font-size:.86rem;margin-top:6px">Simula miles de partidos eligiendo en cada uno un modelo
    posible, para obtener frecuencias con intervalos de confianza. Ve a la pestaña <b>Monte Carlo</b>.</div></div>"""),
                unsafe_allow_html=True)

# =====================================================================
# MODELOS PARALELOS
# =====================================================================
with t_mod:
    md(ui.hero("Modelado paralelo: red neuronal residual + Random Forest",
               f"Comparativa en paralelo: {ui.esc(local)} vs {ui.esc(visitante)}",
               "Probabilidades independientes de cada modelo sobre las mismas variables, su nivel de acuerdo, "
               "las variables más importantes del bosque y el desempeño de cada uno en el conjunto de prueba.",
               "#6366f1"))
    p_rf = pr["p_rf"].mean(axis=0)
    if pr["p_nn"] is not None:
        p_nn = pr["p_nn"].mean(axis=0)
        a, b = int(np.argmax(p_nn)), int(np.argmax(p_rf))
        res = ["Local gana", "Empate", "Visitante gana"]
        if a != b:
            md(ui.alerta(f"Discrepancia entre modelos: la red predice «{res[a]}» y el Random Forest «{res[b]}»",
                         "Cuando los modelos discrepan el partido es más incierto; el ensamble pondera ambos pronósticos."))
        else:
            md(ui.alerta(f"Los dos modelos coinciden: «{res[a]}»",
                         "Ambos modelos señalan el mismo resultado más probable.", ok=True))
        m1, m2 = st.columns(2)
        nn_cuerpo = ui.barras_prob(p_nn, nombres, pr["p_nn"].min(axis=0), pr["p_nn"].max(axis=0))
        m1.markdown(ui.tarjeta_modelo("Red neuronal (MLP residual)", f"{pr['n_redes']} semillas",
                                      "Ensemble de redes con bloques residuales, BatchNorm y Dropout. "
                                      "El rango muestra el desacuerdo entre semillas.", nn_cuerpo, "#6366f1"),
                    unsafe_allow_html=True)
        rf_cuerpo = ui.barras_prob(p_rf, nombres, pr["p_rf"].min(axis=0), pr["p_rf"].max(axis=0))
        m2.markdown(ui.tarjeta_modelo("Random Forest", f"{pr['n_arboles']} árboles",
                                      "Bosque de árboles de decisión sobre las mismas variables. El rango muestra "
                                      "el desacuerdo entre 20 sub-bosques.", rf_cuerpo, "#10b981"),
                    unsafe_allow_html=True)
    else:
        md(ui.tarjeta_modelo("Random Forest", f"{pr['n_arboles']} árboles", "Único modelo disponible.",
                             ui.barras_prob(p_rf, nombres), "#10b981"))

    g1, g2 = st.columns(2)
    with g1:
        md(ui.seccion("Importancia de variables (Gini) del Random Forest"))
        imp = P.importancias(12).reset_index()
        imp.columns = ["Variable", "Importancia"]
        ch = (alt.Chart(imp).mark_bar(color="#10b981", cornerRadiusEnd=4)
              .encode(x=alt.X("Importancia:Q", title=None), y=alt.Y("Variable:N", sort="-x", title=None))
              .properties(height=330, width="container"))
        st.altair_chart(altair_base(ch), theme=None)
    with g2:
        md(ui.seccion("Desempeño en el conjunto de prueba"))
        met = P.config.get("metricas_test")
        if met:
            dm = pd.DataFrame(met).set_index("modelo").rename(index={
                "RedNeuronal": "Red neuronal", "RandomForest": "Random Forest", "Mezcla": "Mezcla (ensamble)"})
            dm.columns = ["Accuracy", "Log-loss", "Brier", "RPS"]
            st.dataframe(dm.style.format({"Accuracy": "{:.1%}", "Log-loss": "{:.3f}", "Brier": "{:.3f}", "RPS": "{:.3f}"}),
                         width="stretch")
            st.caption("Menor es mejor en log-loss, Brier y RPS. Las filas «Base» muestran cuánto aportan "
                       "realmente los modelos frente a referencias simples.")

# =====================================================================
# MONTE CARLO
# =====================================================================
with t_mc:
    md(ui.hero("Superación del determinismo: simulación estocástica",
               f"Motor Monte Carlo: {ui.esc(local)} vs {ui.esc(visitante)}",
               "En cada simulación se elige al azar un modelo posible (las redes y los sub-bosques, ponderados por su "
               "peso) y con sus probabilidades se sortea el resultado. Así se obtienen frecuencias con intervalos de "
               "confianza y se ve cuánto discrepan los modelos.", ui.COL_V))
    st.session_state.setdefault("n_sim", 20000)
    cc1, cc2 = st.columns([3, 1])
    with cc1:
        md('<div class="sect">Número de iteraciones (N)</div>')
        n_sim = st.segmented_control("N", [1000, 5000, 10000, 20000, 50000], key="n_sim",
                                     format_func=lambda v: f"{v:,}".replace(",", "."), label_visibility="collapsed") or 20000
        st.caption("Mayor N = menor margen de error de Monte Carlo (≈ 1/√N).")
    with cc2:
        semilla = st.number_input("Semilla", value=2026, step=1, help="Misma semilla = mismo resultado.")

    r = sim(local, visitante, int(n_sim), int(semilla))
    md(ui.seccion(f"Distribución empírica tras {int(n_sim):,} simulaciones".replace(",", ".")))
    titulos = [f"Victoria local ({local})", "Empate", f"Victoria visitante ({visitante})"]
    kmax = int(np.argmax(r["freq"]))
    for k, col in enumerate(st.columns(3)):
        col.markdown(ui.tarjeta_mc(titulos[k], ui.COLORES[k], r["freq"][k], r["error_mc"][k],
                                   r["rango_lo"][k], r["rango_hi"][k], r["p_mezcla"][k], k == kmax),
                     unsafe_allow_html=True)
    st.write("")
    q1, q2 = st.columns(2)
    with q1:
        md(ui.seccion("Frecuencia con intervalo de confianza al 95%"))
        dfc = pd.DataFrame({"Resultado": ["Local", "Empate", "Visitante"], "Frecuencia": r["freq"],
                            "lo": r["freq"] - 1.96 * r["error_mc"], "hi": r["freq"] + 1.96 * r["error_mc"]})
        dom, rng_ = ["Local", "Empate", "Visitante"], ui.COLORES
        base = alt.Chart(dfc).encode(y=alt.Y("Resultado:N", sort=dom, title=None))
        barras = base.mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Frecuencia:Q", axis=alt.Axis(format="%"), title=None),
            color=alt.Color("Resultado:N", scale=alt.Scale(domain=dom, range=rng_), legend=None))
        err = base.mark_rule(color="#e5e7eb", strokeWidth=2).encode(x="lo:Q", x2="hi:Q")
        st.altair_chart(altair_base((barras + err).properties(height=220, width="container")), theme=None)
    with q2:
        md(ui.seccion("Convergencia: la frecuencia se acerca a la probabilidad de la mezcla"))
        cv = r["convergencia"].copy()
        cv["Resultado"] = cv["resultado"].map({0: "Local", 1: "Empate", 2: "Visitante"})
        lineas = alt.Chart(cv).mark_line(strokeWidth=2).encode(
            x=alt.X("n:Q", scale=alt.Scale(type="log"), title="Simulaciones (escala log)"),
            y=alt.Y("frecuencia:Q", axis=alt.Axis(format="%"), title=None),
            color=alt.Color("Resultado:N", scale=alt.Scale(domain=dom, range=rng_), legend=alt.Legend(orient="bottom")))
        ref = pd.DataFrame({"Resultado": dom, "p": r["p_mezcla"]})
        reglas = alt.Chart(ref).mark_rule(strokeDash=[5, 4], opacity=0.7).encode(
            y="p:Q", color=alt.Color("Resultado:N", scale=alt.Scale(domain=dom, range=rng_), legend=None))
        st.altair_chart(altair_base((lineas + reglas).properties(height=240, width="container")), theme=None)
    st.caption("Las líneas punteadas son las probabilidades exactas de la mezcla; el Monte Carlo converge a ellas. "
               "Su valor está en cuantificar la incertidumbre y en encadenar partidos (pestaña Torneo).")

# =====================================================================
# TORNEO
# =====================================================================
with t_torneo:
    md(ui.hero("Simulación de liga todos contra todos",
               "Torneo Monte Carlo",
               "Cada simulación juega todos los partidos de ida y vuelta usando un mismo modelo posible, de modo que "
               "la incertidumbre del modelo se propaga de forma coherente. Se estiman puntos esperados y probabilidad "
               "de campeón.", ui.COL_E))
    defecto = [e for e in ["Real Madrid", "Barcelona", "Ath Madrid", "Sevilla"] if e in todos]
    elegidos = st.multiselect("Equipos (de 3 a 6)", todos, default=defecto, max_selections=6)
    if len(elegidos) < 3:
        st.info("Elige al menos 3 equipos.")
    else:
        tb = torneo(tuple(elegidos), int(n_sim), int(semilla))
        t1, t2 = st.columns([3, 2])
        with t1:
            st.dataframe(tb.style.format({"Puntos esperados": "{:.1f}", "Puntos p5": "{:.0f}",
                                          "Puntos p95": "{:.0f}", "Prob. campeón": "{:.1%}"}),
                         hide_index=True, width="stretch")
        with t2:
            ch = (alt.Chart(tb).mark_bar(color=ui.COL_E, cornerRadiusEnd=4)
                  .encode(x=alt.X("Prob. campeón:Q", axis=alt.Axis(format="%"), title=None),
                          y=alt.Y("Equipo:N", sort="-x", title=None)).properties(height=40 * len(tb) + 20, width="container"))
            st.altair_chart(altair_base(ch), theme=None)
        st.caption(f"{int(n_sim):,} simulaciones (se ajustan en la pestaña Monte Carlo). Si dos equipos empatan en puntos, "
                   "la probabilidad de campeón se reparte entre ellos.".replace(",", "."))

# =====================================================================
# DATASET
# =====================================================================
with t_data:
    rd = P.resumen_dataset()
    md(ui.hero("Datos", "Dataset de las 5 grandes ligas europeas",
               "Partidos de la Premier League, La Liga, Serie A, Bundesliga y Ligue 1. El ELO se calcula de forma "
               "dinámica y las variables de cada partido usan solo información anterior a su fecha.", "#a855f7"))
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(ui.kpi("Partidos", f"{rd['partidos']:,}".replace(",", ".")), unsafe_allow_html=True)
    k2.markdown(ui.kpi("Equipos", rd["equipos"]), unsafe_allow_html=True)
    k3.markdown(ui.kpi("Ligas", rd["ligas"]), unsafe_allow_html=True)
    k4.markdown(ui.kpi("Desde", f"{rd['desde']} → {rd['hasta'][:7]}"), unsafe_allow_html=True)
    st.write("")
    md(ui.seccion("Ranking ELO actual (equipos activos)"))
    rk = P.ranking_elo(15).reset_index()
    rk.columns = ["Equipo", "ELO"]
    ch = (alt.Chart(rk).mark_bar(color="#a855f7", cornerRadiusEnd=4)
          .encode(x=alt.X("ELO:Q", scale=alt.Scale(zero=False), title=None), y=alt.Y("Equipo:N", sort="-x", title=None))
          .properties(height=380, width="container"))
    st.altair_chart(altair_base(ch), theme=None)
    with st.expander(f"Variables del modelo ({len(FEATURES)})"):
        st.write(", ".join(f"`{v}`" for v in FEATURES))

# =====================================================================
# METODOLOGÍA
# =====================================================================
with t_met:
    md(ui.hero("Cómo funciona", "Metodología y límites",
               "Resumen del proceso para entender qué calcula cada número de la app.", "#64748b"))
    st.markdown(f"""
**1. Variables.** Para cada partido se construyen {len(FEATURES)} variables (ELO, forma reciente, goles a favor y en
contra, rachas y enfrentamientos directos) usando solo información anterior a esa fecha.

**2. Dos modelos.** Una red neuronal residual (ensemble de {len(P.redes)} semillas) y un Random Forest de
{len(P.rf.estimators_)} árboles se entrenan sobre los mismos datos. El peso del Random Forest en la mezcla
(**{P.w_rf:.0%}**) se eligió minimizando el log-loss en validación.

**3. Monte Carlo.** En cada simulación se elige un modelo posible (redes y sub-bosques del Random Forest, ponderados) y
se sortea el resultado con sus probabilidades. La frecuencia converge a la probabilidad de la mezcla; el intervalo de
confianza mide el error por usar un número finito de simulaciones y el rango entre modelos mide su desacuerdo.

**4. Métricas.** Además de accuracy se usan métricas probabilísticas: log-loss, Brier y RPS (que respeta el orden
local > empate > visitante).

**Límites.**
- El modelo predice el **resultado** (1X2), no los goles: no entrega marcadores exactos.
- Las variables de cada equipo se calculan con su historial en el dataset; equipos con pocos partidos recientes tienen
  estimaciones menos fiables.
- El fútbol tiene mucho azar: incluso un buen modelo acierta poco más de la mitad de los resultados.
- Proyecto académico: no es asesoría de apuestas.
""")

md(ui.PIE)
