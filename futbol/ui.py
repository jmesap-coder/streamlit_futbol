"""Estilos y bloques HTML de la app (tema oscuro)."""
import html as _html

COL_L, COL_E, COL_V = "#10b981", "#f59e0b", "#06b6d4"   # local / empate / visitante
COLORES = [COL_L, COL_E, COL_V]
NOMBRES_BASE = ["Local", "Empate", "Visitante"]


def limpiar(s):
    """Quita sangrias y saltos de linea: Markdown trata las lineas con 4+ espacios como codigo."""
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


def esc(x):
    return _html.escape(str(x))


CSS = limpiar("""<style>
.stApp{background:radial-gradient(1100px 520px at 8% -8%,#0c2033 0%,#050816 58%);}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
.block-container{max-width:1180px;padding-top:1.1rem;padding-bottom:3rem;}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
.topbar{display:flex;justify-content:space-between;align-items:center;gap:16px;padding:6px 2px 14px;border-bottom:1px solid #16213a;margin-bottom:10px;}
.brand{display:flex;align-items:center;gap:14px;}
.logo{width:46px;height:46px;border-radius:14px;background:#0b2a2a;border:1px solid #14493f;display:flex;align-items:center;justify-content:center;font-size:24px;}
.brand-t{font-size:1.35rem;font-weight:800;color:#fff;line-height:1.1;}
.brand-s{font-size:.82rem;color:#8b9bb8;margin-top:2px;}
.pill{border:1px solid #1e2b47;background:#0d1628;color:#9fb0cc;border-radius:999px;padding:6px 14px;font-size:.78rem;white-space:nowrap;}
.pill-v{border-radius:10px;padding:7px 14px;font-weight:700;color:#fff;background:#141e33;border:1px solid #243354;}
.mbar{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;border:1px solid #1b2740;background:#0a1120;border-radius:14px;padding:13px 18px;margin:6px 0 14px;font-size:.86rem;color:#8b9bb8;}
.mbar b{color:#fff;}
.card{background:#0b1220;border:1px solid #1b2740;border-radius:18px;padding:24px 28px;margin-bottom:16px;}
.card-top{border-top:4px solid var(--c,#10b981);}
.eyebrow{font-size:.74rem;letter-spacing:.09em;text-transform:uppercase;font-weight:700;color:var(--c,#10b981);display:flex;align-items:center;gap:8px;}
.eyebrow:before{content:"";width:8px;height:8px;border-radius:50%;background:var(--c,#10b981);display:inline-block;}
.h-title{font-size:2.05rem;font-weight:800;color:#fff;line-height:1.15;margin:8px 0 10px;}
.h-sub{color:#8294b3;font-size:1rem;line-height:1.55;max-width:860px;}
.sect{font-size:.8rem;letter-spacing:.07em;text-transform:uppercase;font-weight:700;color:#9fb0cc;margin:14px 0 8px;}
.sect-r{float:right;text-transform:none;letter-spacing:0;font-weight:400;color:#7385a6;}
.tname{font-size:1.5rem;font-weight:800;color:#fff;margin:2px 0 6px;}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:10px;}
.stat{background:#080e1c;border:1px solid #16213a;border-radius:12px;padding:10px 14px;}
.stat-l{font-size:.7rem;color:#7385a6;text-transform:uppercase;letter-spacing:.05em;}
.stat-v{font-size:1.2rem;font-weight:700;color:#fff;font-family:ui-monospace,Menlo,Consolas,monospace;}
.chips{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px;}
.chip{background:#0a1120;border:1px solid #1e2b47;border-radius:10px;padding:7px 12px;font-size:.8rem;color:#b6c3dc;font-family:ui-monospace,Menlo,Consolas,monospace;}
.chip b{color:#fff;}
.stack{display:flex;height:34px;border-radius:10px;overflow:hidden;margin:14px 0 10px;}
.stack div{display:flex;align-items:center;justify-content:center;font-weight:700;color:#04210f;font-size:.86rem;font-family:ui-monospace,Menlo,Consolas,monospace;}
.legend{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;font-size:.84rem;font-family:ui-monospace,Menlo,Consolas,monospace;}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;}
.pbar{margin:14px 0;}
.pbar-h{display:flex;justify-content:space-between;font-size:.86rem;color:#c3cee3;margin-bottom:6px;font-family:ui-monospace,Menlo,Consolas,monospace;}
.pbar-t{height:9px;border-radius:99px;background:#16213a;overflow:hidden;}
.pbar-f{height:100%;border-radius:99px;}
.pbar-r{font-size:.72rem;color:#6f81a3;margin-top:4px;font-family:ui-monospace,Menlo,Consolas,monospace;}
.alert{border-radius:14px;padding:14px 20px;margin-bottom:16px;border:1px solid;}
.alert b{display:block;font-size:1rem;margin-bottom:2px;}
.alert span{font-size:.86rem;color:#a9b6cf;}
.alert-w{background:#1a140c;border-color:#4a3510;}.alert-w b{color:#fbbf24;}
.alert-ok{background:#0b1a16;border-color:#14493f;}.alert-ok b{color:#34d399;}
.mc-n{font-size:2.3rem;font-weight:800;font-family:ui-monospace,Menlo,Consolas,monospace;margin:6px 0 2px;}
.mc-h{display:flex;justify-content:space-between;align-items:center;font-size:.86rem;font-weight:700;}
.mc-h small{color:#7385a6;font-weight:400;font-family:ui-monospace,Menlo,Consolas,monospace;}
.mc-d{font-size:.78rem;color:#8294b3;line-height:1.7;font-family:ui-monospace,Menlo,Consolas,monospace;}
.kpi{background:#0b1220;border:1px solid #1b2740;border-radius:16px;padding:16px 20px;}
.kpi-l{font-size:.72rem;color:#7385a6;text-transform:uppercase;letter-spacing:.06em;}
.kpi-v{font-size:1.7rem;font-weight:800;color:#fff;}
.foot{text-align:center;color:#5d6f90;font-size:.78rem;border-top:1px solid #16213a;margin-top:28px;padding-top:16px;}
/* widgets */
.st-key-card_local,.st-key-card_visit,.st-key-card_ctrl{background:#0b1220;border:1px solid #1b2740;border-radius:18px;padding:22px 26px;}
.st-key-card_local{border-top:4px solid #10b981;}
.st-key-card_visit{border-top:4px solid #06b6d4;}
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid #16213a;}
.stTabs [data-baseweb="tab"]{height:42px;padding:0 16px;border-radius:10px 10px 0 0;color:#8b9bb8;}
.stTabs [aria-selected="true"]{color:#10b981 !important;background:#0d1a1f;}
.stTabs [data-baseweb="tab-highlight"]{background:#10b981;}
div[data-testid="stButton"] button{border-radius:12px;background:#0d1628;border:1px solid #1e2b47;color:#b6c3dc;font-size:.82rem;padding:.35rem .5rem;}
div[data-testid="stButton"] button:hover{border-color:#10b981;color:#10b981;}
div[data-baseweb="select"]>div{background:#080e1c;border-color:#1e2b47;border-radius:12px;}
</style>""")


def header(fecha):
    return limpiar(f"""<div class="topbar"><div class="brand"><div class="logo">⚽</div><div>
    <div class="brand-t">Predictor Fútbol Pro</div>
    <div class="brand-s">Red neuronal + Random Forest &amp; Monte Carlo</div></div></div>
    <div class="pill">Datos hasta {esc(fecha)}</div></div>""")


def barra_matchup(local, visitante, veredicto, conf, color):
    return limpiar(f"""<div class="mbar mono"><div>Matchup activo: <b>{esc(local)}</b> (Local) vs
    <b>{esc(visitante)}</b> (Visitante)</div><div>Consenso ensamble:
    <b style="color:{color}">{esc(veredicto)} ({conf:.0%})</b></div></div>""")


def hero(eyebrow, titulo, texto, color=COL_L):
    return limpiar(f"""<div class="card" style="--c:{color}"><div class="eyebrow">{esc(eyebrow)}</div>
    <div class="h-title">{titulo}</div><div class="h-sub">{texto}</div></div>""")


def seccion(texto, derecha=""):
    r = f'<span class="sect-r">{derecha}</span>' if derecha else ""
    return f'<div class="sect">{r}{esc(texto)}</div>'


def stats_equipo(items):
    celdas = "".join(f'<div class="stat"><div class="stat-l">{esc(l)}</div><div class="stat-v">{esc(v)}</div></div>'
                     for l, v in items)
    return f'<div class="grid">{celdas}</div>'


def barra_apilada(p, etiquetas):
    segs = "".join(
        f'<div style="width:{v * 100:.2f}%;background:{c}">{v:.0%}</div>' if v >= 0.07
        else f'<div style="width:{v * 100:.2f}%;background:{c}"></div>'
        for v, c in zip(p, COLORES))
    leyenda = "".join(
        f'<div style="color:{c}"><span class="dot" style="background:{c}"></span>{esc(e)}: {v:.0%}</div>'
        for v, c, e in zip(p, COLORES, etiquetas))
    return f'<div class="stack">{segs}</div><div class="legend">{leyenda}</div>'


def barras_prob(p, etiquetas, lo=None, hi=None):
    out = ""
    for k, (v, c, e) in enumerate(zip(p, COLORES, etiquetas)):
        rango = (f'<div class="pbar-r">rango entre modelos: {lo[k]:.0%} – {hi[k]:.0%}</div>'
                 if lo is not None else "")
        out += (f'<div class="pbar"><div class="pbar-h"><span>{esc(e)}</span><b style="color:{c}">{v:.0%}</b></div>'
                f'<div class="pbar-t"><div class="pbar-f" style="width:{v * 100:.1f}%;background:{c}"></div></div>{rango}</div>')
    return out


def tarjeta_modelo(titulo, etiqueta, descripcion, cuerpo, color):
    return limpiar(f"""<div class="card card-top" style="--c:{color}"><div class="mc-h"><span style="color:#fff;font-size:1.05rem">
    {esc(titulo)}</span><span class="pill">{esc(etiqueta)}</span></div>
    <div style="color:#8294b3;font-size:.84rem;margin-top:6px">{descripcion}</div>{cuerpo}</div>""")


def alerta(titulo, texto, ok=False):
    return limpiar(f'<div class="alert {"alert-ok" if ok else "alert-w"}"><b>{titulo}</b><span>{texto}</span></div>')


def tarjeta_mc(titulo, color, freq, ee, lo, hi, det, es_max):
    marca = "✓ Sí" if es_max else "✗ No"
    return limpiar(f"""<div class="card" style="--c:{color};margin-bottom:0"><div class="mc-h">
    <span style="color:{color}">{esc(titulo)}</span><small>Argmax: {marca}</small></div>
    <div class="mc-n" style="color:#fff">{freq:.1%}</div>
    <div class="mc-d">IC 95% (Monte Carlo): ±{1.96 * ee:.1%}<br>Rango entre modelos: {lo:.1%} – {hi:.1%}<br>
    Mezcla exacta: {det:.1%}</div></div>""")


def kpi(etiqueta, valor):
    return f'<div class="kpi"><div class="kpi-l">{esc(etiqueta)}</div><div class="kpi-v">{esc(valor)}</div></div>'


def chips_partidos(filas):
    return '<div class="chips">' + "".join(f'<div class="chip">{f}</div>' for f in filas) + "</div>"


PIE = limpiar("""<div class="foot">Predictor Fútbol Pro · Red neuronal residual + Random Forest + simulación Monte Carlo ·
5 grandes ligas europeas · Proyecto académico, no constituye asesoría de apuestas</div>""")
