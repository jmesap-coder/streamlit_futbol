"""Features de cada partido. Codigo copiado tal cual del notebook (sin el calculo de ELO,
que en la app se carga ya calculado desde modelos/elos.json)."""
import numpy as np
import pandas as pd

LABELS = {0: 'Local gana', 1: 'Empate', 2: 'Visitante gana'}
FEATURES = [

    'elo_local', 'elo_visitante', 'diferencia_elo',
    'diferencia_elo_abs', 'elo_promedio',
    # Desempeno Local
    'victorias_ultimos_5_local', 'victorias_ultimos_10_local',
    'promedio_goles_ultimos_5_local', 'promedio_goles_ultimos_10_local',
    'promedio_goles_recibidos_ultimos_5_local', 'promedio_goles_recibidos_ultimos_10_local',
    'empates_ultimos_5_local',
    # Desempeno visitante
    'empates_ultimos_5_visit',
    'victorias_ultimos_5_visit', 'victorias_ultimos_10_visit',
    'promedio_goles_ultimos_5_visit',
    'promedio_goles_recibidos_ultimos_5_visit',
    # Diferencias
    'diferencia_victorias_5', 'diferencia_goles_5',
    # Contexto
    'head_to_head_ultimos_5',

    'forma_ponderada_local',
    'forma_ponderada_visit',
    'racha_victorias_local',
    'racha_victorias_visit',
    'goles_favor_vs_contra_local',
    'goles_favor_vs_contra_visit',
]
# Nota: TODAS las features _local se calculan solo con partidos que ese
# equipo jugo de LOCAL, y todas las _visit solo con partidos que jugo de
# VISITANTE (ver get_team_stats(..., solo_contexto=...) mas abajo). Antes
# se mezclaban ambas condiciones en un mismo promedio.
# Nota: se quitaron 'neutral', 'tipo_torneo_copa_mundo', 'tipo_torneo_eliminatoria'
# y 'experiencia_mundial' porque eran especificas del formato selecciones/Mundial
# (el dataset de clubes no tiene sedes neutrales ni fases de mundial).


#Estadisticas historicas por equipo
def get_team_stats(team, before_date, df, n5=5, n10=10, solo_contexto=None):
    """
    solo_contexto:
      None         -> usa los ultimos partidos del equipo sin importar
                       si jugo de local o de visitante (comportamiento
                       original, se sigue usando para el resto de features)
      'local'      -> solo partidos que el equipo jugo de LOCAL
      'visitante'  -> solo partidos que el equipo jugo de VISITANTE
    Todas las stats "_local" de construir_features/cell 6 usan
    solo_contexto='local', y todas las "_visit" usan 'visitante': el
    rendimiento de un equipo jugando en su cancha no es el mismo que
    jugando afuera, y mezclarlas perdia esa distincion.
    """
    base = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < before_date)
    ]
    if solo_contexto == 'local':
        base = base[base['home_team'] == team]
    elif solo_contexto == 'visitante':
        base = base[base['away_team'] == team]

    past = base.tail(n10)
    if len(past) == 0:
        return dict(
            victorias_ultimos_5=2.0, victorias_ultimos_10=4.0,
            promedio_goles_ultimos_5=1.3, promedio_goles_ultimos_10=1.3,
            promedio_goles_recibidos_ultimos_5=1.2,
            promedio_goles_recibidos_ultimos_10=1.2,
            empates_ultimos_5=1.0,
            forma_ponderada=0.4, racha_victorias=0.0,
            goles_favor_vs_contra=1.0,
        )
    wins, draws, gf, gc = [], [], [], []
    for _, r in past.iterrows():
        if r['home_team'] == team: gs, ga = r['home_score'], r['away_score']
        else:                      gs, ga = r['away_score'], r['home_score']
        gf.append(gs); gc.append(ga)
        wins.append(1 if gs > ga else 0)
        draws.append(1 if gs == ga else 0)

    def sm(lst, n):
        chunk = lst[-n:]
        return float(np.mean(chunk)) if chunk else 0.0

    # Partidos recientes pesan mas
    w = np.exp(np.linspace(-1, 0, len(wins)))
    forma_pond = float(np.average(wins, weights=w)) if wins else 0.4

    racha = 0
    for w_ in reversed(wins):
        if w_ == 1: racha += 1
        else: break


    total_gf = sum(gf[-5:]) if gf else 1.0
    total_gc = sum(gc[-5:]) if gc else 1.0
    ratio = total_gf / (total_gc + 0.5)

    return dict(
        victorias_ultimos_5=sm(wins, n5), victorias_ultimos_10=sm(wins, n10),
        promedio_goles_ultimos_5=sm(gf, n5), promedio_goles_ultimos_10=sm(gf, n10),
        promedio_goles_recibidos_ultimos_5=sm(gc, n5),
        promedio_goles_recibidos_ultimos_10=sm(gc, n10),
        empates_ultimos_5=sm(draws, n5),
        forma_ponderada=forma_pond,
        racha_victorias=float(racha),
        goles_favor_vs_contra=float(ratio),
    )

def head_to_head(team_a, team_b, df, n=5):
    h2h = df[
        ((df['home_team']==team_a)&(df['away_team']==team_b)) |
        ((df['home_team']==team_b)&(df['away_team']==team_a))
    ].tail(n)
    if len(h2h) == 0: return float(n) / 2
    wins = sum(
        1 for _, r in h2h.iterrows()
        if (r['home_team']==team_a and r['home_score']>r['away_score']) or
           (r['away_team']==team_a and r['away_score']>r['home_score'])
    )
    return float(wins)


def construir_features(local, visitante, df, elos_o_elo_l,
                       elo_v=None, fecha_ref=None, fase='liga'):
    """Firma:
      construir_features(local, visitante, df_hist, elos_dict)      -> usa ultimo ELO conocido
      construir_features(h, a, df, elo_l, elo_v, fecha_ref=fecha)   -> ELO explicito (backtesting)
    """
    if fecha_ref is None: fecha_ref = df['date'].max()

    if isinstance(elos_o_elo_l, dict):
        elo_l  = elos_o_elo_l.get(local,     1500.0)
        elo_v_ = elos_o_elo_l.get(visitante, 1500.0)
    else:
        elo_l  = float(elos_o_elo_l)
        elo_v_ = float(elo_v) if elo_v is not None else 1500.0

    stats   = get_team_stats(local,     fecha_ref, df, solo_contexto='local')
    stats_v = get_team_stats(visitante, fecha_ref, df, solo_contexto='visitante')
    h2h     = head_to_head(local, visitante, df)

    return {
        'elo_local':                              elo_l,
        'elo_visitante':                          elo_v_,
        'diferencia_elo':                         elo_l - elo_v_,
        'diferencia_elo_abs':                     abs(elo_l - elo_v_),
        'elo_promedio':                           (elo_l + elo_v_) / 2,
        'victorias_ultimos_5_local':              stats['victorias_ultimos_5'],
        'victorias_ultimos_10_local':             stats['victorias_ultimos_10'],
        'promedio_goles_ultimos_5_local':         stats['promedio_goles_ultimos_5'],
        'promedio_goles_ultimos_10_local':        stats['promedio_goles_ultimos_10'],
        'promedio_goles_recibidos_ultimos_5_local':   stats['promedio_goles_recibidos_ultimos_5'],
        'promedio_goles_recibidos_ultimos_10_local':  stats['promedio_goles_recibidos_ultimos_10'],
        'empates_ultimos_5_local':                stats.get('empates_ultimos_5', 1.0),
        'empates_ultimos_5_visit':                stats_v.get('empates_ultimos_5', 1.0),
        'head_to_head_ultimos_5':                 h2h,
        'victorias_ultimos_5_visit':              stats_v['victorias_ultimos_5'],
        'victorias_ultimos_10_visit':             stats_v['victorias_ultimos_10'],
        'promedio_goles_ultimos_5_visit':         stats_v['promedio_goles_ultimos_5'],
        'promedio_goles_recibidos_ultimos_5_visit': stats_v['promedio_goles_recibidos_ultimos_5'],
        'diferencia_victorias_5':                 stats['victorias_ultimos_5'] - stats_v['victorias_ultimos_5'],
        'diferencia_goles_5':                     stats['promedio_goles_ultimos_5'] - stats_v['promedio_goles_ultimos_5'],
        'forma_ponderada_local':                  stats['forma_ponderada'],
        'forma_ponderada_visit':                  stats_v['forma_ponderada'],
        'racha_victorias_local':                  stats['racha_victorias'],
        'racha_victorias_visit':                  stats_v['racha_victorias'],
        'goles_favor_vs_contra_local':            stats['goles_favor_vs_contra'],
        'goles_favor_vs_contra_visit':            stats_v['goles_favor_vs_contra'],
    }
