# ⚽ Predicción de partidos de clubes (Streamlit)

App web que predice partidos de las 5 grandes ligas europeas con un ensemble de **redes neuronales + Random Forest**
y simulación **Monte Carlo** (probabilidades con incertidumbre, y mini ligas con probabilidad de campeón).

## Estructura

```
streamlit_futbol/
├── app.py                  # interfaz de Streamlit
├── requirements.txt        # dependencias (lo reemplaza el .zip con las versiones exactas de Colab)
├── futbol/
│   ├── features.py         # variables de cada partido (copiadas del notebook)
│   ├── redes.py            # inferencia de las redes con NumPy (no requiere PyTorch)
│   ├── predictor.py        # carga de modelos + Monte Carlo
│   └── ui.py               # estilos (tema oscuro) y bloques HTML
├── modelos/                # artefactos entrenados (los genera el notebook)
└── .streamlit/config.toml
```

## Cómo publicarla

1. **Entrena y exporta.** Ejecuta `Redes_Neuronales_Clubes_2026_IA_v2.ipynb` hasta la última celda (*Exportar artefactos*).
   La celda verifica que la versión NumPy de las redes coincide con PyTorch y descarga `modelos_streamlit.zip`.
2. **Descomprime el zip en la raíz de este repositorio.** Se llenan `modelos/` y se reemplazan `requirements.txt`
   y `futbol/redes.py` (mismo contenido que el del repositorio, ya verificado).
3. **Sube a GitHub:**
   ```bash
   git init
   git add .
   git commit -m "App de prediccion de partidos"
   git branch -M main
   git remote add origin https://github.com/<tu-usuario>/<tu-repo>.git
   git push -u origin main
   ```
4. **Despliega en Streamlit Community Cloud** (share.streamlit.io): *Create app* → elige el repositorio,
   rama `main` y *Main file path* = `app.py`. En *Advanced settings* elige la misma versión de Python
   que imprime la celda de exportación (Colab).

## Probar en tu computador

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notas

- Las versiones de `scikit-learn` y `numpy` deben coincidir con las de Colab (el `requirements.txt` del zip ya las fija),
  porque `rf.joblib` depende de la versión de scikit-learn.
- GitHub rechaza archivos de más de 100 MB: la celda de exportación reduce el número de árboles del Random Forest
  si `rf.joblib` pasara de 80 MB.
- Para actualizar los datos, vuelve a entrenar en el notebook, exporta de nuevo y reemplaza `modelos/`.
