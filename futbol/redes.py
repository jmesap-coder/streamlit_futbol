"""Inferencia de las redes neuronales con NumPy puro (sin PyTorch).

La arquitectura es la del notebook (RedNeuronalFutbol): entrada -> bloque residual -> mid -> salida,
con Linear + BatchNorm1d + ReLU. En modo evaluacion el Dropout no hace nada y BatchNorm usa sus
estadisticas acumuladas, asi que basta con multiplicaciones de matrices. La celda de exportacion del
notebook comprueba que esta version da las mismas probabilidades que PyTorch antes de generar el .zip.
"""
import numpy as np

BN_EPS = 1e-5   # valor por defecto de nn.BatchNorm1d


def _lin(x, p, k):
    return x @ p[f"{k}.weight"].T + p[f"{k}.bias"]


def _bn(x, p, k):
    return (x - p[f"{k}.running_mean"]) / np.sqrt(p[f"{k}.running_var"] + BN_EPS) * p[f"{k}.weight"] + p[f"{k}.bias"]


def _relu(x):
    return np.maximum(x, 0.0)


def _forward(x, p):
    """x: (n, n_features) ya escalado -> probabilidades (n, 3) de UNA red."""
    h = _relu(_bn(_lin(x, p, "entrada.0"), p, "entrada.1"))
    b = _bn(_lin(_relu(_bn(_lin(h, p, "res1.block.0"), p, "res1.block.1")), p, "res1.block.4"), p, "res1.block.5")
    h = _relu(h + b)                                                       # conexion residual
    h = _relu(_bn(_lin(h, p, "mid.0"), p, "mid.1"))
    h = _relu(_bn(_lin(h, p, "mid.4"), p, "mid.5"))
    z = _lin(h, p, "salida")
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class RedesNumpy:
    """Ensemble de redes cargado desde un .npz (claves 'm<i>__<nombre_del_parametro>')."""

    def __init__(self, ruta=None):
        self.modelos = []
        if ruta is not None:
            datos = np.load(ruta)
            n = 1 + max(int(k.split("__")[0][1:]) for k in datos.files)
            for i in range(n):
                pref = f"m{i}__"
                self.modelos.append({k[len(pref):]: datos[k].astype(np.float64)
                                     for k in datos.files if k.startswith(pref)})

    def __len__(self):
        return len(self.modelos)

    def probs(self, x):
        """(n_redes, n, 3): probabilidades de cada red."""
        x = np.asarray(x, dtype=np.float64)
        return np.stack([_forward(x, p) for p in self.modelos])
