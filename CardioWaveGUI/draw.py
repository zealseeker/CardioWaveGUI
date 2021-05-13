import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from cdwave.hillcurve import HillCurve, fsigmoid, TCPL


def plot_hillcurve(curve: HillCurve, density=0.1, ax:Axes=None, ylabel=None):
    if ax == None:
        ax = plt.subplots()
    cc = np.arange(curve.c_min, curve.c_max, density)
    vsigmoid = np.vectorize(lambda x: fsigmoid(
        x, *curve.popt) * curve.p_diff + curve.p_min)
    ax.plot(cc, vsigmoid(cc))
    ax.plot(curve.concentrations, curve.responses, 'o', label="normal")
    ax.text(0.7, 0.2, 'EC50: {:.2f}'.format(curve.EC50), transform=ax.transAxes)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax

def plot_tcpl_curves(curves: "list[TCPL]", density=0.1, ax:Axes=None, ylabel=None):
    if ax == None:
        ax = plt.subplots()
    curve = curves[0]
    cc = np.arange(curve.c_min, curve.c_max, density)
    for curve in curves:
        fnc = np.vectorize(curve.predict)
        label = '{}-{:.2f}'.format(type(curve).__name__, curve.RMSD)
        ax.plot(cc, fnc(cc), label=label)
    ax.text(0.7, 0.2, 'EC50: {:.2f}'.format(curves[0].EC50), transform=ax.transAxes)
    ax.plot(curve.concentrations, curve.responses, 'o', label="normal")
    ax.legend()
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax
