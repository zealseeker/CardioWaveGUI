import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from cdwave.hillcurve import HillCurve, fsigmoid


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
