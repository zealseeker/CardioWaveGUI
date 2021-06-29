# Copyright (C) 2021 by University of Cambridge

# This software and algorithm was developed as part of the Cambridge Alliance
# for Medicines Safety (CAMS) initiative, funded by AstraZeneca and
# GlaxoSmithKline

# This program is made available under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the
# License, or at your option, any later version.
from typing import List
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

def plot_tcpl_curves(curves: List[TCPL], density=0.1, ax:Axes=None, ylabel=None):
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

def draw_multiple(figure, df, span=0):
    def make_slice(array):
        if span > 0:
            return array[:span]
        else:
            return array
    n = len(df)  # Only support 8 now
    if n > 8:
        df = df.iloc[:8]
    figure.clear()
    axes = figure.subplots(2, 4, sharex='all', sharey='all')
    for i, row in df.reset_index().iterrows():
        ax = axes[i//4][i % 4]
        ax.plot(make_slice(row.signal['x']), make_slice(row.signal['y']), '-')
        ax.set_title('{:.3f}'.format(row['concentration']))
