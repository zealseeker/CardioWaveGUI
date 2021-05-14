import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import Qt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap, Normalize
from matplotlib.cm import get_cmap
from cdwave.data import WaveformFull, Dataset
from CardioWaveGUI import config

COLUMNS = list('ABCDEFGHIJKLMNOP')


class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data: pd.DataFrame, cmap: Colormap):
        super().__init__()
        self._data = data
        self.max = data.max().max()
        self.min = data.min().min()
        self.normalize = Normalize(self.min, self.max)
        self.cmap = cmap
        self.generate_colors()

    def value_to_color(self, value):
        if not pd.isna(value) and self.cmap is not None:
            color = self.cmap(self.normalize(value))
            qcolor = QtGui.QColor()
            qcolor.setRgbF(*color)
            return qcolor
        else:
            return None

    def generate_colors(self):
        df = self._data
        cdf = df.copy()
        for col in cdf.columns:
            cdf[col] = df[col].apply(self.value_to_color)
        self.colors = cdf

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if pd.isna(value):
                return ''
            return '{:.1f}'.format(value)

        if role == Qt.BackgroundRole and self.cmap is not None:
            return self.colors.iloc[index.row(), index.column()]

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, parent) -> int:
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Vertical:
                return str(self._data.index[section])


class PlateViewer(QtWidgets.QDialog):
    display_parameters = ['n_peak', 'avg_amplitude', 'freq']
    color_maps = ['None', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                  'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                  'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                  'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                  'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

    def __init__(self, main_form, parent, dataset: Dataset):
        super(PlateViewer, self).__init__(parent)
        self.parameterCombo: QtWidgets.QComboBox = None
        uic.loadUi(os.path.join(config.root, 'plateviewer.ui'), self)
        self.dataset = dataset
        self.df = dataset.get_parameter_df()
        self.main_form = main_form
        self.platedf = None
        self.show_plates()
        self.show_parameters()
        self.show_state()
        self.show_cmaps()
        self.plateCombo.activated.connect(self.show_plate)
        self.stateCombo.activated.connect(self.show_plate)
        self.parameterCombo.activated.connect(self.show_plate)
        self.colorCombo.activated.connect(self.show_plate)
        self.tableView.clicked.connect(self.select_waveform)
        self.show_plate()

    def show_plate(self):
        plate = self.plateCombo.currentText()
        state = self.stateCombo.currentText()
        parameter = self.display_parameters[self.parameterCombo.currentIndex()]
        df = self.df
        df = df[(df['plate'] == plate) & (
            df['state'] == state)].set_index('well')
        self.platedf = df
        columns = COLUMNS
        dfdata = {}
        for col in columns:
            dfdata[col] = []
            for i in range(1, 25):
                well = col + str(i)
                if well in df.index:
                    dfdata[col].append(df.loc[well, parameter])
                else:
                    dfdata[col].append(np.nan)
        df = pd.DataFrame(dfdata, index=list(range(1, 25))).transpose()
        cmap_text = self.colorCombo.currentText()
        if cmap_text == 'None':
            cmap = None
        else:
            cmap = get_cmap(cmap_text)
        self.model = TableModel(df, cmap)
        self.tableView.setModel(self.model)

    def show_plates(self):
        plates = self.df.plate.unique()
        self.plateCombo.addItems(plates)
        self.plateCombo.setCurrentIndex(0)

    def show_state(self):
        states = self.df.state.unique()
        self.stateCombo.addItems(states)
        self.stateCombo.setCurrentIndex(0)

    def show_parameters(self):
        self.parameterCombo.addItems(
            WaveformFull.parameter_df.loc[self.display_parameters, 'Name'].values)
        self.parameterCombo.setCurrentIndex(0)

    def show_cmaps(self):
        self.colorCombo.addItems(self.color_maps)
        self.colorCombo.setCurrentIndex(4)

    def select_waveform(self, signal: QtCore.QModelIndex):
        if self.platedf is None:
            return
        row, col = signal.row(), signal.column()
        well = COLUMNS[row] + str(col+1)
        if well in self.platedf.index:
            waveform = self.platedf.loc[well, 'waveform']
            self.main_form.select_one_waveform(waveform)
