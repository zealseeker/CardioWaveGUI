# Copyright (C) 2021 by University of Cambridge

# This software and algorithm was developed as part of the Cambridge Alliance
# for Medicines Safety (CAMS) initiative, funded by AstraZeneca and
# GlaxoSmithKline

# This program is made available under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the
# License, or at your option, any later version.
import sys
import os
import logging
import ctypes
import pickle
import webbrowser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QDir, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QDirModel, QTableWidgetItem, QGraphicsScene
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from cdwave import data, hillcurve, fnc
from cdwave.fnc import Waveform
from cdwave import __version__ as cdwave_version

from CardioWaveGUI import __version__
from CardioWaveGUI import draw, plateview, config

logger = logging.getLogger(__name__)
Form, Window = uic.loadUiType(os.path.join(config.root, "main.ui"))
logger.setLevel('INFO')
DEBUG = False


class ProcessWindow(QtWidgets.QDialog):
    str_signal = pyqtSignal(str)
    list_signal = pyqtSignal(list)

    def __init__(self, parent):
        super(ProcessWindow, self).__init__(parent)
        uic.loadUi(os.path.join(config.root, 'process_dialog.ui'), self)
        self.pushButton.clicked.connect(self.close)
        self.total = 0
        self.i = 0
        self.status = 0
        self.process_steps = [10, 30, 100]
        self.process_lenth = [20, 60]
        self.list_signal.connect(self.set_main_process)
        self.str_signal.connect(self.add_text_line)

    def process_wrapper(self, status=-1, total=0):
        self.list_signal.emit([status, total])

    def text_wrapper(self, text):
        self.str_signal.emit(text)

    def set_main_process(self, args):
        status = args[0]
        total = args[1]
        if status == 0 and total != 0:
            self.progressBar.setValue(self.process_steps[self.status])
            self.total = total
            self.i = 0
        elif status == 1:
            self.i += 1
            sub_ratio = self.i / self.total
            self.progressBar_2.setValue(sub_ratio * 100)
            self.progressBar.setValue(
                self.process_steps[self.status] + sub_ratio * self.process_lenth[self.status])
        elif status == 2:
            self.status += 1
            self.progressBar_2.setValue(100)
            self.progressBar.setValue(self.process_steps[self.status])
        if self.status == 2:
            self.pushButton.setEnabled(True)

    def add_text(self, text):
        self.plainTextEdit.insertPlainText(text)

    def add_text_line(self, text: str):
        line = text + '\n'
        self.add_text(line)

    def set_sub_process(self, n):
        self.progressBar_2.setValue(n)


class NewForm(Form):
    def __init__(self):
        self.fixed_columns = set()
        self.filters = {}
        self.filter_column_items = []
        self.waveform_dict = {}
        self.item_table_index = {}
        self.config = config.Configure()
        self.waveform: Waveform = None
        self.selecting_parameter = None
        self.core_data: data.Dataset = None
        self.browse_dir = None
        self.window = None
        self.scene1 = None
        self.scene2 = None
        self.scene3 = None
        self.figure1 = None
        self.figure2 = None
        self.figure3 = None
        self.canvas1 = None
        self.canvas2 = None
        self.canvas3 = None
        self.toolbar1 = None
        self.toolbar2 = None
        super().__init__()

    def connections(self, window):
        self.window = window
        self.pushButton.clicked.connect(self.show_dir)
        self.treeView.clicked.connect(self.select_file)
        self.tableWidget.clicked.connect(self.filter_table_click)
        self.tableWidget_2.clicked.connect(self.item_table_click)
        self.table2.clicked.connect(self.parameter_table_click)
        self.bt_fix.clicked.connect(self.fit_curve)
        self.bt_add_filter.clicked.connect(self.add_filter_from_list)
        self.bt_add_filter_2.clicked.connect(self.add_filter_from_table)
        self.bt_delete_filter.clicked.connect(self.delete_filter)
        self.bt_select.clicked.connect(self.select_waveforms)
        self.bt_select_2.clicked.connect(self.select_waveforms_quick)
        self.bt_quick_search.clicked.connect(self.quick_search)
        self.lineEdit_2.textEdited.connect(self.event_search_filter_values)
        self.scene1 = QGraphicsScene(self.graphicsView)
        self.scene2 = QGraphicsScene(self.gview)
        self.scene3 = QGraphicsScene(self.graphicsView_2)
        self.graphicsView.setScene(self.scene1)
        self.gview.setScene(self.scene2)
        self.graphicsView_2.setScene(self.scene3)
        self.figure1 = plt.figure(figsize=(4, 3), dpi=80)
        self.figure2 = plt.figure(figsize=(4, 3), dpi=80)
        self.figure3 = plt.figure(figsize=(4, 3), dpi=80)
        self.canvas1 = FigureCanvas(self.figure1)
        self.canvas2 = FigureCanvas(self.figure2)
        self.canvas3 = FigureCanvas(self.figure3)
        self.canvas1.setGeometry(
            0, 0, self.graphicsView.width(), self.graphicsView.height())
        self.canvas2.setGeometry(0, 0, self.gview.width(), self.gview.height())
        self.canvas3.setGeometry(
            0, 0, self.graphicsView_2.width(), self.graphicsView_2.height())
        self.toolbar1 = NavigationToolbar(self.canvas1, self.widget)
        self.toolbar2 = NavigationToolbar(self.canvas2, self.widget_2)
        self.scene1.addWidget(self.canvas1)
        self.scene2.addWidget(self.canvas2)
        self.scene3.addWidget(self.canvas3)
        self.actionThis_waveform.triggered.connect(self.export_waveform)
        self.action_fft.triggered.connect(self.fft_transform)
        self.action_welch.triggered.connect(self.welch_transform)
        self.comboBox.activated.connect(self.activate_filter_list)
        self.comboBox_2.activated.connect(self.activate_filter_table)
        self.comboBox_3.activated.connect(self.activate_item_table)
        self.window.closeEvent = self.closeEvent
        # Menu
        self.actionThis_waveform.triggered.connect(self.export_waveform)
        self.actionParameters.triggered.connect(self.export_parameters)
        self.actionBug_Report.triggered.connect(self.bug_report)
        self.actionAbout.triggered.connect(self.about_page)
        self.actionPlate_Viewer.triggered.connect(self.show_plate_viewer)
        self.actionOpen.triggered.connect(self.open_file)
        self.actionFrom_CSV_file.triggered.connect(self.import_file)
        if 'path' in self.config.all_config['general']:
            self.lineEdit.setText(self.config.all_config['general']['path'])

    def closeEvent(self, _):
        self.config.save(self)

    def select_file(self, signal):
        file_path = self.treeView.model().filePath(signal)
        if self.treeView.model().isDir(signal):
            self.browse_dir = file_path
            self.lineEdit.setText(file_path)
        else:
            self.selected_file(file_path)

    def selected_file(self, file_path):
        try:
            dataset = data.Dataset.loaddata(file_path)
            if not dataset:
                return
        except (ValueError, AttributeError, ModuleNotFoundError, pickle.PickleError) as err:
            if DEBUG:
                raise Exception(
                    'Cannot open the file {}'.format(file_path)) from err
            QMessageBox(QMessageBox.Critical, "Cannot open the file",
                        "Please check if you selected a correct file",
                        QMessageBox.Ok).exec_()
            return
        self.core_data = dataset
        self.config.load(file_path, self)
        self.comboBox.clear()
        self.comboBox.addItems(dataset.filterable_columns)
        self.comboBox.setCurrentIndex(1)
        self.comboBox_2.clear()
        self.comboBox_2.addItems(dataset.filterable_columns)
        self.activate_filter_list(1)

    def show_dir(self):
        model = QDirModel()
        model.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)
        model.setNameFilters(["*.pickle", '*.pickle.gz'])
        _dir = self.lineEdit.text()
        if not _dir:
            _dir = '.'
        self.treeView.setModel(model)
        self.treeView.setRootIndex(model.index(_dir))

    def filter_table_click(self, signal):
        column = signal.data()
        if not column:
            return None
        temp_filters = self.filters.copy()
        temp_filters[self.comboBox_2.currentText()] = column
        df = self.core_data.filter_by_filters(temp_filters, replace=False)
        self.draw_waveforms(df)
        self.show_item_table(df)

    def item_table_click(self, signal):
        item = signal.data()
        if not item:
            return None
        if item not in self.waveform_dict:
            logger.error("Cannot get the waveform")
            return None
        waveform = self.waveform_dict[item]
        index = self.item_table_index[item]
        self.window.statusBar().showMessage("waveform id: {}".format(index))
        self.select_one_waveform(waveform)

    def parameter_table_click(self, signal):
        item = signal.data()
        if not item:
            return None
        parameter_dict = {'Freq': 'freq', 'R/D': 'rd_ratio', 'Lambda': 'avg_lambda',
                          'Peaks': 'n_peak', 'Up': 'up_length', 'Max': 'maximum',
                          'L-std': 'std_lambda', 'A-std': 'std_amplitude',
                          'PW10': 'PW10_mean', 'stdPW10': 'PW10_std',
                          'PW25': 'PW25_mean', 'stdPW25': 'PW25_std',
                          'PW50': 'PW50_mean', 'stdPW50': 'PW50_std',
                          'PW80': 'PW80_mean', 'stdPW80': 'PW80_std',
                          'PW90': 'PW90_mean', 'stdPW90': 'PW90_std'}
        if item in parameter_dict:
            p = parameter_dict[item]
            self.selecting_parameter = p
        else:
            return False
        self.fit_curve(None)

    def fit_curve(self, _):
        f = self.figure2
        f.clear()
        ax = f.add_subplot(111)
        if self.selecting_parameter is None:
            self.selecting_parameter = 'freq'
        p = self.selecting_parameter
        df = self.core_data.filtered_df.copy()
        df[p] = [x.get_parameters()[p] for x in df.waveform]
        try:
            # popt, perr, curve = hillcurve.fit_parameter(df, p)
            conc, resp = df.concentration, df[p]
            curves = [hillcurve.TCPLHill(conc, resp),
                      hillcurve.TCPLGainLoss(conc, resp),
                      hillcurve.TCPLPlain(conc, resp)]
            # self.window.statusBar().showMessage('{}, {}'.format(popt, perr))
            draw.plot_tcpl_curves(curves, ax=ax, ylabel=p)
        except ValueError as e:
            logger.warning("Cannot fit the curve, %s", str(e))
        except RuntimeError as e:
            logger.warning("Cannot fit the curve, %s", str(e))
        self.canvas2.draw()

    def welch_transform(self, _):
        if self.waveform is None:
            return False
        signals = self.waveform.resample()
        frq, psd = fnc.wave_transform(signals, method='welch')
        # Frequency should not be higher than 1, so we pick only first 100
        frq = frq[:100]
        psd = psd[:100]
        self.waveform.draw_series(self.figure2, pd.Series(psd, index=frq))
        self.canvas2.draw()
        return True

    def fft_transform(self, _):
        if self.waveform is None:
            return None
        signals = self.waveform.resample()
        frq, psd = fnc.wave_transform(signals, method='fft')
        # Frequency should not be higher than 1, so we pick only first 100
        frq = frq[:100]
        psd = psd[:100]
        ax = self.waveform.draw_series(self.figure2, pd.Series(psd, index=frq))
        ax.text(0.7, 0.2, 'FFT Ratio: {:.2f}'.format(
            self.waveform.calc_fft_freq_ratio()), transform=ax.transAxes)
        self.canvas2.draw()

    def test(self):
        self.transfer_gsk('data')

    def export_waveform(self):
        if self.waveform:
            default_directory = self.config.all_config['general'].get(
                'export_directory', '.')
            fname, _ = QFileDialog.getSaveFileName(self.window, 'Save waveform',
                                                   default_directory, "Text File (*.csv)")
            if not fname:
                return None
            self.config.all_config['general']['export_directory'] = os.path.dirname(
                fname)
            try:
                self.waveform.export(fname)
            except PermissionError:
                self.window.statusBar().showMessage("Permission denied")

    def activate_filter_list(self, index):
        column = self.core_data.filterable_columns[index]
        items = set(self.core_data.filtered_df[column].astype(str))
        items = [str(x)for x in sorted(items)]
        self.filter_column_items = items
        self.listWidget.clear()
        self.listWidget.addItems(items)

    def activate_filter_table(self, _):
        self.show_filter_table()

    def show_filter_table(self):
        column = self.comboBox_2.currentText()
        temp_filters = self.filters.copy()
        if column in temp_filters:
            del temp_filters[column]
        df = self.core_data.filter_by_filters(temp_filters, False)
        items = set(df[column].astype('str'))
        items = [str(x)for x in sorted(items)]
        self.tableWidget.clear()
        for i, column in enumerate(items):
            col = i % self.tableWidget.columnCount()
            row = i // self.tableWidget.columnCount()
            table_widget_item = QTableWidgetItem(column)
            self.tableWidget.setRowCount(row + 1)
            self.tableWidget.setItem(row, col, table_widget_item)

    def add_filter(self, column, value):
        filter_name = '{}={}'.format(column, value)
        if column in self.filters:
            self.listWidget_2.clear()
            del self.filters[column]
            for item in self.filters:
                self.listWidget_2.addItem(
                    "{}={}".format(item, self.filters[item]))
        if self.core_data.dtypes[column] == pd.np.float64:
            value = float(value)
        self.filters[column] = value
        self.listWidget_2.addItem(filter_name)
        self.core_data.filter_by_filters(self.filters)

    def add_filter_from_list(self):
        column = self.comboBox.currentText()
        if self.listWidget.currentItem() is None:
            return
        value = self.listWidget.currentItem().text()
        self.add_filter(column, value)

    def add_filter_from_table(self):
        column = self.comboBox_2.currentText()
        if self.tableWidget.currentItem() is not None:
            value = self.tableWidget.currentItem().text()
            self.add_filter(column, value)

    def event_search_filter_values(self, text):
        items = self.filter_column_items
        new_items = []
        for x in items:
            if text in x:
                new_items.append(x)
        self.listWidget.clear()
        self.listWidget.addItems(new_items)

    def load_filters(self, filters):
        self.filters = filters
        self.listWidget_2.clear()
        for item in filters:
            filter_name = '{}={}'.format(item, filters[item])
            self.listWidget_2.addItem(filter_name)
        self.core_data.filter_by_filters(self.filters)

    def delete_filter(self):
        item = self.listWidget_2.currentItem()
        if not item:
            return
        filter_name = item.text()
        row = self.listWidget_2.row(item)
        column = filter_name.split('=')[0]
        del self.filters[column]
        self.listWidget_2.takeItem(row)
        self.core_data.filter_by_filters(self.filters)
        self.update_tables()
        if column == self.comboBox.currentText():
            self.activate_filter_list(self.comboBox.currentIndex())

    def select_waveforms_quick(self):
        filters_text = self.plainTextEdit.toPlainText()
        if len(filters_text) != 0:
            lines = []
            for line in filters_text.split('\n'):
                items = line.split('=')
                self.filters[items[0]] = items[1]
                lines.append(line)
            self.listWidget_2.clear()
            self.listWidget_2.addItems(lines)
            self.core_data.filter_by_filters(self.filters)
        self.draw_waveforms()
        self.update_tables()

    def select_waveforms(self):
        if not self.core_data:
            logger.warning("No core_data, please load the file")
            return
        self.draw_waveforms(self.core_data.filtered_df)
        self.update_tables()

    def draw_waveforms(self, dataframe=None):
        if dataframe is None:
            dataframe = self.core_data.filtered_df
        logging.debug('Len(df) = %d', len(dataframe))
        span = 100 if self.checkBox_2.isChecked() else 0
        draw.draw_multiple(self.figure3, dataframe, span)
        self.canvas3.draw()

    def select_one_waveform(self, waveform: data.WaveformFull):
        series = waveform.get_signal_series()
        self.waveform = Waveform(series)
        r = waveform.get_parameters()
        if self.checkBox.isChecked() and self.waveform.get_peaks():
            self.waveform.analyse()
            self.waveform.draw_status(self.figure2)
            r = self.waveform.get_parameters()
        if r:
            properties = [
                (0, 1, 'maximum'),
                (0, 3, 'avg_valley'),
                (1, 1, 'n_peak'),
                (1, 3, 'avg_lambda'),
                (2, 1, 'freq'),
                (2, 3, 'std_lambda'),
                (3, 1, 'up_length'),
                (3, 3, 'full_down'),
                (4, 1, 'max_combo_peaks'),
                (4, 3, 'down_length'),
                (5, 1, 'rms'),
                (5, 3, 'uniform'),
                (6, 1, 'avg_amplitude'),
                (6, 3, 'std_amplitude'),
                (7, 1, 'avg_tail'),
                (7, 2, 'std_tail'),
                (7, 3, 'tail_proportion'),
                (8, 1, 'avg_shoulder'),
                (8, 3, 'avg_shoulder_tail'),
                (9, 1, 'PW10_mean'),
                (9, 3, 'PW10_std'),
                (10, 1, 'PW25_mean'),
                (10, 3, 'PW25_std'),
                (11, 1, 'PW50_mean'),
                (11, 3, 'PW50_std'),
                (12, 1, 'PW80_mean'),
                (12, 3, 'PW80_std'),
                (13, 1, 'PW90_mean'),
                (13, 3, 'PW90_std')
            ]
            for prop in properties:
                p = r[prop[2]]
                if isinstance(p, np.floating):
                    item = '{:.3f}'.format(p)
                else:
                    item = str(p)
                self.table2.setItem(prop[0], prop[1], QTableWidgetItem(item))
        self.waveform.draw_series(self.figure1)
        self.canvas1.draw()
        self.canvas2.draw()

    def activate_item_table(self, _):
        self.update_tables()

    def show_item_table(self, dataframe: pd.DataFrame = None):
        column = self.comboBox_3.currentText()
        df = dataframe if dataframe is not None else self.core_data.filtered_df
        df = df.reset_index()
        if len(df) > 8:
            self.window.statusBar().showMessage("More than 8 items in the table")
            logger.warning("More than 8 items in the table")
            df = df.iloc[:8]
        else:
            self.window.statusBar().showMessage("")
        self.waveform_dict = {}
        self.item_table_index = {}
        for i, row in df.iterrows():
            if isinstance(row[column], float):
                text = "{:.3f}".format(row[column])
            else:
                text = str(row[column])
            self.waveform_dict[text] = row['waveform']
            self.item_table_index[text] = row['index']
            t_col = i % self.tableWidget_2.columnCount()
            t_row = i // self.tableWidget_2.columnCount()
            tableWidgetItem = QTableWidgetItem(text)
            self.tableWidget_2.setRowCount(t_row + 1)
            self.tableWidget_2.setItem(t_row, t_col, tableWidgetItem)

    def update_tables(self):
        self.show_filter_table()
        self.show_item_table()

    def quick_search(self):
        if 'default_waveform' in self.config.crt_config:
            default = self.config.crt_config['default_waveform']
        else:
            default = 0
        num, ok = QInputDialog.getInt(
            self.window, "Select a waveform", "enter a number", value=default)
        if ok:
            if num < 0 or (self.core_data is None) or num >= len(self.core_data.waveforms):
                self.window.statusBar().showMessage("waveform {} does not exist".format(num))
            else:
                waveform = self.core_data.waveforms[num]
                self.filters = {}
                lines = []
                for field in ['compound', 'well', 'plate', 'state']:
                    self.filters[field] = waveform.profile[field]
                    lines.append('{}={}'.format(
                        field, waveform.profile[field]))
                self.listWidget_2.clear()
                self.listWidget_2.addItems(lines)
                self.core_data.filter_by_filters(self.filters)
                self.draw_waveforms()
                self.update_tables()
                self.config.crt_config['default_waveform'] = num

    # menu
    def bug_report(self):
        webbrowser.open_new('https://github.com/zealseeker/CardioWave/issues')

    def about_page(self):
        text = """CardioWave is a tool for waveform analysis

        URL: http://https://github.com/zealseeker/CardioWave/
        Author: Hongbin Yang
        CardioWave Version: {}
        GUI version: {}
        """.format(cdwave_version, __version__)
        QMessageBox(QMessageBox.Information, "CardioWave - viewer",
                    text,
                    QMessageBox.Ok).exec_()

    def export_parameters(self):
        default_directory = self.config.all_config['general'].get(
            'export_directory', '.')
        fname, _ = QFileDialog.getSaveFileName(self.window, 'Save waveform',
                                               default_directory, "Text File (*.csv)")
        if not fname:
            return None
        self.config.all_config['general']['export_directory'] = os.path.dirname(
            fname)
        dataset = self.core_data
        df = dataset.get_parameter_df()
        df.to_csv(fname)

    def open_file(self):
        default_directory = self.config.all_config['general'].get(
            'path', '.')
        fname, _ = QFileDialog.getOpenFileName(
            self.window, 'Open a wavefom file', default_directory, "Gzip File (*.pickle.gz)")
        self.selected_file(fname)

    def import_file(self):
        default_directory = self.config.all_config['general'].get(
            'path', '.')
        fname, _ = QFileDialog.getOpenFileName(
            self.window, 'Open a wavefom file', default_directory, "CSV File (*.csv)")
        oname, _ = QFileDialog.getSaveFileName(
            self.window, 'Save waveform data to', default_directory, "(*.*)"
        )
        if not oname:
            return False
        dialog = ProcessWindow(self.window)
        dialog.show()
        dialog.process_wrapper(0, 1)
        dialog.add_text_line('Loading ...')
        try:
            dataset = data.StandardCSVLoader(fname).transfer()
        except Exception as e:
            dialog.add_text_line(e)
            return False
        dialog.process_wrapper(2, 1)
        dialog.add_text_line('Saving ...')
        dataset.save(oname)
        dialog.add_text_line('Done!')
        dialog.process_wrapper(2, 1) # Repeat because it does not calculate parameters

    def show_plate_viewer(self):
        if self.core_data is None:
            QMessageBox(QMessageBox.Warning, "Error",
                        "Select the datset first.", QMessageBox.Ok).exec_()
            return
        plateviewer = plateview.PlateViewer(self, self.window, self.core_data)
        plateviewer.show()


def main():
    if sys.platform == 'win32':
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            'CardioWaveGUI')
    app = QApplication([])
    icon = QIcon(os.path.join(config.root, 'ico.png'))
    app.setWindowIcon(icon)
    app.setApplicationName('CardioWave')
    _window = Window()
    _window.setWindowIcon(icon)
    form = NewForm()
    form.setupUi(_window)
    form.connections(_window)
    form.show_dir()
    _window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    DEBUG = True
    logger.setLevel('DEBUG')
    main()
