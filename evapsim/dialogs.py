'''
Created on Jan 3, 2023

@author: adamt
'''
import logging
import sys

from PyQt5 import QtWidgets, QtGui
import qt_ui
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg,
                                                NavigationToolbar2QT)


log = logging.getLogger("evapsim")


class HelpDialog(QtWidgets.QDialog):
    '''Creates pop up window for help info
    '''
    def __init__(self):
        super().__init__()
        self.ui_help = qt_ui.Ui_Dialog_HelpTree()
        self.ui_help.setupUi(self)


class SimulatorWindow(object):
    '''
    '''

    def __init__(self, sim, window):
        self.Sim = sim
        self.window = window
        self.app = qt_ui.Ui_MainWindow()
        self.app.setupUi(window)

        self.help = HelpDialog()
        self.help.hide()

        self.ui_log()
        self.ui_graphs()
        self.ui_inputs()
        self.ui_buttons()
        self.ui_presets()

        self.app.pushButton_Pause.setDisabled(True)
        self.app.pushButton_Abort_Run.setDisabled(True)
        log.info("Application window init complete")

    def ui_graphs(self):
        '''
        '''
        self.graph_model = FigureCanvasQTAgg(self.Sim.graph_ani)
        self.layout_model = QtWidgets.QWidget(self.app.graphicsView_Model)
        self.grid_model = QtWidgets.QGridLayout(self.layout_model)
        self.grid_model.addWidget(self.graph_model)

        self.graph_evap_top = FigureCanvasQTAgg(self.Sim.graph_ray1)
        self.layout_top = QtWidgets.QWidget(
            self.app.graphicsView_Evap_Profile_Top)
        self.grid_top = QtWidgets.QGridLayout(self.layout_top)
        self.grid_top.addWidget(self.graph_evap_top)

        self.graph_evap_bot = FigureCanvasQTAgg(self.Sim.graph_ray2)
        self.layout_bot = QtWidgets.QWidget(
            self.app.graphicsView_Evap_Profile_Bottom)
        self.grid_bot = QtWidgets.QGridLayout(self.layout_bot)
        self.grid_bot.addWidget(self.graph_evap_bot)

        self.toolbar = NavigationToolbar2QT(self.graph_model,
                                            self.app.graphicsView_Model)
        self.grid_model.addWidget(self.toolbar)

        self.graph_model.draw_idle()
        self.graph_evap_top.draw_idle()
        self.graph_evap_bot.draw_idle()

    def ui_log(self):
        '''
        '''
        class QTextEditLogger(logging.Handler):
            '''
            '''
            def __init__(self, parent):
                super().__init__()
                self.widget = QtWidgets.QPlainTextEdit(parent)
                self.widget.setReadOnly(True)

            def emit(self, record):
                msg = self.format(record)
                self.widget.appendPlainText(msg)

        logTexBox = QTextEditLogger(self.window)
        logTexBox.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logging.getLogger().addHandler(logTexBox)
        logging.getLogger().setLevel(logging.DEBUG)

        self.layout_log = QtWidgets.QWidget(self.app.textBrowser_Log)
        self.layout_log.setGeometry(self.app.textBrowser_Log.geometry())
        self.grid_log = QtWidgets.QGridLayout(self.layout_log)
        self.grid_log.addWidget(logTexBox.widget)

    def ui_inputs(self):
        '''Setup inputs to take doubles only
        '''
        double = QtGui.QDoubleValidator()
        self.app.lineEdit_Evaporation_Rate.setValidator(double)
        self.app.lineEdit_Evaporation_Time.setValidator(double)
        self.app.lineEdit_Grid_Space.setValidator(double)
        self.app.lineEdit_Model_Limit.setValidator(double)
        self.app.lineEdit_Model_Resolution.setValidator(double)
        self.app.lineEdit_Raycast_Length.setValidator(double)
        self.app.lineEdit_epsGrid.setValidator(double)
        self.app.lineEdit_epsIntersect.setValidator(double)
        self.app.lineEdit_epsMerge.setValidator(double)
        self.app.lineEdit_epsModel.setValidator(double)
        self.app.lineEdit_epsModeltArea.setValidator(double)
        self.app.lineEdit_growth_rate_Xi.setValidator(double)

        integer = QtGui.QIntValidator()
        self.app.lineEdit_decGrid.setValidator(integer)
        self.app.lineEdit_decIntersect.setValidator(integer)
        self.app.lineEdit_decMerge.setValidator(integer)
        self.app.lineEdit_decModel.setValidator(integer)

    def ui_buttons(self):
        '''
        '''
        def dialog_clear():
            '''
            '''
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            ret = msg.question(self.window, "Clear All?",
                               "Are you sure you want to clear data?",
                               buttons=QtWidgets.QMessageBox.Yes |
                               QtWidgets.QMessageBox.No)
            if ret == msg.Yes:
                self.Sim.reset()
                self.graph_model.draw_idle()
                self.graph_evap_top.draw_idle()
                self.graph_evap_bot.draw_idle()

        def dialog_abort():
            '''
            '''
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            ret = msg.question(self.window, "Abort Sim?",
                               "Are you sure you want to abort run?",
                               buttons=QtWidgets.QMessageBox.Yes |
                               QtWidgets.QMessageBox.No)
            if ret == msg.Yes:
                self.Sim.simulation.pause()
                self.Sim.simulation._stop()
                self.app.pushButton_Pause.setDisabled(True)
                self.app.pushButton_Abort_Run.setDisabled(True)
                self.app.pushButton_Start.setDisabled(False)

        def sim_start():
            '''
            '''
            self.app.pushButton_Pause.setDisabled(False)
            self.app.pushButton_Abort_Run.setDisabled(False)
            self.app.pushButton_Start.setDisabled(True)
            self.Sim.run()

        def sim_pause():
            '''
            '''
            if self.Sim.paused:
                log.info("Resuming")
                self.Sim.simulation.resume()
                self.Sim.paused = False
                self.app.pushButton_Pause.setText("Pause")
            else:
                log.info("Simulation paused")
                self.Sim.simulation.pause()
                self.Sim.paused = True
                self.app.pushButton_Pause.setText("Resume")

        def sim_reset():
            '''
            '''
            self.toolbar._update_view()
            self.toolbar.home()

        def sim_quit():
            '''
            '''
            log.info("Program End")
            sys.exit(QtWidgets.QApplication(sys.argv).exit())

        self.app.pushButton_Abort_Run.clicked.connect(dialog_abort)
        self.app.pushButton_Clear_Run.clicked.connect(dialog_clear)
        self.app.pushButton_Reset_Graph.clicked.connect(sim_reset)

        self.app.pushButton_Load_Evap_Profile.clicked.connect(
            self.Sim.load_csv_angletime)
        self.app.pushButton_Load_Model.clicked.connect(
            self.Sim.load_csv_model)
        self.app.pushButton_Save_Evap_Profile.clicked.connect(
            self.Sim.save_csv_model)
        self.app.pushButton_load_settings.clicked.connect(
            self.Sim.load_csv_settings)

        self.app.pushButton_Start.clicked.connect(sim_start)
        self.app.pushButton_Pause.clicked.connect(sim_pause)
        self.app.pushButton_Quit.clicked.connect(sim_quit)

        self.app.actionHelp.triggered.connect(self.help.show)
        log.info("Application UI buttons connected")

    def ui_presets(self):
        '''
        '''
        self.app.lineEdit_Model_Resolution.setText("10")
        self.app.lineEdit_Grid_Space.setText("2.5")
        self.app.lineEdit_Raycast_Length.setText("5000")
        self.app.lineEdit_Model_Limit.setText("10000")

        self.app.lineEdit_Evaporation_Rate.setText("1")
        self.app.lineEdit_Evaporation_Time.setText("100")

        self.app.progressBar.setProperty("value", 0)
