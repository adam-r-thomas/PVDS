
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg,
                                                NavigationToolbar2QT)

import logging
log = logging.getLogger("evapsim")


class Application_Qt(object):
    """
    GUI Window for the simulator
    """

    def __init__(self, simulator):
        self.sim = simulator
        self.MainWindow = QtWidgets.QMainWindow()
        self.setupUi(self.MainWindow)
        self.connectUi()

        self.graphcanvas = FigureCanvasQTAgg(self.sim.graph_ani)
        self.layoutPlot = QtWidgets.QWidget(self.graphicsView)
        self.gridPlot = QtWidgets.QGridLayout(self.layoutPlot)
        self.gridPlot.addWidget(self.graphcanvas)

        self.graphcanvas2 = FigureCanvasQTAgg(self.sim.graph_ray)
        self.layoutPlot2 = QtWidgets.QWidget(self.graphicsView_2)
        self.gridPlot2 = QtWidgets.QGridLayout(self.layoutPlot2)
        self.gridPlot2.addWidget(self.graphcanvas2)

        self.toolbar = NavigationToolbar2QT(self.graphcanvas,
                                            self.graphicsView)
        self.gridPlot.addWidget(self.toolbar)

        self.graphcanvas.draw_idle()
        self.graphcanvas2.draw_idle()

        self.pushButton_pause.setDisabled(True)
        log.info("Application window init complete")

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1015, 725)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.frame_inputs = QtWidgets.QFrame(self.centralwidget)
        self.frame_inputs.setGeometry(QtCore.QRect(20, 35, 350, 649))
        self.frame_inputs.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_inputs.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_inputs.setObjectName("frame_inputs")

        self.layoutWidget = QtWidgets.QWidget(self.frame_inputs)
        self.layoutWidget.setGeometry(QtCore.QRect(5, 5, 340, 140))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        self.label_model_resolution = QtWidgets.QLabel(self.layoutWidget)
        self.label_model_resolution.setObjectName("label_model_resolution")
        self.gridLayout.addWidget(self.label_model_resolution, 0, 0, 1, 1)
        self.lineEdit_model_resolution = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_model_resolution.setObjectName("lineEdit_model_resolution")  # noqa
        self.gridLayout.addWidget(self.lineEdit_model_resolution, 0, 1, 1, 1)
        self.label_model_resolution_unit = QtWidgets.QLabel(self.layoutWidget)
        self.label_model_resolution_unit.setObjectName("label_model_resolution_unit")  # noqa
        self.gridLayout.addWidget(self.label_model_resolution_unit, 0, 2, 1, 1)

        self.label_evap_rate = QtWidgets.QLabel(self.layoutWidget)
        self.label_evap_rate.setObjectName("label_evap_rate")
        self.gridLayout.addWidget(self.label_evap_rate, 1, 0, 1, 1)
        self.lineEdit_evap_rate = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_evap_rate.setObjectName("lineEdit_evap_rate")
        self.gridLayout.addWidget(self.lineEdit_evap_rate, 1, 1, 1, 1)
        self.label_evap_rate_unit = QtWidgets.QLabel(self.layoutWidget)
        self.label_evap_rate_unit.setObjectName("label_evap_rate_unit")
        self.gridLayout.addWidget(self.label_evap_rate_unit, 1, 2, 1, 1)

        self.label_evap_time = QtWidgets.QLabel(self.layoutWidget)
        self.label_evap_time.setObjectName("label_evap_time")
        self.gridLayout.addWidget(self.label_evap_time, 2, 0, 1, 1)
        self.lineEdit_evap_time = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_evap_time.setObjectName("lineEdit_evap_time")
        self.gridLayout.addWidget(self.lineEdit_evap_time, 2, 1, 1, 1)
        self.label_evap_time_unit = QtWidgets.QLabel(self.layoutWidget)
        self.label_evap_time_unit.setObjectName("label_evap_time_unit")
        self.gridLayout.addWidget(self.label_evap_time_unit, 2, 2, 1, 1)

        self.label_grid_space = QtWidgets.QLabel(self.layoutWidget)
        self.label_grid_space.setObjectName("label_grid_space")
        self.gridLayout.addWidget(self.label_grid_space, 3, 0, 1, 1)
        self.lineEdit_grid_space = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_grid_space.setObjectName("lineEdit_grid_space")
        self.gridLayout.addWidget(self.lineEdit_grid_space, 3, 1, 1, 1)
        self.label_grid_space_unit = QtWidgets.QLabel(self.layoutWidget)
        self.label_grid_space_unit.setObjectName("label_grid_space_unit")
        self.gridLayout.addWidget(self.label_grid_space_unit, 3, 2, 1, 1)

        self.label_raycast_length = QtWidgets.QLabel(self.layoutWidget)
        self.label_raycast_length.setObjectName("label_raycast_length")
        self.gridLayout.addWidget(self.label_raycast_length, 4, 0, 1, 1)
        self.lineEdit_raycast_length = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_raycast_length.setObjectName("lineEdit_raycast_length")
        self.gridLayout.addWidget(self.lineEdit_raycast_length, 4, 1, 1, 1)
        self.label_raycast_length_unit = QtWidgets.QLabel(self.layoutWidget)
        self.label_raycast_length_unit.setObjectName("label_raycast_length_unit")  # noqa
        self.gridLayout.addWidget(self.label_raycast_length_unit, 4, 2, 1, 1)

        self.splitter_left = QtWidgets.QSplitter(self.frame_inputs)
        self.splitter_left.setGeometry(QtCore.QRect(5, 160, 130, 70))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.splitter_left.sizePolicy().hasHeightForWidth())
        self.splitter_left.setSizePolicy(sizePolicy)
        self.splitter_left.setOrientation(QtCore.Qt.Vertical)
        self.splitter_left.setObjectName("splitter_left")
        self.pushButton_load_model = QtWidgets.QPushButton(self.splitter_left)
        self.pushButton_load_model.setObjectName("pushButton_load_model")
        self.pushButton_load_model.setMinimumSize(QtCore.QSize(0, 20))
        self.pushButton_load_model.setMaximumSize(QtCore.QSize(130, 20))
        self.pushButton_load_AvsT = QtWidgets.QPushButton(self.splitter_left)
        self.pushButton_load_AvsT.setObjectName("pushButton_load_AvsT")
        self.pushButton_load_AvsT.setMinimumSize(QtCore.QSize(0, 20))
        self.pushButton_load_AvsT.setMaximumSize(QtCore.QSize(130, 20))
        self.pushButton_save_model = QtWidgets.QPushButton(self.splitter_left)
        self.pushButton_save_model.setObjectName("pushButton_save_model")
        self.pushButton_save_model.setMinimumSize(QtCore.QSize(0, 20))
        self.pushButton_save_model.setMaximumSize(QtCore.QSize(130, 20))

        self.splitter_right = QtWidgets.QSplitter(self.frame_inputs)
        self.splitter_right.setGeometry(QtCore.QRect(215, 160, 130, 70))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.splitter_right.sizePolicy().hasHeightForWidth())
        self.splitter_right.setSizePolicy(sizePolicy)
        self.splitter_right.setOrientation(QtCore.Qt.Vertical)
        self.splitter_right.setObjectName("splitter_right")
        self.pushButton_simAbort = QtWidgets.QPushButton(self.splitter_right)
        self.pushButton_simAbort.setObjectName("pushButton_simAbort")
        self.pushButton_simAbort.setMinimumSize(QtCore.QSize(0, 20))
        self.pushButton_simAbort.setMaximumSize(QtCore.QSize(130, 20))
        self.pushButton_simReset = QtWidgets.QPushButton(self.splitter_right)
        self.pushButton_simReset.setObjectName("pushButton_simReset")
        self.pushButton_simReset.setMinimumSize(QtCore.QSize(0, 20))
        self.pushButton_simReset.setMaximumSize(QtCore.QSize(130, 20))
        self.pushButton_reset_graph = QtWidgets.QPushButton(self.splitter_right)  # noqa
        self.pushButton_reset_graph.setObjectName("pushButton_reset_graph")
        self.pushButton_reset_graph.setMinimumSize(QtCore.QSize(0, 20))
        self.pushButton_reset_graph.setMaximumSize(QtCore.QSize(130, 20))

        self.label = QtWidgets.QLabel(self.frame_inputs)
        self.label.setGeometry(QtCore.QRect(5, 240, 345, 40))
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.frame_inputs)
        self.label_2.setGeometry(QtCore.QRect(5, 280, 345, 40))
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.frame_inputs)
        self.label_3.setGeometry(QtCore.QRect(5, 320, 345, 70))
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.frame_inputs)
        self.label_4.setGeometry(QtCore.QRect(5, 400, 345, 70))
        self.label_4.setWordWrap(True)
        self.label_4.setObjectName("label_4")

        self.pushButton_pause = QtWidgets.QPushButton(self.frame_inputs)
        self.pushButton_pause.setGeometry(QtCore.QRect(180, 620, 80, 23))
        self.pushButton_pause.setObjectName("pushButton_pause")
        self.progressBar = QtWidgets.QProgressBar(self.frame_inputs)
        self.progressBar.setGeometry(QtCore.QRect(5, 590, 340, 25))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.pushButton_quit = QtWidgets.QPushButton(self.frame_inputs)
        self.pushButton_quit.setGeometry(QtCore.QRect(5, 620, 80, 23))
        self.pushButton_quit.setObjectName("pushButton_quit")
        self.pushButton_start = QtWidgets.QPushButton(self.frame_inputs)
        self.pushButton_start.setGeometry(QtCore.QRect(265, 620, 80, 23))
        self.pushButton_start.setObjectName("pushButton_start")

        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(380, 5, 620, 680))
        self.tabWidget.setObjectName("tabWidget")

        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.graphicsView = QtWidgets.QGraphicsView(self.tab)
        self.graphicsView.setGeometry(QtCore.QRect(-1, -1, 620, 655))
        self.graphicsView.setObjectName("graphicsView")
        self.tabWidget.addTab(self.tab, "")

        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.tab_2)
        self.graphicsView_2.setGeometry(QtCore.QRect(-1, -1, 620, 655))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.tabWidget.addTab(self.tab_2, "")

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        log.info("Application UI setup complete")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(
            _translate("MainWindow", "Evaporation Simulator"))
        self.label_model_resolution.setText(
            _translate("MainWindow", "Model Resolution"))  # noqa
        self.lineEdit_model_resolution.setText(
            _translate("MainWindow", "100"))
        self.label_model_resolution_unit.setText(
            _translate("MainWindow", "Angstroms"))  # noqa
        self.label_evap_rate.setText(
            _translate("MainWindow", "Evaporation Rate"))  # noqa
        self.lineEdit_evap_rate.setText(
            _translate("MainWindow", "5"))
        self.label_evap_rate_unit.setText(
            _translate("MainWindow", "Angstroms/sec"))  # noqa
        self.label_evap_time.setText(
            _translate("MainWindow", "Evaporation Time"))  # noqa
        self.lineEdit_evap_time.setText(
            _translate("MainWindow", "10"))
        self.label_evap_time_unit.setText(
            _translate("MainWindow", "Seconds"))
        self.label_grid_space.setText(
            _translate("MainWindow", "Grid Space"))
        self.lineEdit_grid_space.setText(
            _translate("MainWindow", "1"))
        self.label_grid_space_unit.setText(
            _translate("MainWindow", "Angstroms"))
        self.label_raycast_length.setText(
            _translate("MainWindow", "Grid Raycast Length"))
        self.lineEdit_raycast_length.setText(
            _translate("MainWindow", "100000"))
        self.label_raycast_length_unit.setText(
            _translate("MainWindow", "Angstroms"))
        self.pushButton_load_model.setText(
            _translate("MainWindow", "Load Model"))
        self.pushButton_load_AvsT.setText(
            _translate("MainWindow", "Load Angle vs Time"))
        self.pushButton_save_model.setText(
            _translate("MainWindow", "Save Evap Model"))
        self.pushButton_simAbort.setText(
            _translate("MainWindow", "Abort Sim"))
        self.pushButton_simReset.setText(
            _translate("MainWindow", "Clear Models"))
        self.pushButton_reset_graph.setText(
            _translate("MainWindow", "Reset View"))
        self.label_2.setText(
            _translate("MainWindow", "<b>Evaporation Rate:</b> Angstrom / sec of material deposition. Rate is affected by angle: cos(theta)."))  # noqa
        self.label_3.setText(
            _translate("MainWindow", "<b>Grid Space:</b> Smallest vertice resolution allowed. This helps prevent impossible shapes from forming. Two vertices that are closer then this set distance are merged."))  # noqa
        self.label_4.setText(
            _translate("MainWindow", "<b>Raycast Length:</b> Length of the ray cast from each vertice. Too short a ray can have evaporation in bad places. Too long a ray can produce math errors (float point problems)."))  # noqa
        self.label.setText(
            _translate("MainWindow", "<b>Model Resolution:</b> The density of the raycasts on the model."))  # noqa
        self.pushButton_pause.setText(
            _translate("MainWindow", "Pause"))
        self.pushButton_quit.setText(
            _translate("MainWindow", "Quit"))
        self.pushButton_start.setText(
            _translate("MainWindow", "Start"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab),
                                  _translate("MainWindow", "Model"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2),
                                  _translate("MainWindow", "Raycast"))
        log.info("Application UI translate complete")

    def connectUi(self):
        """
        Connect buttons to the simulator
        Ensure inputs are only numbers
        """
        validator = QtGui.QDoubleValidator()
        self.lineEdit_model_resolution.setValidator(validator)
        self.lineEdit_evap_rate.setValidator(validator)
        self.lineEdit_evap_time.setValidator(validator)
        self.lineEdit_grid_space.setValidator(validator)
        self.lineEdit_raycast_length.setValidator(validator)

        self.pushButton_load_model.clicked.connect(self.sim.load_csv_model)
        self.pushButton_load_AvsT.clicked.connect(self.sim.load_csv_angletime)
        self.pushButton_save_model.clicked.connect(self.sim.save_csv_model)

        self.pushButton_simAbort.clicked.connect(self.dialog_abort)
        self.pushButton_simReset.clicked.connect(self.dialog_clear)
        self.pushButton_reset_graph.clicked.connect(self.displayReset)

        self.pushButton_start.clicked.connect(self.simStart)
        self.pushButton_quit.clicked.connect(self._quit)
        self.pushButton_pause.clicked.connect(self.simPause)
        log.info("Application UI buttons connected")

    def simAbort(self):
        """
        Replace this with Abort run
        """
        self.sim.abort()

    def simStart(self):
        self.pushButton_pause.setDisabled(False)
        # self.pushButton_start.setDisabled(True)
        self.sim.run()

    def simPause(self):
        if self.sim.paused:
            log.info("Resuming")
            self.sim.simulation.resume()
            self.sim.paused = False
        else:
            log.info("Simulator paused")
            self.sim.simulation.pause()
            self.sim.paused = True

    def displayReset(self):
        self.toolbar._update_view()
        self.toolbar.home()

    def dialog_clear(self):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        ret = msg.question(self.MainWindow, "Clear All?",
                           "Are you sure you want to clear data?",
                           buttons=QtWidgets.QMessageBox.Yes |
                           QtWidgets.QMessageBox.No)
        if ret == msg.Yes:
            self.sim.reset()
            self.graphcanvas.draw_idle()
            self.graphcanvas2.draw_idle()

    def dialog_abort(self):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        ret = msg.question(self.MainWindow, "Abort Sim?",
                           "Are you sure you want to abort run?",
                           buttons=QtWidgets.QMessageBox.Yes |
                           QtWidgets.QMessageBox.No)
        if ret == msg.Yes:
            self.simAbort()

    def _quit(self):
        log.info("Program End")
        sys.exit(QtWidgets.QApplication(sys.argv).exit())
