
import math
import time
import sys

from numba import cuda

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg,
                                                NavigationToolbar2QT)

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import cycle

from evapsim.physics import (grid_gpu, intersection_gpu,
                             merge_gpu, model_gpu)
from evapsim import application

import logging
log = logging.getLogger("evapsim")

assert cuda.detect()
device = cuda.get_current_device()


class HelpDialog(QtWidgets.QDialog):
    '''Creates pop up window for help info
    '''
    def __init__(self):
        super().__init__()
        self.ui_help = application.Ui_Dialog_HelpTree()
        self.ui_help.setupUi(self)


class SimulatorWindow(object):
    '''
    '''

    def __init__(self, sim, window):
        self.Sim = sim
        self.window = window
        self.app = application.Ui_MainWindow()
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


class Simulator(object):
    '''
    Main class object of simulation code

    :Model: A list of points given in .csv. First column is x, second
        column is y. Units are in microns. This list is passed in and converted
        to angstroms. The simulation consideres that the model is drawn with
        a single continous line, a profile outline. This outline is saved as
        a set of vertices.

    :Vertices: Points that break up straight lines and are used to cast rays
        for the intersection tests.

    :Model_Resolution: The model vertices are "gridded" where points evenly
        spaced by a set amount in the GUI and placed between vertices if
        possible.

    :Grip_Space:  Determines if a pair of points will be averaged
        together if they get too close to each other. This is critical. As a
        pair of points that are growning too close to each other can
        possibly pass each other. This pass forms a intersection triangle and
        will grow as a triangle on the model surface. Acting like a tumor of
        sorts and degrading the model quality in that particular area.

    :Raycast_Length: Length of the ray used to check for intersections. As the
        model does not understand the shape this value must be long enough to
        ensure intersections when casted. i.e too small a ray won't intersect
        with anything and evaporation will occur in non-sensical places.

    Explanation:
    Instead of casting in rays to the object. I take the vertices, place an
    amount of points between each vertice (determined by model resolution)
    and cast rays from each point (now called the model grid). The rays are
    a certain length that should be large enough to cross through the
    entire model (regardless of angle). These rays have the same model
    shape. Thus a grid of the model can be constructed in the GPU.

    For easier testing of the overall method during debugging. Each call to
    the GPU keeps a separatly named np.array. Future optimizations should seek
    to keep the array in GPU memory until the model_grid_gpu method
    is called. Then a copy of the array should be sent to the plotting window
    for display.

    :var epsilon: Is the computer determined zero. Anything smaller than
        that is assumed to be zero.

    :param tickrate: The average amount of time that passes for each cycle of
        the angle. Assists to ensure the evaporation is correct for each step
        of the angle during the simulation.

    :array avst_ini: the angle vs time as a 2D numpy array.
            Angle = 1, Time = 0

    :array model_x: the np.array of the set of x points of the model
    :array model_y: the np.array of the set of y points of the model
    Additional np.array variants of model_x, model_y. Unique array naming for
    debugging purposes
    :array vert_x: = model_x
    :array vert_y: = model_y
    :array merge_x: = vert_x
    :array mergy_y: = vert_y
    '''

    def __init__(self):
        '''
        Build initial parameters of the evaporation simulator
        '''
        # Setup CUDA device
        self.device = device
        self.tpb = device.WARP_SIZE
        self.tpb_2d = (self.tpb // 2, self.tpb // 2)
        self.tickrate = 1.0

        # Setup Graphs
        self.model_ini = [[], []]
        self.model_x = []
        self.model_y = []
        self.avst_ini = [[], [], []]

        self.graph_ani = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax_ani = self.graph_ani.add_subplot(111)

        self.graph_ray1 = plt.Figure(figsize=(6, 3), dpi=100)
        self.ax_ray1 = self.graph_ray1.add_subplot(111)

        self.graph_ray2 = plt.Figure(figsize=(6, 3), dpi=100)
        self.ax_ray2 = self.graph_ray2.add_subplot(111)
        self.graphs()
        self.paused = False
        log.info("Simulator init complete.")

        # Start simulator window
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = QtWidgets.QMainWindow()
        log.info("Application Window running:")
        self.gui = SimulatorWindow(self, self.window)
        self.window.show()
        self.simulation_parameters()
        log.info("Simulator running:")
        sys.exit(self.app.exec_())

    def graphs(self):
        """Setup graphs Model and Raycast
        """
        # Model animation
        self.ax_ani.grid(1)
        self.ax_ani.set_xlabel("Model Pos. X (A)", fontsize=12)
        self.ax_ani.set_ylabel("Model Pos. Y (A)", fontsize=12)
        self.ax_ani.set_xlim(auto=True)
        self.ax_ani.set_ylim(auto=True)

        self.line_ani, = self.ax_ani.plot([], [], 'r-', lw=2)
        self.line_angle, = self.ax_ani.plot([], [], 'g-', lw=0.5)
        self.time_text = self.ax_ani.text(
            0.01, 1.01, '', transform=self.ax_ani.transAxes)
        self.angle_text = self.ax_ani.text(
            0.25, 1.01, '', transform=self.ax_ani.transAxes)
        self.line_static, = self.ax_ani.plot([], [], 'k-', lw=2)

        self.graph_ani.tight_layout()

        # Raycast profile
        self.ax_ray1.grid(1)
        self.ax_ray2.grid(1)
        self.ax_ray1.set_xlabel("Time (s)", fontsize=12)
        self.ax_ray2.set_xlabel("Time (s)", fontsize=12)
        self.ax_ray1.set_ylabel("Theta (rad)", fontsize=12)
        self.ax_ray2.set_ylabel("Phi (rad)", fontsize=12)

        self.graph_ray1.tight_layout()
        self.graph_ray2.tight_layout()

    def reset(self):
        """
        Resets simulator to start state
        """
        self.model_ini = [[], []]
        self.model_x = []
        self.model_y = []
        self.avst_ini = [[], [], []]

        self.ax_ani.cla()
        self.ax_ray1.cla()
        self.ax_ray2.cla()

        self.graphs()

    def run(self):
        '''
        Run the simulation and present results to animation graph
        '''
        log.info("Simulator run started.")
        self.simulation_parameters()

        try:
            assert self.model_x != []
            assert self.model_y != []
            assert self.avst_ini != [[], [], []]

        except AssertionError:
            log.info("Model assertion error. Check model and angle parameters")
            self.gui.app.pushButton_Pause.setDisabled(True)
            self.gui.app.pushButton_Abort_Run.setDisabled(True)
            self.gui.app.pushButton_Start.setDisabled(False)
            return

        def ani_init():
            '''
            When blit=True this is called each frame - set's parameters to
            nothing (wipes the frame)
            '''
            self.line_ani.set_data([], [])
            self.line_angle.set_data([], [])
            self.line_static.set_data([], [])
            self.time_text.set_text('')
            self.angle_text.set_text('')
            return (self.line_ani,
                    self.line_angle,
                    self.time_text,
                    self.angle_text,
                    self.line_static)

        def animate(i):
            '''
            Intersection function core code that displays each finished
            calculation on the self.graph_ani
            '''
            timer_start = time.process_time()
            log.info("")
            log.info("Evap Step: %s" % timer_start)

            # Setup display info on graph
            self.time_text.set_text('Time: %.2f' % self.timer)
            self.timer += self.elapse_time
            angle = next(self.cycle_angle)
            phi = next(self.cycle_phi)
            self.angle_text.set_text('Angle: %.3f' % angle)
            log.info("Cycle: %s" % angle)
            log.info("Loop: %s" % self.loop_counter)

            if len(self.model_x) > self.model_limit:
                if self.boolModelRes:
                    print("Model size limit hit. Reducing model points.")
                    log.info("Model size limit hit. Reducing model points.")
                    self.model_x, self.model_y = self.model_derez(self.model_x,
                                                                  self.model_y)

            # Run model intersection test
            try:
                log.info("GPU: Intersection")
                self.intersect_result = self.calc_intersection(self.model_x,
                                                               self.model_y,
                                                               angle)
            except:  # noqa: I do not know all failure modes from GPU
                tb = sys.exc_info()
                log.exception(tb)
                log.error("Failure on self.calc_intersection")
                print("Error occurred with self.calc_intersection")
                print(tb)
                self.gui.app.pushButton_Pause.click()
                return

            # Determine new vertices from the model on the grid | Adds material
            try:
                log.info("GPU: Model Update")
                self.vert_x, self.vert_y, self.vert_i = self.model_update_gpu(
                    self.model_x,
                    self.model_y,
                    self.intersect_result,
                    angle, phi)
            except:  # noqa: I do not know all failure modes from GPU
                tb = sys.exc_info()
                log.exception(tb)
                log.error("Failure on self.model_update_gpu")
                print("Error occurred with self.model_update_gpu")
                self.gui.app.pushButton_Pause.click()
                return

            # Merge vertices that are too close
            try:
                if self.grid:
                    log.info("GPU: Model Merge")
                    self.merge_x, self.merge_y = self.model_merge(
                        self.vert_x,
                        self.vert_y,
                        self.vert_i)
                else:
                    self.merge_x = self.vert_x
                    self.merge_y = self.vert_y
            except:  # noqa: I do not know all failure modes from GPU
                tb = sys.exc_info()
                log.exception(tb)
                log.error("Failure on self.model_merge")
                print("Error occurred with self.model_merge")
                self.gui.app.pushButton_Pause.click()
                return

            # Re-grid the model
            try:
                log.info("GPU: Model Grid")
                self.model_x, self.model_y = self.model_grid_gpu(self.merge_x,
                                                                 self.merge_y)
            except:  # noqa: I do not know all failure modes from GPU
                tb = sys.exc_info()
                log.exception(tb)
                log.error("Failure on self.model_grid_gpu")
                print("Error occurred with self.model_grid_gpu")
                self.gui.app.pushButton_Pause.click()
                return

            # Draw results of intersection test
            self.line_static.set_data(self.model_ini[0], self.model_ini[1])
            self.line_ani.set_data(self.vert_x, self.vert_y)

            # Draw the current ray cast direction (green line)
            self.line_angle.set_data(
                [(max(self.model_x) + min(self.model_x)) / 2.0,
                 ((max(self.model_x) + min(self.model_x)) / 2.0) +
                 round(self.raycast_length * math.sin(angle), 10)],
                [0, round(self.raycast_length * math.cos(angle), 10)])

            # Update progress to user
            self.gui.app.progressBar.setProperty(
                "value", 100 * i // (self.total_fps - 1))

            # Remaining calculations
            timer_end = time.process_time()
            log.info("Evap Time: %s" % timer_end)
            self.loop_counter += 1

            if i == (self.total_fps - 1):
                # Simulation complete
                self.gui.app.pushButton_Start.setDisabled(False)
                self.gui.app.pushButton_Pause.setDisabled(True)
                self.gui.app.pushButton_Abort_Run.setDisabled(True)

            return (self.line_ani,
                    self.line_angle,
                    self.time_text,
                    self.angle_text,
                    self.line_static)

        fps = (len(self.avst_ini[0]) - 1) / self.avst_ini[0][-1]
        self.total_fps = int(fps * self.evaporation_time)
        self.cycle_angle = cycle(self.avst_ini[1])
        self.cycle_phi = cycle(self.avst_ini[2])
        self.elapse_time = self.avst_ini[0][-1] / (len(self.avst_ini[0]) - 1)
        self.timer = 0
        self.loop_counter = 0

        self.simulation = animation.FuncAnimation(self.graph_ani,
                                                  animate,
                                                  frames=self.total_fps,
                                                  init_func=ani_init,
                                                  blit=False,
                                                  repeat=False)

        self.gui.graph_model.draw_idle()

    def simulation_parameters(self):
        '''
        Starting parameters of the evaporation - pulls values from the GUI
        '''
        self.model_resolution = float(
            self.gui.app.lineEdit_Model_Resolution.text())

        self.evap_rate_text = float(
            self.gui.app.lineEdit_Evaporation_Rate.text())

        self.evaporation_rate = (float(
            self.gui.app.lineEdit_Evaporation_Rate.text()) * self.tickrate)

        self.evaporation_time = float(
            self.gui.app.lineEdit_Evaporation_Time.text())

        self.gridspace = float(
            self.gui.app.lineEdit_Grid_Space.text())

        self.raycast_length = float(
            self.gui.app.lineEdit_Raycast_Length.text())

        self.model_limit = float(
            self.gui.app.lineEdit_Model_Limit.text())

        self.average_divets = bool(
            self.gui.app.checkBox_divet.checkState())

        self.average_peaks = bool(
            self.gui.app.checkBox_peaks.checkState())

        self.corner = bool(
            self.gui.app.checkBox_corners.checkState())

        self.grid = bool(
            self.gui.app.checkBox_grid.checkState())

        self.boolModelRes = bool(
            self.gui.app.checkBox_modelRes.checkState())

        self.epsIntersect = float(
            self.gui.app.lineEdit_epsIntersect.text())

        self.epsGrid = float(
            self.gui.app.lineEdit_epsGrid.text())

        self.epsModeltArea = float(
            self.gui.app.lineEdit_epsModeltArea.text())

        self.epsModel = float(
            self.gui.app.lineEdit_epsModel.text())

        self.epsMerge = float(
            self.gui.app.lineEdit_epsMerge.text())

        self.decIntersect = int(
            self.gui.app.lineEdit_decIntersect.text())

        self.decGrid = int(
            self.gui.app.lineEdit_decGrid.text())

        self.decModel = int(
            self.gui.app.lineEdit_decModel.text())

        self.decMerge = int(
            self.gui.app.lineEdit_decMerge.text())

        self.growthXi = float(
            self.gui.app.lineEdit_growth_rate_Xi.text())

        if self.gui.app.radioButton_linearrule.isChecked():
            self.growthDirection: 'linear' = 0
        if self.gui.app.radioButton_cosinerule.isChecked():
            self.growthDirection: 'cosine' = 1
        if self.gui.app.radioButton_tangentrule.isChecked():
            self.growthDirection: 'tangent' = 2

        log.info("Parameters loaded from GUI")

    def load_csv_model(self):
        '''
        Load in csv file containing vertices of model (x,y) points
        Assumes data is in angstroms

        Model x (A)    Model y (A)
        x data        y data
        .            .
        .            .
        .            .
        x data end    y data end
        '''
        filepath, _ = QFileDialog.getOpenFileName(
            caption="Evaporation Simulator - Load model file",
            filter="*.csv")
        if filepath:
            filepath = Path(filepath)
            try:
                assert '.csv' in filepath.suffix
                df = pd.read_csv(filepath, dtype=np.float64,
                                 usecols=['Model x (A)', 'Model y (A)'])
                self.model_resolution = float(
                    self.gui.app.lineEdit_Model_Resolution.text())

                self.model_ini = np.array([list(df['Model x (A)'].dropna()),
                                           list(df['Model y (A)'].dropna())])

                self.line_static.set_data(self.model_ini[0], self.model_ini[1])

                self.model_x, self.model_y = self.model_grid_gpu(
                    self.model_ini[0],
                    self.model_ini[1])

                xpad = (max(self.model_ini[0]) - min(self.model_ini[0])) * 0.05
                ypad = (max(self.model_ini[1]) - min(self.model_ini[1])) * 0.05

                self.ax_ani.set_xlim(min(self.model_ini[0]) - xpad,
                                     max(self.model_ini[0]) + xpad)
                self.ax_ani.set_ylim(min(self.model_ini[1]) - ypad,
                                     max(self.model_ini[1]) + ypad)

                self.gui.graph_model.draw_idle()
                log.info("Model file loaded: %s" % filepath)
            except AssertionError or ValueError:
                log.info("Invalid file selected.")

    def load_csv_angletime(self):
        '''
        Load in time (col0) and angle (col1)

        Returns:
        :param tickrate: the average time step between each angle
        :array avst_ini: the angle vs time as a 2D numpy array.
            Angle = 1:2, Time = 1
        '''
        filepath, _ = QFileDialog.getOpenFileName(
            caption="Evaporation Simulator - Load angle vs time file",
            filter="*.csv")
        if filepath:
            try:
                filepath = Path(filepath)
                assert '.csv' in filepath.suffix
                df = pd.read_csv(filepath, dtype=np.float64,
                                 usecols=['Time (sec)',
                                          'Theta (deg)',
                                          'Phi (deg)'])

                theta_deg = [
                    math.radians(x) for x in list(df['Theta (deg)'].dropna())]
                phi_deg = [
                    math.radians(x) for x in list(df['Phi (deg)'].dropna())]
                self.avst_ini = np.array(
                    [list(df['Time (sec)'].dropna()), theta_deg, phi_deg])

                tempa = []
                timelist = list(df['Time (sec)'].dropna())
                for i in range(1, (len(df['Time (sec)'].dropna()))):
                    tempa.append(timelist[i] - timelist[i - 1])
                self.tickrate = np.array(tempa).mean()
                del tempa
                del timelist

                self.ax_ray1.cla()
                self.ax_ray2.cla()

                self.ax_ray1.plot(self.avst_ini[0], self.avst_ini[1],
                                  label="ax1")
                self.ax_ray2.plot(self.avst_ini[0], self.avst_ini[2],
                                  label="ax2")

                self.ax_ray1.grid(1)
                self.ax_ray2.grid(1)
                self.ax_ray1.set_xlabel("Time (s)", fontsize=12)
                self.ax_ray2.set_xlabel("Time (s)", fontsize=12)
                self.ax_ray1.set_ylabel("Theta (rad)", fontsize=12)
                self.ax_ray2.set_ylabel("Phi (rad)", fontsize=12)

                self.graph_ray1.tight_layout()
                self.graph_ray2.tight_layout()

                self.gui.graph_evap_top.draw_idle()
                self.gui.graph_evap_bot.draw_idle()
                log.info("AvsT file loaded: %s" % filepath)
            except AssertionError or ValueError:
                log.info("Invalid file selected.")

    def load_csv_settings(self):
        filepath, _ = QFileDialog.getOpenFileName(
            caption="Evaporation Simulator - Load settings file",
            filter="*.csv")
        if filepath:
            try:
                filepath = Path(filepath)
                assert '.csv' in filepath.suffix
                df = pd.read_csv(filepath,
                                 usecols=['Settings Name', 'Settings Val'])

                svals = df['Settings Val'].dropna()

                self.gui.app.lineEdit_Model_Resolution.setText(str(svals[0]))
                self.gui.app.lineEdit_Evaporation_Rate.setText(str(svals[1]))
                self.gui.app.lineEdit_Evaporation_Time.setText(str(svals[2]))
                self.gui.app.lineEdit_Grid_Space.setText(str(svals[3]))
                self.gui.app.lineEdit_Raycast_Length.setText(str(svals[4]))
                self.gui.app.lineEdit_Model_Limit.setText(str(svals[5]))

                self.gui.app.checkBox_divet.setChecked(svals[6])
                self.gui.app.checkBox_peaks.setChecked(svals[7])
                self.gui.app.checkBox_corners.setChecked(svals[8])
                self.gui.app.checkBox_grid.setChecked(svals[9])
                self.gui.app.checkBox_modelRes.setChecked(svals[10])

                self.gui.app.lineEdit_epsIntersect.setText(str(svals[11]))
                self.gui.app.lineEdit_epsGrid.setText(str(svals[12]))
                self.gui.app.lineEdit_epsModel.setText(str(svals[13]))
                self.gui.app.lineEdit_epsModeltArea.setText(str(svals[14]))
                self.gui.app.lineEdit_epsMerge.setText(str(svals[15]))

                self.gui.app.lineEdit_decIntersect.setText(str(int(svals[16])))
                self.gui.app.lineEdit_decGrid.setText(str(int(svals[17])))
                self.gui.app.lineEdit_decModel.setText(str(int(svals[18])))
                self.gui.app.lineEdit_decMerge.setText(str(int(svals[19])))

                log.info("Settings file loaded: %s" % filepath)
            except AssertionError or ValueError:  # TODO: verify this works
                log.info("Invalid file selected.")

    def save_csv_model(self):
        '''
        Save the evaporation model to a csv file
        '''
        filepath, _ = QFileDialog.getSaveFileName(
            caption="Evaporation Simulator - Save evaporation model",
            filter="*.csv")
        if filepath:
            filepath = Path(filepath + ".csv")
            df1 = pd.DataFrame()
            df1['Model x (A)'] = self.model_ini[0]
            df1['Model y (A)'] = self.model_ini[1]

            df2 = pd.DataFrame()
            df2['Evap x (A)'] = self.model_x
            df2['Evap y (A)'] = self.model_y

            df3 = pd.DataFrame()
            df3['Time (sec)'] = self.avst_ini[0]
            df3['Theta (deg)'] = [math.degrees(x) for x in self.avst_ini[1]]
            df3['Phi (deg)'] = [math.degrees(x) for x in self.avst_ini[2]]

            df4 = pd.DataFrame()
            df4['Settings Name'] = ['Model Resolution (A)',
                                    'Evaporation Rate (A/sec)',
                                    'Evaporation Time (sec)',
                                    'Grid space (A)',
                                    'Raycast Length (A)',
                                    'Model Limit (Pts)',
                                    'Average out divets',
                                    'Average out peaks',
                                    'Preserve corners',
                                    'Enforce Grid Space',
                                    'Enforce Model Limit',
                                    'GPU 0 Intersection',
                                    'GPU 0 Grid',
                                    'GPU 0 Model',
                                    'GPU 0 tArea',
                                    'GPU 0 Merge',
                                    'GPU D Intersection',
                                    'GPU D Grid',
                                    'GPU D Model',
                                    'GPU D Merge']
            df4['Settings Val'] = [self.model_resolution,
                                   self.evap_rate_text,
                                   self.evaporation_time,
                                   self.gridspace,
                                   self.raycast_length,
                                   self.model_limit,
                                   self.gui.app.checkBox_divet.checkState(),
                                   self.gui.app.checkBox_peaks.checkState(),
                                   self.gui.app.checkBox_corners.checkState(),
                                   self.gui.app.checkBox_grid.checkState(),
                                   self.gui.app.checkBox_modelRes.checkState(),
                                   self.epsIntersect,
                                   self.epsGrid,
                                   self.epsModel,
                                   self.epsModeltArea,
                                   self.epsMerge,
                                   self.decIntersect,
                                   self.decGrid,
                                   self.decModel,
                                   self.decMerge]

            df = pd.concat([df1, df2, df3, df4], axis=1)
            df.to_csv(filepath, index=False)
            log.info("Save complete - file saved to: %s" % filepath)

    def model_grid_gpu(self, input_x, input_y):
        """
        WARNING: Core code function

        :array input_x: np.array of type float64
        :array input_y: np.array of type float64

        :array output_x: np.array of type float64
        :array output_y: np.array of type float64

        TODO: ydim is static. find efficient method for dynamic grid allocation
        On first pass (model load) iterate through model, find the max
        distance between points, asses the current model resolution and
        from that define the ydim. After that, find efficient way to asses
        this same distance and adjust ydim accordingly.

        See block comment in grid_gpu for details.
        """
        xdim = len(input_x)
        ydim = 1000

        output_x = np.full(shape=(xdim, ydim),
                           fill_value=math.nan, dtype=np.float64)
        output_y = np.full(shape=(xdim, ydim),
                           fill_value=math.nan, dtype=np.float64)

        bpg_x = (output_x.shape[0] + self.tpb_2d[0]) // self.tpb_2d[0]
        bpg_y = (output_x.shape[1] + self.tpb_2d[1]) // self.tpb_2d[1]
        bpg_2d = (bpg_x, bpg_y)

        grid_gpu[bpg_2d, self.tpb_2d](
            input_x, input_y,
            output_x, output_y, self.model_resolution,
            self.epsGrid, self.decGrid)

        output_x = output_x.reshape(1, xdim * ydim)
        output_y = output_y.reshape(1, xdim * ydim)

        output_x = output_x[~np.isnan(output_x)]
        output_y = output_y[~np.isnan(output_y)]

        return output_x, output_y

    def calc_intersection(self, input_x, input_y, angle):
        '''
        WARNING: Core Code Function

        :array input_x: np.array of type float64
        :array input_y: np.array of type float64
        :array output_i: np.array of type int8. Is the intersection results

        See intersection_gpu method for block comment details.
        '''
        output_i = np.full(len(input_x), 0, dtype=np.int8)

        bpg_x = (len(input_x) + self.tpb_2d[0]) // self.tpb_2d[0]
        bpg_y = (len(input_y) + self.tpb_2d[1]) // self.tpb_2d[1]
        bpg_2d = (bpg_x, bpg_y)

        intersection_gpu[bpg_2d, self.tpb_2d](
            input_x, input_y, output_i,
            angle, self.raycast_length,
            self.epsIntersect, self.decIntersect)
        return output_i

    def model_update_gpu(self, input_x, input_y, input_i, theta, phi):
        """
        Take intersection data and model, add material to model according to
        the evaporation rate

        :array input_x: np.array of type float64
        :array input_y: np.array of type float64
        :array output_i: np.array of type int8. Is the intersection results

        See block comment in the model_gpu method for details.

        Using math.nan to fill in 'empty' values of the output arrays. Valid
        results replace the nan values. Leftover nan's are stripped.
        """
        xdim = input_x.shape[0]
        ydim = 2

        output_x = np.full(shape=(xdim, ydim),
                           fill_value=math.nan, dtype=np.float64)
        output_y = np.full(shape=(xdim, ydim),
                           fill_value=math.nan, dtype=np.float64)
        output_i = np.full(shape=(xdim, ydim),
                           fill_value=math.nan, dtype=np.float64)

        rate = self.evaporation_rate
        Rx = round(self.raycast_length * math.sin(theta), 10)
        Ry = round(self.raycast_length * math.cos(theta), 10)
        Rz = round(self.raycast_length * math.sin(phi), 10)

        bpg = int(np.ceil(xdim / self.tpb))

        model_gpu[bpg, self.tpb](
            input_x, input_y, input_i,
            theta, Rx, Ry, Rz, rate,
            output_x, output_y, output_i,
            self.average_divets, self.average_peaks, self.corner,
            self.epsModel, self.epsModeltArea, self.decModel,
            self.growthXi, self.growthDirection)

        output_x = output_x.reshape(1, xdim * ydim)
        output_y = output_y.reshape(1, xdim * ydim)
        output_i = output_i.reshape(1, xdim * ydim)

        output_x = output_x[~np.isnan(output_x)]
        output_y = output_y[~np.isnan(output_y)]
        output_i = output_i[~np.isnan(output_i)]

        return output_x, output_y, output_i

    def model_merge(self, input_x, input_y, input_i):
        """
        See block comment in method merge_gpu for details.
        """
        xdim = input_x.shape[0]

        output_x = np.full(xdim, fill_value=math.nan, dtype=np.float64)
        output_y = np.full(xdim, fill_value=math.nan, dtype=np.float64)

        bpg = int(np.ceil(xdim / self.tpb))

        merge_gpu[bpg, self.tpb](
            input_x, input_y, input_i,
            output_x, output_y, self.gridspace,
            self.epsMerge, self.decMerge)

        output_x = output_x[~np.isnan(output_x)]
        output_y = output_y[~np.isnan(output_y)]

        return output_x, output_y
        # return input_x, input_y

    def model_derez(self, input_x, input_y):
        """Takes current model x,y data and selects points at a minimum of
        model resolution distance away from each other. This should slim down
        the weight of the model in terms of calculations. Particularly for
        curved surfaces that start growing.
        """
        xn = []
        yn = []
        i = 0
        j = 0
        while j < len(input_x):
            if j == 0 or j == len(input_x) - 1:
                xn.append(input_x[j])
                yn.append(input_y[j])
                j += 1
            else:
                Wx = input_x[j] - input_x[i]
                Wy = input_y[j] - input_y[i]
                W = math.sqrt(Wx ** 2 + Wy ** 2)
                if W > self.model_resolution:
                    xn.append(input_x[j])
                    yn.append(input_y[j])
                    i = j
                    j += 1
                else:
                    j += 1

        return np.array(xn), np.array(yn)
