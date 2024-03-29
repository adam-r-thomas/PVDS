
import math
import time
import sys

from numba import cuda

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog

import matplotlib.pyplot as plt
from matplotlib import animation


import numpy as np
import pandas as pd
from pathlib import Path
from itertools import cycle

from evapsim.physics import (intersection_cpu, intersection_gpu,
                             grid_cpu, grid_gpu,
                             model_cpu, model_gpu,
                             merge_cpu, merge_gpu)

from evapsim.dialogs import SimulatorWindow

import logging
log = logging.getLogger("evapsim")

try:
    assert cuda.detect()
    device = cuda.get_current_device()
    tpb = device.WARP_SIZE
    tpb_2d = (tpb // 2, tpb // 2)

except AssertionError:
    log.info("No GPU found with CUDA capabilities. Switching to CPU mode.")
    device = None
    tpb = 0
    tpb_2d = (0, 0)


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
        # corrects for evaporation rate if not per second
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

            if len(self.model_x) > self.model_limit and self.boolModelRes:
                print("Model size limit hit. Reducing model points.")
                log.info("Model size limit hit. Reducing model points.")
                self.model_x, self.model_y = self.model_reduce(
                    self.model_x, self.model_y)

            # Run model intersection test
            try:
                log.info("Intersection Test")
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
                log.info("Model Update")
                self.vert_x, self.vert_y, self.vert_i = self.calc_model(
                    self.model_x,
                    self.model_y,
                    self.intersect_result,
                    angle, phi)
            except:  # noqa: I do not know all failure modes from GPU
                tb = sys.exc_info()
                log.exception(tb)
                log.error("Failure on self.calc_model")
                print("Error occurred with self.calc_model")
                self.gui.app.pushButton_Pause.click()
                return

            # Merge vertices that are too close
            try:
                if self.grid:
                    log.info("Merge Vertices")
                    self.merge_x, self.merge_y = self.calc_merge(
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
                log.info("Grid Vertices")
                self.model_x, self.model_y = self.calc_grid(self.merge_x,
                                                            self.merge_y)
            except:  # noqa: I do not know all failure modes from GPU
                tb = sys.exc_info()
                log.exception(tb)
                log.error("Failure on self.calc_grid")
                print("Error occurred with self.calc_grid")
                self.gui.app.pushButton_Pause.click()
                return

            # Draw results of intersection test
            self.line_static.set_data(self.model_ini[0], self.model_ini[1])
            # self.line_ani.set_data(self.vert_x, self.vert_y)
            self.line_ani.set_data(self.model_x, self.model_y)

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
        if self.gui.app.radioButton_costanrule.isChecked():
            self.growthDirection: 'costan' = 3

        if self.gui.app.radioButton_growthangle_a.isChecked():
            self.growthAngle = 0
        if self.gui.app.radioButton_growthangle_t.isChecked():
            self.growthAngle = 1

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

                self.model_x, self.model_y = self.calc_grid(
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
                                 usecols=['Settings Name', 'Settings Val'],
                                 index_col=0)

                self.gui.app.lineEdit_Model_Resolution.setText(
                    str(df.loc['Model Resolution (A)'][0]))
                self.gui.app.lineEdit_Evaporation_Rate.setText(
                    str(df.loc['Evaporation Rate (A/sec)'][0]))
                self.gui.app.lineEdit_Evaporation_Time.setText(
                    str(df.loc['Evaporation Time (sec)'][0]))
                self.gui.app.lineEdit_Grid_Space.setText(
                    str(df.loc['Grid space (A)'][0]))
                self.gui.app.lineEdit_Raycast_Length.setText(
                    str(df.loc['Raycast Length (A)'][0]))
                self.gui.app.lineEdit_Model_Limit.setText(
                    str(df.loc['Model Limit (Pts)'][0]))

                self.gui.app.checkBox_divet.setChecked(
                    df.loc['Average out divets'][0])
                self.gui.app.checkBox_peaks.setChecked(
                    df.loc['Average out peaks'][0])
                self.gui.app.checkBox_corners.setChecked(
                    df.loc['Preserve corners'][0])
                self.gui.app.checkBox_grid.setChecked(
                    df.loc['Enforce Grid Space'][0])
                self.gui.app.checkBox_modelRes.setChecked(
                    df.loc['Enforce Model Limit'][0])

                self.gui.app.lineEdit_epsIntersect.setText(
                    str(df.loc['GPU 0 Intersection'][0]))
                self.gui.app.lineEdit_epsGrid.setText(
                    str(df.loc['GPU 0 Grid'][0]))
                self.gui.app.lineEdit_epsModel.setText(
                    str(df.loc['GPU 0 Model'][0]))
                self.gui.app.lineEdit_epsModeltArea.setText(
                    str(df.loc['GPU 0 tArea'][0]))
                self.gui.app.lineEdit_epsMerge.setText(
                    str(df.loc['GPU 0 Merge'][0]))

                self.gui.app.lineEdit_decIntersect.setText(
                    str(int(df.loc['GPU D Intersection'][0])))
                self.gui.app.lineEdit_decGrid.setText(
                    str(int(df.loc['GPU D Grid'][0])))
                self.gui.app.lineEdit_decModel.setText(
                    str(int(df.loc['GPU D Model'][0])))
                self.gui.app.lineEdit_decMerge.setText(
                    str(int(df.loc['GPU D Merge'][0])))

                self.gui.app.lineEdit_growth_rate_Xi.setText(
                    str(df.loc['Directional Dependence'][0]))

                self.growthDirection = int(df.loc['Growth Type'][0])
                if self.growthDirection == 1:
                    self.gui.app.radioButton_cosinerule.setChecked(True)
                elif self.growthDirection == 2:
                    self.gui.app.radioButton_tangentrule.setChecked(True)
                elif self.growthDirection == 3:
                    self.gui.app.radioButton_costanrule.setChecked(True)
                else:
                    self.gui.app.radioButton_linearrule.setChecked(True)

                self.growthAngle = int(df.loc['Growth Angle'][0])
                if self.growthAngle == 0:
                    self.gui.app.radioButton_growthangle_a.setChecked(True)
                if self.growthAngle == 1:
                    self.gui.app.radioButton_growthangle_t.setChecked(True)

                log.info("Settings file loaded: %s" % filepath)
            except (AssertionError, ValueError, KeyError) as _error:
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
                                    'GPU D Merge',
                                    'Directional Dependence',
                                    'Growth Type',
                                    'Growth Angle']

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
                                   self.decMerge,
                                   self.growthXi,
                                   self.growthDirection,
                                   self.growthAngle]

            df = pd.concat([df1, df2, df3, df4], axis=1)
            df.to_csv(filepath, index=False)
            log.info("Save complete - file saved to: %s" % filepath)

    def calc_grid(self, input_x, input_y):
        """See block comment in physics.grid for details.

        TODO: ydim is static. find efficient method for dynamic grid allocation
        On first pass (model load) iterate through model, find the max
        distance between points, asses the current model resolution and
        from that define the ydim. After that, find efficient way to asses
        this same distance and adjust ydim accordingly.

        """
        xdim = len(input_x)
        ydim = 1000

        output_x = np.full(shape=(xdim, ydim),
                           fill_value=math.nan, dtype=np.float64)
        output_y = np.full(shape=(xdim, ydim),
                           fill_value=math.nan, dtype=np.float64)

        if device:
            bpg_x = (output_x.shape[0] + tpb_2d[0]) // tpb_2d[0]
            bpg_y = (output_x.shape[1] + tpb_2d[1]) // tpb_2d[1]
            bpg_2d = (bpg_x, bpg_y)
            grid_gpu[bpg_2d, tpb_2d](
                input_x, input_y,
                output_x, output_y, self.model_resolution,
                self.epsGrid, self.decGrid)
        else:  # No GPU
            grid_cpu(
                input_x, input_y,
                output_x, output_y, self.model_resolution,
                self.epsGrid, self.decGrid)

        output_x = output_x.reshape(1, xdim * ydim)
        output_y = output_y.reshape(1, xdim * ydim)

        output_x = output_x[~np.isnan(output_x)]
        output_y = output_y[~np.isnan(output_y)]

        return output_x, output_y

    def calc_intersection(self, input_x, input_y, angle):
        '''See physics.intersection method for block comment details.
        '''
        output_i = np.full(len(input_x), 0, dtype=np.int8)

        if device:
            bpg_x = (len(input_x) + tpb_2d[0]) // tpb_2d[0]
            bpg_y = (len(input_y) + tpb_2d[1]) // tpb_2d[1]
            bpg_2d = (bpg_x, bpg_y)
            intersection_gpu[bpg_2d, tpb_2d](
                input_x, input_y, output_i,
                angle, self.raycast_length,
                self.epsIntersect, self.decIntersect)
        else:  # No GPU
            intersection_cpu(
                input_x, input_y, output_i,
                angle, self.raycast_length,
                self.epsIntersect, self.decIntersect)

        return output_i

    def calc_model(self, input_x, input_y, input_i, theta, phi):
        """See block comment in the phyics.model method for details.

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
        # print("Self Tests")
        if device:
            bpg = int(np.ceil(xdim / tpb))
            model_gpu[bpg, tpb](
                input_x, input_y, input_i,
                theta, Rx, Ry, Rz, rate,
                output_x, output_y, output_i,
                self.average_divets, self.average_peaks, self.corner,
                self.epsModel, self.epsModeltArea, self.decModel,
                self.growthXi, self.growthDirection, self.growthAngle)
        else:  # No GPU
            model_cpu(
                input_x, input_y, input_i,
                theta, Rx, Ry, Rz, rate,
                output_x, output_y, output_i,
                self.average_divets, self.average_peaks, self.corner,
                self.epsModel, self.epsModeltArea, self.decModel,
                self.growthXi, self.growthDirection, self.growthAngle)

        output_x = output_x.reshape(1, xdim * ydim)
        output_y = output_y.reshape(1, xdim * ydim)
        output_i = output_i.reshape(1, xdim * ydim)

        output_x = output_x[~np.isnan(output_x)]
        output_y = output_y[~np.isnan(output_y)]
        output_i = output_i[~np.isnan(output_i)]

        return output_x, output_y, output_i

    def calc_merge(self, input_x, input_y, input_i):
        """
        See block comment in method physics.merge for details.
        """
        xdim = input_x.shape[0]

        output_x = np.full(xdim, fill_value=math.nan, dtype=np.float64)
        output_y = np.full(xdim, fill_value=math.nan, dtype=np.float64)

        if device:
            bpg = int(np.ceil(xdim / tpb))
            merge_gpu[bpg, tpb](
                input_x, input_y, input_i,
                output_x, output_y, self.gridspace,
                self.epsMerge, self.decMerge)
        else:
            merge_cpu(
                input_x, input_y, input_i,
                output_x, output_y, self.gridspace,
                self.epsMerge, self.decMerge)

        output_x = output_x[~np.isnan(output_x)]
        output_y = output_y[~np.isnan(output_y)]

        return output_x, output_y
        # return input_x, input_y

    def model_reduce(self, input_x, input_y):
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
