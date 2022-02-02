# -*- coding: utf-8 -*-
'''
Created on Oct 5, 2021

@author: Adam Thomas
ma043604

Reviewing the E-Simulator-044 Release-001 Code
Created on Oct 17, 2017

TODO: cProfile

TODO: Possible low res pass on a model when vertice density gets too high
Example is sin wave starting at 2000 points ends at 2 million points after
500nm evaporation

I'm not focused on improving the animation speed itself. I do not have a
good handling with GUI. The main calculation however, can be used in
other programs that implement faster drawing.

See https://docs.continuum.io/numbapro/CUDAJit/
for the documentation to CUDA
and https://people.duke.edu/~ccc14/sta-663/CUDAPython.html
for an introduction to GPU programming in general

NOTE:
When installing LiClipse / Eclipse the Enivronment PATH  variable needs
to be setup.
Run conda from the menu such that the conda env is active
>>> PATH
Copy the respective paths
Preferences>PyDev>Interpreters>Python Interpreters> Select Interpreter
Make new Environment variable PATH that is the copied paths from above
'''

# import cProfile
import math
import time
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg,
                                                NavigationToolbar2QT)

from numba import cuda
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import cycle

import logging

logging.basicConfig(filename='Evap_Sim.log',
                    encoding='utf-8',
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    filemode='w')
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

stream = logging.StreamHandler()
stream.setLevel(logging.DEBUG)
log.addHandler(stream)

assert cuda.detect()

device = cuda.get_current_device()
tpb = device.WARP_SIZE
tpb_2d = (tpb // 2, tpb // 2)
# max block size 1024

log.info("CUDA device loaded. Starting GPU methods:")


@cuda.jit('void(float64[:], float64[:], int8[:], float64, float64)')
def intersection_gpu(Px, Py, Vi, raycast_sin_angle, raycast_cos_angle):
    """
    Core calculation code : This is sent to the Nvidia GPU
    See:
    https://www.codeproject.com/Tips/862988/Find-the-Intersection-Point-of-Two-Line-Segments
    https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
    for a detailed explanation of the method I followed for implementing this
    into the GPU.

    :array Px: numpy array of the x data points of the model
    :array Py: numpy array of the y data points of the model
    :array Vi: empty numpy array of same size of Px or Py. This holds the
    intersection test results. 0 is no intersection, 1 is intersection

    :float raycast_sin_angle: rounded math.sin(angle) of deposition angle
    :float raycast_cos_angle: rounded math.cos(angle) of deposition angle 
    """  # noqa
    epsilon = 1e-16
    i, j = cuda.grid(2)
    if i < Px.shape[0] and j < Py.shape[0] - 1:
        if i == j or i == j + 1:
            pass  # Line intersecting itself

        else:
            Rx = raycast_sin_angle
            Ry = raycast_cos_angle
            Sx = Px[j + 1] - Px[j]
            Sy = Py[j + 1] - Py[j]
            QPx = Px[j] - Px[i]
            QPy = Py[j] - Py[i]
            PQx = Px[i] - Px[j]
            PQy = Py[i] - Py[j]
            RxS = (Rx * Sy) - (Ry * Sx)
            QPxR = (QPx * Ry) - (QPy * Rx)
            if (math.fabs(RxS) <= epsilon) and (math.fabs(QPxR) <= epsilon):
                # Collinear: Overlapping lines is considered intersecting
                qpr = QPx * Rx + QPy * Ry
                pqs = PQx * Sx + PQy * Sy
                RR = Rx * Rx + Ry * Ry
                SS = Sx * Sx + Sy * Sy
                if (0 <= qpr and qpr <= RR) or (0 <= pqs and pqs <= SS):
                    # The two lines are collinear and overlapping
                    Vi[i] = 1
                else:
                    pass  # The two lines are collinear but disjoint
            elif (math.fabs(RxS) <= epsilon) \
                    and not (math.fabs(QPxR) <= epsilon):
                pass  # Parallel and Non-Intersecting
            else:
                t = (QPx * Sy - QPy * Sx) / RxS
                u = (QPx * Ry - QPy * Rx) / RxS
                if not (RxS <= epsilon) and (0.0 <= t and t <= 1.0) and \
                        (0.0 <= u and u <= 1.0):
                    # Intersection found = model_grid + t*r
                    Vi[i] = 1
                else:
                    # The lines are not parallel but do not intersect
                    pass


log.info("Method: intersection_gpu has loaded.")


@cuda.jit('void(float64[:], float64[:], float64[:,:], float64[:,:], float64)')
def grid_gpu(Px, Py, Vx, Vy, model_resolution):
    '''
    TODO: Better implementation of a ragged list from numpy needed
    Currently the array is a defined size in y (mirrored of x). If it exceeds
    the x dimension this method will fail.

    The model contains a set of vertices i, each i then contains a set of
    points j that are spaced between i and i+1. The model is not square as
    the distance between each i may vary. So upon export, flatten the array.

    Take vertices from model and get grid points. Points are used in
    raycasting and intersection tests.

    In order to correctly place vertices along the lines I need to
    determine the direction of the line with respect to the model. Such
    that a line is not drawn backwards. In other words: If I wanted to draw
    a shape without lifting my pencil, what would be the order of points
    along the grid in order to achieve this.
    '''
    epsilon = 1e-16
    i, j = cuda.grid(2)
    dec_acc = 6  # round off floating points errors

    if i == Px.shape[0] - 1 and j == 0:
        # Close the shape
        Vx[i, j] = Px[i]
        Vy[i, j] = Py[i]

    elif i < Px.shape[0] - 1:
        # Segment width (W)
        Wx = Px[i + 1] - Px[i]
        Wy = Py[i + 1] - Py[i]
        W = math.sqrt(Wx ** 2 + Wy ** 2)
        grid_points = math.floor(W / model_resolution)

        if grid_points == 0 and j == 0:
            # The vertices happen to be smaller distance than resolution
            Vx[i, j] = Px[i]
            Vy[i, j] = Py[i]

        elif grid_points != 0:
            x = Wx / grid_points
            xs = math.copysign(1.0, x)
            x = xs * x
            y = Wy / grid_points
            ys = math.copysign(1.0, y)
            y = ys * y

            if j < grid_points:
                if math.fabs(Wx) <= epsilon:  # line is vertical
                    if Wy > 0:  # positive
                        Vx[i, j] = Px[i]
                        Vy[i, j] = round(Py[i] + (y * j), dec_acc)
                    else:  # negative
                        Vx[i, j] = Px[i]
                        Vy[i, j] = round(Py[i] - (y * j), dec_acc)
                elif math.fabs(Wy) <= epsilon:  # line is horizontal
                    if Wx > 0:  # positive
                        Vx[i, j] = round(Px[i] + (x * j), dec_acc)
                        Vy[i, j] = Py[i]
                    else:  # negative
                        Vx[i, j] = round(Px[i] - (x * j), dec_acc)
                        Vy[i, j] = Py[i]
                else:  # line is sloped
                    if Wx > 0 and Wy > 0:  # Slope in Q1
                        Vx[i, j] = round(Px[i] + (x * j), dec_acc)
                        Vy[i, j] = round(Py[i] + (y * j), dec_acc)
                    elif Wx < 0 and Wy > 0:  # Slope in Q2
                        Vx[i, j] = round(Px[i] - (x * j), dec_acc)
                        Vy[i, j] = round(Py[i] + (y * j), dec_acc)
                    elif Wx < 0 and Wy < 0:  # Slope in Q3
                        Vx[i, j] = round(Px[i] - (x * j), dec_acc)
                        Vy[i, j] = round(Py[i] - (y * j), dec_acc)
                    else:  # Slope in Q4
                        Vx[i, j] = round(Px[i] + (x * j), dec_acc)
                        Vy[i, j] = round(Py[i] - (y * j), dec_acc)


log.info("Method: grid_gpu has loaded.")


@cuda.jit('void(float64[:], float64[:], int8[:], float64, float64, float64, float64, float64[:,:], float64[:,:], float64[:,:])')  # noqa
def model_gpu(Px, Py, Pi, angle, Rx, Ry, rate, Vx, Vy, Vi):
    '''
    WARNING: Core Code Function

    This is where material is added to the model. I have yet to perfect a
    solution to only adding needed points without accidently dropping some. So
    for now, I am keeping all points added to the model. It is not that
    efficient to do so. However, since modern GPU's are really good at handling
    tons of points I am not too worried at this moment. The only thing
    that will slow this down is np.array handling to remove excess nan values
    when adding new points to the model.

    Take intersection data and model, add material to model according to
    the evaporation rate

    There are a set of assumptions made on how the evaporation is adding
    material to the model. From these assumptions we can extrapolate how
    vertices near or in shaded region can be handled to save on
    computational time.

        1: A point that has equal evaporation to either side is considered a
        point with no information. However, should this point also be not
        'straight' with respect to its nearest neighbors it contains
        information of the model itself.

        2: Information is contained in a point or set of points that do not
        have equal evaporation on either side

        3: The evaporation between 2 points that have no information can be
        omitted

    With this set of rules only points that have varying evaporations
    contain the information to replot the model with minimal points. The
    points that are corners of the model that have no evaporation are also
    preserved.

    These points are the new vertices of the model.

    The points inbetween vertices are no longer needed as the model will
    regrid these vertices to have an equal resolution.
    '''
    epsilon = 1e-10
    i = cuda.grid(1)
    dec_acc = 4  # round off floating points errors

    if i == 0 or i == Px.shape[0] - 1 or Pi[i] != 0:
        # Pin start of model | Pin end of model | point in shadow - no change
        # For start/end of model - possible to preserve point and add material?
        # Get some odd growth near the ends (forcing them to be zero there)
        Vx[i, 0] = Px[i]
        Vy[i, 0] = Py[i]
        Vi[i, 0] = Pi[i]

    elif i < Px.shape[0] - 1 and i > 1:
        # Point is now assumed to be in evaporant
        # Determine from a set of 3 points A:i-1, B:i, C:i+1 if it is straight
        # value should be zero (smaller than epsilon). Is area of triangle
        straight = (Px[i - 1] * (Py[i] - Py[i + 1])
                    + Px[i] * (Py[i + 1] - Py[i - 1])
                    + Px[i + 1] * (Py[i - 1] - Py[i]))

        if math.fabs(straight) <= epsilon:
            # Line segment is straight
            Sx = Px[i + 1] - Px[i - 1]
            Sy = Py[i + 1] - Py[i - 1]
            numerator = (Rx * Sx) + (Ry * Sy)
            denominator = math.sqrt(Rx ** 2 + Ry ** 2) \
                * math.sqrt(Sx ** 2 + Sy ** 2)
            theta = math.acos(
                max(-1.0, min(1.0, (numerator / denominator))))
            t = math.sin(theta) * rate
            Ax = Px[i] + t * math.sin(angle)
            Ay = Py[i] + t * math.cos(angle)
            Vx[i, 0] = round(Ax, dec_acc)
            Vy[i, 0] = round(Ay, dec_acc)
            Vi[i, 0] = 0

        else:
            # A corner: check how evaporation is landing around it
            if Pi[i - 1] != 0 and Pi[i + 1] != 0:
                # A divet or sharp point that has evaporation on it but not
                # its nearest neighbors. Considered to be non-real and is
                # averaged with its nearest neighbors
                Sx = round((Px[i + 1] + Px[i - 1]) / 2.0, dec_acc)
                Sy = round((Py[i + 1] + Py[i - 1]) / 2.0, dec_acc)
                Vx[i, 0] = Sx
                Vy[i, 0] = Sy
                Vi[i, 0] = 1

            elif Pi[i - 1] == 0 and Pi[i + 1] == 0:
                # Corner in evap with either side having different evap t
                # A : Left point of vertice
                Sx = Px[i] - Px[i - 1]
                Sy = Py[i] - Py[i - 1]
                numerator = (Rx * Sx) + (Ry * Sy)
                denominator = math.sqrt(Rx ** 2 + Ry ** 2) * \
                    math.sqrt(Sx ** 2 + Sy ** 2)
                theta = math.acos(
                    max(-1.0, min(1.0, (numerator / denominator))))
                t1 = math.sin(theta) * rate
                p0_x = Px[i - 1] + t1 * math.sin(angle)
                p0_y = Py[i - 1] + t1 * math.cos(angle)
                p1_x = Px[i] + t1 * math.sin(angle)
                p1_y = Py[i] + t1 * math.cos(angle)

                # B : Right point of vertice
                Sx = Px[i + 1] - Px[i]
                Sy = Py[i + 1] - Py[i]
                numerator = (Rx * Sx) + (Ry * Sy)
                denominator = math.sqrt(Rx ** 2 + Ry ** 2) * \
                    math.sqrt(Sx ** 2 + Sy ** 2)
                theta = math.acos(
                    max(-1.0, min(1.0, (numerator / denominator))))
                t2 = math.sin(theta) * rate
                p3_x = Px[i] + t2 * math.sin(angle)
                p3_y = Py[i] + t2 * math.cos(angle)
                p2_x = Px[i + 1] + t2 * math.sin(angle)
                p2_y = Py[i + 1] + t2 * math.cos(angle)
                # Start intersection calculation
                Rx = (p1_x - p0_x)
                Ry = (p1_y - p0_y)
                Sx = (p3_x - p2_x)
                Sy = (p3_y - p2_y)

                RxS = (Rx * Sy) - (Ry * Sx)
                QPx = p2_x - p0_x
                QPy = p2_y - p0_y

                if not math.fabs(RxS) <= epsilon:  # Avoid divide by zeros
                    s = (QPx * Ry - QPy * Rx) / RxS
                    t = (QPx * Sy - QPy * Sx) / RxS
                    s = round(s, 10)
                    t = round(t, 10)
                    # If either segment has a valid intersection point.
                    # Use that as the growth point
                    if t >= 0 and t <= 1.0:
                        i_x = p0_x + (t * Rx)
                        i_y = p0_y + (t * Ry)
                        Vx[i, 0] = round(i_x, dec_acc)
                        Vy[i, 0] = round(i_y, dec_acc)
                        Vi[i, 0] = s  # 0
                        Vi[i, 1] = t
                    elif s >= 0 and s <= 1.0:
                        i_x = p2_x + (s * Sx)
                        i_y = p2_y + (s * Sy)
                        Vx[i, 0] = round(i_x, dec_acc)
                        Vy[i, 0] = round(i_y, dec_acc)
                        Vi[i, 0] = s  # 0
                        Vi[i, 1] = t
                    else:
                        # The interesction is not within reasonable bounds at
                        # the corner.
                        # Check if co-linear If so add line like a flat evap
                        straight = (Px[i - 1] * (Py[i] - Py[i + 1]) +
                                    Px[i] * (Py[i + 1] - Py[i - 1]) +
                                    Px[i + 1] * (Py[i - 1] - Py[i]))
                        straight = round(straight, 10)
                        if math.fabs(straight) <= epsilon:
                            # Line segment is roughly straight
                            Sx = Px[i + 1] - Px[i - 1]
                            Sy = Py[i + 1] - Py[i - 1]
                            numerator = (Rx * Sx) + (Ry * Sy)
                            denominator = math.sqrt(Rx ** 2 + Ry ** 2) * \
                                math.sqrt(Sx ** 2 + Sy ** 2)
                            theta = math.acos(
                                max(-1.0, min(1.0, (numerator / denominator))))
                            t = math.sin(theta) * rate
                            Ax = Px[i] + t * math.sin(angle)
                            Ay = Py[i] + t * math.cos(angle)
                            Vx[i, 0] = round(Ax, dec_acc)
                            Vy[i, 0] = round(Ay, dec_acc)
                            Vi[i, 0] = 0
                # else:
                #     pass  # TODO: Check real co-linear cases

            elif Pi[i - 1] != 0:
                # Shaded corner evap | Preserve the i point
                Sx = Px[i + 1] - Px[i]
                Sy = Py[i + 1] - Py[i]
                numerator = (Rx * Sx) + (Ry * Sy)
                denominator = math.sqrt(Rx ** 2 + Ry ** 2) * \
                    math.sqrt(Sx ** 2 + Sy ** 2)
                theta = math.acos(
                    max(-1.0, min(1.0, (numerator / denominator))))
                t = math.sin(theta) * rate
                Ax = Px[i] + t * math.sin(angle)
                Ay = Py[i] + t * math.cos(angle)
                Vx[i, 0] = Px[i]
                Vy[i, 0] = Py[i]
                Vi[i, 0] = 1
                Vx[i, 1] = round(Ax, dec_acc)
                Vy[i, 1] = round(Ay, dec_acc)
                Vi[i, 1] = 0

            elif Pi[i + 1] != 0:
                # Shaded corner evap | Preserve the i point
                Sx = Px[i] - Px[i - 1]
                Sy = Py[i] - Py[i - 1]
                numerator = (Rx * Sx) + (Ry * Sy)
                denominator = math.sqrt(Rx ** 2 + Ry ** 2) * \
                    math.sqrt(Sx ** 2 + Sy ** 2)
                theta = math.acos(
                    max(-1.0, min(1.0, (numerator / denominator))))
                t = math.sin(theta) * rate
                Ax = Px[i] + t * math.sin(angle)
                Ay = Py[i] + t * math.cos(angle)
                Vx[i, 0] = round(Ax, dec_acc)
                Vy[i, 0] = round(Ay, dec_acc)
                Vi[i, 0] = 0
                Vx[i, 1] = Px[i]
                Vy[i, 1] = Py[i]
                Vi[i, 1] = 1


log.info("Method: model_gpu has loaded.")


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], float64)')
def merge_gpu(Px, Py, Vx, Vy, gridspace):
    """
    Merge points together that fall within the defined gridspace. Currently
    implemented as a simple merge function. My fancier versions produced some
    interesting artifacts.

    If two vertices fall within the model's resolution merge them by averaging
    their values. This leads to 'rounded' corners but greatly helps reduce
    impossible shapes from forming.
    """
    i = cuda.grid(1)

    if i == 0 or i == Px.shape[0] - 1:
        # Pin start of model | Pin end of model
        Vx[i] = Px[i]
        Vy[i] = Py[i]

    else:
        Wx = Px[i + 1] - Px[i]
        Wy = Py[i + 1] - Py[i]
        W = math.sqrt(Wx ** 2 + Wy ** 2)
        grid_points = math.floor(W / gridspace)
        if grid_points != 0:
            # Append vertice
            Vx[i] = Px[i]
            Vy[i] = Py[i]
        else:
            Ux = (Px[i + 1] + Px[i - 1]) / 2.0
            Uy = (Py[i + 1] + Py[i - 1]) / 2.0
            Vx[i] = Ux
            Vy[i] = Uy


log.info("Method: merge_gpu has loaded.")


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
    loop_counter = 0

    def __init__(self):
        '''
        Build initial parameters of the evaporation simulator
        '''
        self.device = device
        self.tickrate = 1.0

        self.graph_ani = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax_ani = self.graph_ani.add_subplot(111, aspect='equal')
        self.ax_ani.grid(1)
        self.ax_ani.set_xlabel("Model Pos. X (A)", fontsize=12)
        self.ax_ani.set_ylabel("Model Pos. Y (A)", fontsize=12)
        self.ax_ani.set_xlim(auto=True)
        self.ax_ani.set_ylim(auto=True)

        self.line_ani, = self.ax_ani.plot([], [], 'r-', lw=2)
        self.line_angle, = self.ax_ani.plot([], [], 'g-', lw=0.5)
        self.time_text = self.ax_ani.text(
            0.01, 0.95, '', transform=self.ax_ani.transAxes)
        self.angle_text = self.ax_ani.text(
            0.01, 0.90, '', transform=self.ax_ani.transAxes)
        self.line_static, = self.ax_ani.plot([], [], 'k-', lw=2)

        self.graph_ani.tight_layout()
        log.info("Simulator init complete.")

    def run(self):
        '''
        Run the simulation and present results to animation graph
        '''
        log.info("Simulator run started.")
        self.simulation_parameters()

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
            # print(self.loop_counter)
            # self.loop_counter += 1
            # for i in range(self.model_x.shape[0]):
            #     print(i, "|", self.model_x[i], self.model_y[i], "|", self.vert_x[i], self.vert_y[i], "|", self.merge_x[i], self.merge_y[i])  # noqa
            #     print(i, "|", self.model_x[i], self.model_y[i], "|", self.vert_x[i], self.vert_y[i], "|", self.vert_i[i, 0], self.vert_i[i, 1])  # noqa

            timer_start = time.process_time()
            log.info("")
            log.info("Evap Step: %s" % timer_start)

            # Setup display info on graph
            self.time_text.set_text('Time: %.2f' % self.timer)
            self.timer += self.elapse_time
            angle = next(self.cycle_angle)
            self.angle_text.set_text('Angle: %.3f' % angle)
            log.info("Cycle: %s" % angle)
            log.info("Loop: %s" % self.loop_counter)

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
                return

            # Determine new vertices from the model on the grid | Adds material
            try:
                log.info("GPU: Model Update")
                self.vert_x, self.vert_y, self.vert_i = self.model_update_gpu(
                    self.model_x,
                    self.model_y,
                    self.intersect_result,
                    angle)
            except:  # noqa: I do not know all failure modes from GPU
                tb = sys.exc_info()
                log.exception(tb)
                log.error("Failure on self.model_update_gpu")
                print("Error occurred with self.model_update_gpu")
                return

            # Merge vertices that are too close
            try:
                log.info("GPU: Model Merge")
                self.merge_x, self.merge_y = self.model_merge(
                    self.vert_x,
                    self.vert_y)
            except:  # noqa: I do not know all failure modes from GPU
                tb = sys.exc_info()
                log.exception(tb)
                log.error("Failure on self.model_merge")
                print("Error occurred with self.model_merge")
                return

            # # Re-grid the model
            try:
                log.info("GPU: Model Grid")
                self.model_x, self.model_y = self.model_grid_gpu(self.merge_x,
                                                                 self.merge_y)
            except:  # noqa: I do not know all failure modes from GPU
                tb = sys.exc_info()
                log.exception(tb)
                log.error("Failure on self.model_grid_gpu")
                print("Error occurred with self.model_grid_gpu")
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
            # app.progress.set(100 * i // self.total_fps)
            app.progressBar.setProperty("value", 100 * i // self.total_fps)

            # Remaining calculations
            timer_end = time.process_time()
            # seconds = (timer_end - timer_start) * (self.total_fps - i)
            # m, s = divmod(seconds, 60)
            # h, m = divmod(m, 60)

            # Spacing in string to center text
            # app.progresstext.set(
            #     '                           Time Remaining: %d:%02d:%02d' % (h, m, s))  # noqa
            log.info("Evap Time: %s" % timer_end)
            self.loop_counter += 1
            # log.info("Evap Seconds: %s" % seconds)
            return (self.line_ani,
                    self.line_angle,
                    self.time_text,
                    self.angle_text,
                    self.line_static)

        fps = (len(self.avst_ini[0]) - 1) / self.avst_ini[0][-1]
        self.total_fps = int(fps * self.evaporation_time)
        self.cycle_angle = cycle(self.avst_ini[1])
        self.elapse_time = self.avst_ini[0][-1] / (len(self.avst_ini[0]) - 1)
        self.timer = 0

        self.simulation = animation.FuncAnimation(self.graph_ani,
                                                  animate,
                                                  frames=self.total_fps,
                                                  init_func=ani_init,
                                                  blit=False,
                                                  repeat=False)

        app.graphcanvas.draw_idle()

    def simulation_parameters(self):
        '''
        Starting parameters of the evaporation - pulls values from the GUI
        '''
        self.model_resolution = float(app.lineEdit_model_resolution.text())
        self.evaporation_rate = (float(app.lineEdit_evap_rate.text())
                                 * self.tickrate)

        self.evaporation_time = float(app.lineEdit_evap_time.text())
        self.gridspace = float(app.lineEdit_grid_space.text())
        self.raycast_length = float(app.lineEdit_raycast_length.text())
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
            assert '.csv' in filepath.suffix
            df = pd.read_csv(filepath, dtype=np.float64)
            assert 'Model x (A)' in df.columns
            assert 'Model y (A)' in df.columns

            self.model_ini = np.array([list(df['Model x (A)'].dropna()),
                                       list(df['Model y (A)'].dropna())])

            self.line_static.set_data(self.model_ini[0], self.model_ini[1])

            self.model_x, self.model_y = self.model_grid_gpu(self.model_ini[0],
                                                             self.model_ini[1])

            xpad = (max(self.model_ini[0]) - min(self.model_ini[0])) * 0.05
            ypad = (max(self.model_ini[1]) - min(self.model_ini[1])) * 0.05

            self.ax_ani.set_xlim(min(self.model_ini[0]) - xpad,
                                 max(self.model_ini[0]) + xpad)
            self.ax_ani.set_ylim(min(self.model_ini[1]) - ypad,
                                 max(self.model_ini[1]) + ypad)

            app.graphcanvas.draw_idle()
            log.info("Model file loaded: %s" % filepath)

    def load_csv_angletime(self):
        '''
        Load in time (col0) and angle (col1)

        Returns:
        :param tickrate: the average time step between each angle
        :array avst_ini: the angle vs time as a 2D numpy array.
            Angle = 0, Time = 1
        '''
        filepath, _ = QFileDialog.getOpenFileName(
            caption="Evaporation Simulator - Load angle vs time file",
            filter="*.csv")
        if filepath:
            filepath = Path(filepath)
            assert '.csv' in filepath.suffix
            df = pd.read_csv(filepath, dtype=np.float64)
            assert 'Time (sec)' in df.columns
            assert 'Angle (deg)' in df.columns

            list_deg = [math.radians(x) for x in list(df['Angle (deg)'].dropna())]  # noqa
            self.avst_ini = np.array([list(df['Time (sec)'].dropna()), list_deg])  # noqa

            tempa = []
            timelist = list(df['Time (sec)'].dropna())
            for i in range(1, (len(df['Time (sec)'].dropna()))):
                tempa.append(timelist[i] - timelist[i - 1])
            self.tickrate = np.array(tempa).mean()
            del tempa
            del timelist

            self.line_static.set_data(self.avst_ini[0], self.avst_ini[1])

            xpad = (max(self.avst_ini[0]) - min(self.avst_ini[0])) * 0.05
            ypad = (max(self.avst_ini[1]) - min(self.avst_ini[1])) * 0.05

            self.ax_ani.set_xlim(min(sim.avst_ini[0]) - xpad,
                                 max(sim.avst_ini[0]) + xpad)
            self.ax_ani.set_ylim(min(sim.avst_ini[1]) - ypad,
                                 max(sim.avst_ini[1]) + ypad)

            app.graphcanvas.draw_idle()
            log.info("AvsT file loaded: %s" % filepath)

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
            df3['Angle (deg)'] = [math.degrees(x) for x in self.avst_ini[1]]

            df4 = pd.DataFrame()
            df4['Model Resolution (A)'] = [self.model_resolution]
            df4['Evaporation Rate (A/sec)'] = [self.evaporation_rate
                                               / self.tickrate]
            df4['Evaporation Time (sec)'] = [self.evaporation_time]
            df4['Grid space (A)'] = [self.gridspace]
            df4['Raycast Length (A)'] = [self.raycast_length]

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

        bpg_x = (output_x.shape[0] + tpb_2d[0]) // tpb_2d[0]
        bpg_y = (output_x.shape[1] + tpb_2d[1]) // tpb_2d[1]
        bpg_2d = (bpg_x, bpg_y)

        grid_gpu[bpg_2d, tpb_2d](input_x, input_y,
                                 output_x, output_y,
                                 self.model_resolution)

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

        bpg_x = (len(input_x) + tpb_2d[0]) // tpb_2d[0]
        bpg_y = (len(input_y) + tpb_2d[1]) // tpb_2d[1]
        bpg_2d = (bpg_x, bpg_y)

        raycast_sin = round(
            self.raycast_length * math.sin(angle), 10)
        raycast_cos = round(
            self.raycast_length * math.cos(angle), 10)

        intersection_gpu[bpg_2d, tpb_2d](input_x, input_y,
                                         output_i,
                                         raycast_sin, raycast_cos)
        return output_i

    def model_update_gpu(self, input_x, input_y, input_i, angle):
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
        Rx = round(self.raycast_length * math.sin(angle), 10)
        Ry = round(self.raycast_length * math.cos(angle), 10)

        bpg = int(np.ceil(xdim / tpb))

        model_gpu[bpg, tpb](input_x, input_y, input_i,
                            angle, Rx, Ry, rate,
                            output_x, output_y, output_i)

        output_x = output_x.reshape(1, xdim * ydim)
        output_y = output_y.reshape(1, xdim * ydim)
        # output_i = output_i.reshape(1, xdim * ydim)

        output_x = output_x[~np.isnan(output_x)]
        output_y = output_y[~np.isnan(output_y)]
        # output_i = output_i[~np.isnan(output_i)]

        return output_x, output_y, output_i

    def model_merge(self, input_x, input_y):
        """
        See block comment in method merge_gpu for details.
        """
        xdim = input_x.shape[0]

        output_x = np.full(xdim, fill_value=math.nan, dtype=np.float64)
        output_y = np.full(xdim, fill_value=math.nan, dtype=np.float64)

        bpg = int(np.ceil(xdim / tpb))

        merge_gpu[bpg, tpb](input_x, input_y,
                            output_x, output_y,
                            self.gridspace)

        output_x = output_x[~np.isnan(output_x)]
        output_y = output_y[~np.isnan(output_y)]

        return output_x, output_y


class Application_Qt(object):
    """
    Convert .ui to .py files with:
    pyuic5 -x -o pyfilename.py design.ui

    Used Qt Designer to setup initial GUI implementation.
    """

    def __init__(self, simulator):
        self.sim = simulator
        self.MainWindow = QtWidgets.QMainWindow()
        self.setupUi(self.MainWindow)
        self.connectUi()

        self.graphcanvas = FigureCanvasQTAgg(sim.graph_ani)
        self.layoutPlot = QtWidgets.QWidget(self.graphicsView)

        self.gridPlot = QtWidgets.QGridLayout(self.layoutPlot)
        self.gridPlot.addWidget(self.graphcanvas)

        self.toolbar = NavigationToolbar2QT(self.graphcanvas,
                                            self.graphicsView)
        self.gridPlot.addWidget(self.toolbar)

        self.graphcanvas.draw_idle()
        # Buttons not ready yet
        # self.pushButton_reset_graph.hide()
        self.pushButton_reset.hide()
        log.info("Application window init complete")

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 700)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.frame_inputs = QtWidgets.QFrame(self.centralwidget)
        self.frame_inputs.setGeometry(QtCore.QRect(20, 20, 350, 635))
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
        self.splitter_left.setOrientation(QtCore.Qt.Vertical)
        self.splitter_left.setObjectName("splitter_left")
        self.pushButton_load_model = QtWidgets.QPushButton(self.splitter_left)
        self.pushButton_load_model.setObjectName("pushButton_load_model")
        self.pushButton_load_AvsT = QtWidgets.QPushButton(self.splitter_left)
        self.pushButton_load_AvsT.setObjectName("pushButton_load_AvsT")
        self.pushButton_save_model = QtWidgets.QPushButton(self.splitter_left)
        self.pushButton_save_model.setObjectName("pushButton_save_model")

        self.splitter_right = QtWidgets.QSplitter(self.frame_inputs)
        self.splitter_right.setGeometry(QtCore.QRect(215, 160, 130, 70))
        self.splitter_right.setOrientation(QtCore.Qt.Vertical)
        self.splitter_right.setObjectName("splitter_right")
        self.pushButton_show_model = QtWidgets.QPushButton(self.splitter_right)
        self.pushButton_show_model.setObjectName("pushButton_show_model")
        self.pushButton_show_AvsT = QtWidgets.QPushButton(self.splitter_right)
        self.pushButton_show_AvsT.setObjectName("pushButton_show_AvsT")
        self.pushButton_reset_graph = QtWidgets.QPushButton(self.splitter_right)  # noqa
        self.pushButton_reset_graph.setObjectName("pushButton_reset_graph")

        self.label_2 = QtWidgets.QLabel(self.frame_inputs)
        self.label_2.setGeometry(QtCore.QRect(5, 280, 350, 30))
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.frame_inputs)
        self.label_3.setGeometry(QtCore.QRect(5, 320, 350, 50))
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.frame_inputs)
        self.label_4.setGeometry(QtCore.QRect(5, 380, 350, 60))
        self.label_4.setWordWrap(True)
        self.label_4.setObjectName("label_4")
        self.label = QtWidgets.QLabel(self.frame_inputs)
        self.label.setGeometry(QtCore.QRect(5, 240, 350, 30))
        self.label.setWordWrap(True)
        self.label.setObjectName("label")

        self.pushButton_reset = QtWidgets.QPushButton(self.frame_inputs)
        self.pushButton_reset.setGeometry(QtCore.QRect(100, 610, 80, 23))
        self.pushButton_reset.setObjectName("pushButton_reset")
        self.progressBar = QtWidgets.QProgressBar(self.frame_inputs)
        self.progressBar.setGeometry(QtCore.QRect(5, 580, 340, 25))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.pushButton_quit = QtWidgets.QPushButton(self.frame_inputs)
        self.pushButton_quit.setGeometry(QtCore.QRect(5, 610, 80, 23))
        self.pushButton_quit.setObjectName("pushButton_quit")
        self.pushButton_start = QtWidgets.QPushButton(self.frame_inputs)
        self.pushButton_start.setGeometry(QtCore.QRect(265, 610, 80, 23))
        self.pushButton_start.setObjectName("pushButton_start")

        self.graphicsView = QtWidgets.QFrame(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(380, 10, 1000, 1000))
        self.graphicsView.setObjectName("graphicsView")

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
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
        self.pushButton_show_model.setText(
            _translate("MainWindow", "Show Model"))
        self.pushButton_show_AvsT.setText(
            _translate("MainWindow", "Show Angle vs Time"))
        self.pushButton_reset_graph.setText(
            _translate("MainWindow", "Reset Graph"))
        self.label_2.setText(
            _translate("MainWindow", "<b>Evaporation Rate:</b> Angstrom / sec of material deposition. Rate is affected by angle: cos(theta)."))  # noqa
        self.label_3.setText(
            _translate("MainWindow", "<b>Grid Space:</b> Smallest vertice resolution allowed. This helps prevent impossible shapes from forming. Two vertices that are closer then this set distance are merged."))  # noqa
        self.label_4.setText(
            _translate("MainWindow", "<b>Raycast Length:</b> Length of the ray cast from each vertice. Too short a ray can have evaporation in bad places. Too long a ray can produce math errors (float point problems)."))  # noqa
        self.label.setText(
            _translate("MainWindow", "<b>Model Resolution:</b> The density of the raycasts on the model."))  # noqa
        self.pushButton_reset.setText(
            _translate("MainWindow", "Reset"))
        self.pushButton_quit.setText(
            _translate("MainWindow", "Quit"))
        self.pushButton_start.setText(
            _translate("MainWindow", "Start"))
        log.info("Applicaiton UI translate complete")

    def connectUi(self):
        """
        Connnect buttons to the simulator
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

        self.pushButton_show_model.clicked.connect(self.displayModel)
        self.pushButton_show_AvsT.clicked.connect(self.displayWave)
        self.pushButton_reset_graph.clicked.connect(self.displayReset)

        self.pushButton_start.clicked.connect(self.sim.run)
        self.pushButton_quit.clicked.connect(self._quit)
        log.info("Application UI buttons connected")

    def displayWave(self):
        try:
            xpad = (max(self.sim.avst_ini[1])
                    - min(self.sim.avst_ini[1])) * 0.05
            ypad = (max(self.sim.avst_ini[0])
                    - min(self.sim.avst_ini[0])) * 0.05

            self.sim.line_static.set_data(self.sim.avst_ini[0],
                                          self.sim.avst_ini[1])

            self.sim.ax_ani.set_xlim(min(self.sim.avst_ini[0]) - xpad,
                                     max(self.sim.avst_ini[0]) + xpad)
            self.sim.ax_ani.set_ylim(min(self.sim.avst_ini[1]) - ypad,
                                     max(self.sim.avst_ini[1]) + ypad)
            self.graphcanvas.draw_idle()

        except AttributeError:
            log.info("No wave present to reset.")

    def displayModel(self):
        try:
            xpad = (max(self.sim.model_ini[0])
                    - min(self.sim.model_ini[0])) * 0.05
            ypad = (max(self.sim.model_ini[1])
                    - min(self.sim.model_ini[1])) * 0.05

            self.sim.line_static.set_data(self.sim.model_ini[0],
                                          self.sim.model_ini[1])
            self.sim.ax_ani.set_xlim(min(self.sim.model_ini[0]) - xpad,
                                     max(self.sim.model_ini[0]) + xpad)
            self.sim.ax_ani.set_ylim(min(self.sim.model_ini[1]) - ypad,
                                     max(self.sim.model_ini[1]) + ypad)
            self.graphcanvas.draw_idle()
        except AttributeError:
            log.info("No model present to reset.")

    def displayReset(self):
        self.toolbar._update_view()
        self.toolbar.home()

    def _quit(self):
        log.info("Program End")
        sys.exit(window.exit())


if __name__ == '__main__':
    log.info("Program Start")
    sim = Simulator()
    log.info("Simulator running:")

    window = QtWidgets.QApplication(sys.argv)
    app = Application_Qt(sim)
    log.info("Application Window running:")
    app.MainWindow.show()

    sim.simulation_parameters()
    sys.exit(window.exec_())
