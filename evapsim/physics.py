
import math
from numba import cuda

import logging
log = logging.getLogger("evapsim")


@cuda.jit('''void(float64[:], float64[:], int8[:], float64, float64,
float64, int8)''')
def intersection_gpu(Px, Py, Vi, angle, cast,
                     epsilon, dec):
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
    raycast_sin = round(
            self.raycast_length * math.sin(angle), 10)
        raycast_cos = round(
            self.raycast_length * math.cos(angle), 10)
    """  # noqa

    i, j = cuda.grid(2)

    if i < Px.shape[0] and j < Py.shape[0] - 1:
        if i == j or i == j + 1:
            # Line intersecting itself
            pass

        else:
            Rx = round(math.sin(angle) * cast, dec)
            Ry = round(math.cos(angle) * cast, dec)
            Sx = round(Px[j + 1] - Px[j], dec)
            Sy = round(Py[j + 1] - Py[j], dec)

            QPx = round(Px[j] - Px[i], dec)
            QPy = round(Py[j] - Py[i], dec)
            PQx = round(Px[i] - Px[j], dec)
            PQy = round(Py[i] - Py[j], dec)

            RxS = round((Rx * Sy) - (Ry * Sx), dec)
            QPxR = round((QPx * Ry) - (QPy * Rx), dec)
            if (math.fabs(RxS) <= epsilon) and (math.fabs(QPxR) <= epsilon):
                # Colinear: Overlapping lines is considered intersecting
                qpr = round(QPx * Rx + QPy * Ry, dec)
                pqs = round(PQx * Sx + PQy * Sy, dec)
                RR = round(Rx * Rx + Ry * Ry, dec)
                SS = round(Sx * Sx + Sy * Sy, dec)
                if (0 <= qpr and qpr <= RR) or (0 <= pqs and pqs <= SS):
                    # The two lines are colinear and overlapping
                    Vi[i] = 1
                else:
                    # The two lines are colinear but disjoint
                    # Vi[i] = 0
                    pass
            elif (math.fabs(RxS) <= epsilon) \
                    and not (math.fabs(QPxR) <= epsilon):
                # Parallel and Non-Intersecting
                # Vi[i] = 0
                pass
            else:
                t = round((QPx * Sy - QPy * Sx) / RxS, dec)
                u = round((QPx * Ry - QPy * Rx) / RxS, dec)
                if not (math.fabs(RxS) <= epsilon)\
                   and (0.0 <= t and t <= 1.0) and (0.0 <= u and u <= 1.0):
                    # Intersection found = model_grid + t*r
                    Vi[i] = 1
                else:
                    # The lines are not parallel but do not intersect
                    # Vi[i] = 0
                    pass


@cuda.jit('''void(float64[:], float64[:], float64[:,:], float64[:,:], float64,
float64, int8)''')
def grid_gpu(Px, Py, Vx, Vy, model_resolution, epsilon, dec):
    '''
    TODO: Better implementation of a ragged list from numpy needed
    Currently the array is a defined size in y (mirrored of x). If it exceeds
    the x dimension this method will fail.

    The model contains a set of vertices i, each i then contains a set of
    points j that are spaced between i and i+1. The model is not square as
    the distance between each i may vary. So upon export, flatten the array.

    Take vertices from model and get grid points. Points are used in
    ray casting and intersection tests.

    In order to correctly place vertices along the lines I need to
    determine the direction of the line with respect to the model. Such
    that a line is not drawn backwards. In other words: If I wanted to draw
    a shape without lifting my pencil, what would be the order of points
    along the grid in order to achieve this.
    '''
    i, j = cuda.grid(2)

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
                        Vy[i, j] = round(Py[i] + (y * j), dec)
                    else:  # negative
                        Vx[i, j] = Px[i]
                        Vy[i, j] = round(Py[i] - (y * j), dec)
                elif math.fabs(Wy) <= epsilon:  # line is horizontal
                    if Wx > 0:  # positive
                        Vx[i, j] = round(Px[i] + (x * j), dec)
                        Vy[i, j] = Py[i]
                    else:  # negative
                        Vx[i, j] = round(Px[i] - (x * j), dec)
                        Vy[i, j] = Py[i]
                else:  # line is sloped
                    if Wx > 0 and Wy > 0:  # Slope in Q1
                        Vx[i, j] = round(Px[i] + (x * j), dec)
                        Vy[i, j] = round(Py[i] + (y * j), dec)
                    elif Wx < 0 and Wy > 0:  # Slope in Q2
                        Vx[i, j] = round(Px[i] - (x * j), dec)
                        Vy[i, j] = round(Py[i] + (y * j), dec)
                    elif Wx < 0 and Wy < 0:  # Slope in Q3
                        Vx[i, j] = round(Px[i] - (x * j), dec)
                        Vy[i, j] = round(Py[i] - (y * j), dec)
                    else:  # Slope in Q4
                        Vx[i, j] = round(Px[i] + (x * j), dec)
                        Vy[i, j] = round(Py[i] - (y * j), dec)


@cuda.jit('''void(float64[:], float64[:], int8[:], float64,
float64, float64, float64, float64,
float64[:,:], float64[:,:], float64[:,:],
boolean, boolean, boolean, float64, float64, int8)''')
def model_gpu(Px, Py, Pi, angle,
              Rx, Ry, Rz, rate,
              Vx, Vy, Vi,
              divet, peak, corner, epsilon, tArea, dec):
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

    Old 2D method for acute angle:
    numerator = (Rx * Sx) + (Ry * Sy)
            denominator = math.sqrt(Rx ** 2 + Ry ** 2) \
                * math.sqrt(Sx ** 2 + Sy ** 2)
    For 3D we must consider the normal to the plane and the raycast evap line
    Segment S then has a parallel line 1 unit away T such that we have a finite
    plane of width 1. The cross product yields our normal vector N.
    S cross T = N, since T is parallel to S in Z, T = (0, 0, 1)
    The resulting cross then = N = Sy xhat - Sx yhat such that
    N = (Sy, -Sx, 0)
    Thus the new dot product between the evap ray R and the Normal to the plane
    is cos theta =
    numerator = (Rx * Sy) - (Ry * Sx)
    denominator = math.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2) \
        * math.sqrt(Sy ** 2 + Sx ** 2)

    Has 2 zero checks.
        1: area of triangle to determine if a line is straight
        2: intersection tests.
    '''
    i = cuda.grid(1)

    if i == 0 or i == Px.shape[0] - 1:
        # Pin start of model | Pin end of model
        Vx[i, 0] = Px[i]
        Vy[i, 0] = Py[i]
        Vi[i, 0] = 1

    elif Pi[i] != 0:
        # Point in shadow
        if Pi[i - 1] == 0 and Pi[i + 1] == 0 and divet:
            # A divet with material landing around it. Likely none physical
            # average it out with nearest neighbors
            Sx = round((Px[i + 1] + Px[i - 1]) / 2.0, dec)
            Sy = round((Py[i + 1] + Py[i - 1]) / 2.0, dec)
            Vx[i, 0] = Sx
            Vy[i, 0] = Sy
            Vi[i, 0] = 0
        # Add shaded point. No change.
        else:
            Vx[i, 0] = Px[i]
            Vy[i, 0] = Py[i]
            Vi[i, 0] = 1

    elif i < Px.shape[0] - 1 and i > 1:
        # Point is now assumed to be in evaporant
        # Determine from a set of 3 points A:i-1, B:i, C:i+1 if it is straight
        # value should be zero (smaller than epsilon). Is area of triangle
        area: 'area of triangle' = (Px[i - 1] * (Py[i] - Py[i + 1])
                                    + Px[i] * (Py[i + 1] - Py[i - 1])
                                    + Px[i + 1] * (Py[i - 1] - Py[i]))

        if math.fabs(area) <= tArea:
            # Line segment is straight
            Sx = Px[i + 1] - Px[i - 1]
            Sy = Py[i + 1] - Py[i - 1]
            n = (Rx * Sy) - (Ry * Sx)
            d = math.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2) \
                * math.sqrt(Sy ** 2 + Sx ** 2)
            θ = math.acos(max(-1.0, min(1.0, (n / d))))
            t = math.fabs(math.cos(θ) * rate)
            Ax = Px[i] + t * math.sin(angle)
            Ay = Py[i] + t * math.cos(angle)
            Vx[i, 0] = round(Ax, dec)
            Vy[i, 0] = round(Ay, dec)
            Vi[i, 0] = 0

        else:
            # A corner: check how evaporation is landing around it
            if Pi[i - 1] != 0 and Pi[i + 1] != 0:
                # A peak that has evaporation on it but not
                # its nearest neighbors. Considered to be non-real and is
                # averaged with its nearest neighbors
                if peak:
                    Sx = round((Px[i + 1] + Px[i - 1]) / 2.0, dec)
                    Sy = round((Py[i + 1] + Py[i - 1]) / 2.0, dec)
                    Vx[i, 0] = Sx
                    Vy[i, 0] = Sy
                    Vi[i, 0] = 1
                else:  # how to treat this? for now drop point.
                    pass

            elif Pi[i - 1] == 0 and Pi[i + 1] == 0:
                # Corner in evap with either side having different evap t
                # A : Left point of vertice
                Sx = Px[i] - Px[i - 1]
                Sy = Py[i] - Py[i - 1]
                n = (Rx * Sy) - (Ry * Sx)
                d = math.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2) \
                    * math.sqrt(Sy ** 2 + Sx ** 2)
                θ = math.acos(max(-1.0, min(1.0, (n / d))))
                t1 = math.fabs(math.cos(θ) * rate)
                p0_x = Px[i - 1] + t1 * math.sin(angle)
                p0_y = Py[i - 1] + t1 * math.cos(angle)
                p1_x = Px[i] + t1 * math.sin(angle)
                p1_y = Py[i] + t1 * math.cos(angle)

                # B : Right point of vertice
                Sx = Px[i + 1] - Px[i]
                Sy = Py[i + 1] - Py[i]
                n = (Rx * Sy) - (Ry * Sx)
                d = math.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2) \
                    * math.sqrt(Sy ** 2 + Sx ** 2)
                θ = math.acos(max(-1.0, min(1.0, (n / d))))
                t2 = math.fabs(math.cos(θ) * rate)
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
                    s = round(s, dec)
                    t = round(t, dec)
                    # If either segment has a valid intersection point.
                    # Use that as the growth point
                    if t >= 0 and t <= 1.0:
                        i_x = p0_x + (t * Rx)
                        i_y = p0_y + (t * Ry)
                        Vx[i, 0] = round(i_x, dec)
                        Vy[i, 0] = round(i_y, dec)
                        Vi[i, 0] = 0
                        # Vi[i, 1] = t
                    elif s >= 0 and s <= 1.0:
                        i_x = p2_x + (s * Sx)
                        i_y = p2_y + (s * Sy)
                        Vx[i, 0] = round(i_x, dec)
                        Vy[i, 0] = round(i_y, dec)
                        Vi[i, 0] = 0
                        # Vi[i, 1] = t
                    else:
                        # The intersection is not within reasonable bounds at
                        # the corner. Potential that corner is too flat or
                        # too sharp.

                        # Force a line segment from the nearest neighbors
                        # Then add material w.r.t. the forced line segment
                        Sx = Px[i + 1] - Px[i - 1]
                        Sy = Py[i + 1] - Py[i - 1]
                        n = (Rx * Sy) - (Ry * Sx)
                        d = math.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2) \
                            * math.sqrt(Sy ** 2 + Sx ** 2)
                        θ = math.acos(max(-1.0, min(1.0, (n / d))))
                        t = math.fabs(math.cos(θ) * rate)
                        Ax = Px[i] + t * math.sin(angle)
                        Ay = Py[i] + t * math.cos(angle)
                        Vx[i, 0] = round(Ax, dec)
                        Vy[i, 0] = round(Ay, dec)
                        Vi[i, 0] = 0

                else:  # Avoid divide by zeros
                    # Force a line segment from the nearest neighbors
                    # Then add material w.r.t. the forced line segment
                    Sx = Px[i + 1] - Px[i - 1]
                    Sy = Py[i + 1] - Py[i - 1]
                    n = (Rx * Sy) - (Ry * Sx)
                    d = math.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2) \
                        * math.sqrt(Sy ** 2 + Sx ** 2)
                    θ = math.acos(max(-1.0, min(1.0, (n / d))))
                    t = math.fabs(math.cos(θ) * rate)
                    Ax = Px[i] + t * math.sin(angle)
                    Ay = Py[i] + t * math.cos(angle)
                    Vx[i, 0] = round(Ax, dec)
                    Vy[i, 0] = round(Ay, dec)
                    Vi[i, 0] = 0

            elif Pi[i - 1] != 0 and Pi[i + 1] == 0:
                # Shaded corner evap | Preserve the i point
                Sx = Px[i + 1] - Px[i]
                Sy = Py[i + 1] - Py[i]
                n = (Rx * Sy) - (Ry * Sx)
                d = math.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2) \
                    * math.sqrt(Sy ** 2 + Sx ** 2)
                θ = math.acos(max(-1.0, min(1.0, (n / d))))
                t = math.fabs(math.cos(θ) * rate)
                Ax = Px[i] + t * math.sin(angle)
                Ay = Py[i] + t * math.cos(angle)
                if corner:
                    Vx[i, 0] = Px[i]
                    Vy[i, 0] = Py[i]
                    Vi[i, 0] = 1
                    Vx[i, 1] = round(Ax, dec)
                    Vy[i, 1] = round(Ay, dec)
                    Vi[i, 1] = 0
                else:
                    Vx[i, 0] = round(Ax, dec)
                    Vy[i, 0] = round(Ay, dec)
                    Vi[i, 0] = 0

            elif Pi[i + 1] != 0 and Pi[i - 1] == 0:
                # Shaded corner evap | Preserve the i point
                Sx = Px[i] - Px[i - 1]
                Sy = Py[i] - Py[i - 1]
                n = (Rx * Sy) - (Ry * Sx)
                d = math.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2) \
                    * math.sqrt(Sy ** 2 + Sx ** 2)
                θ = math.acos(max(-1.0, min(1.0, (n / d))))
                t = math.fabs(math.cos(θ) * rate)
                Ax = Px[i] + t * math.sin(angle)
                Ay = Py[i] + t * math.cos(angle)
                if corner:
                    Vx[i, 0] = round(Ax, dec)
                    Vy[i, 0] = round(Ay, dec)
                    Vi[i, 0] = 0
                    Vx[i, 1] = Px[i]
                    Vy[i, 1] = Py[i]
                    Vi[i, 1] = 1
                else:
                    Vx[i, 0] = round(Ax, dec)
                    Vy[i, 0] = round(Ay, dec)
                    Vi[i, 0] = 0


@cuda.jit('''void(float64[:], float64[:], float64[:],
float64[:], float64[:], float64, float64, int8)''')
def merge_gpu(Px, Py, Pi,
              Vx, Vy, gridspace, epsilon, dec):
    """
    Merge points together that fall within the defined gridspace. Currently
    implemented as a simple merge function. My fancier versions produced some
    interesting artifacts.

    If two vertices fall within the model's resolution merge them by averaging
    their values. This leads to 'rounded' corners but greatly helps reduce
    impossible shapes from forming.

    This is happening right after new material is added, thus we can lock
    points that didn't get new material to preserve profile at corners.
    """
    i = cuda.grid(1)

    if i == 0 or i == Px.shape[0] - 1:
        # Pin start of model | Pin end of model
        Vx[i] = Px[i]
        Vy[i] = Py[i]

    else:
        Wx = Px[i + 1] - Px[i - 1]
        Wy = Py[i + 1] - Py[i - 1]
        W = math.sqrt(Wx ** 2 + Wy ** 2)
        grid_points = math.floor(W / (gridspace * 2))
        if grid_points != 0:
            # Append vertice, points are not smaller than gridspace
            Vx[i] = Px[i]
            Vy[i] = Py[i]
        else:
            Ux = round((Px[i + 1] + Px[i - 1]) / 2.0, dec)
            Uy = round((Py[i + 1] + Py[i - 1]) / 2.0, dec)
            if Pi[i] != 0:
                # Pin shaded point
                # Vx[i] = Px[i]
                # Vy[i] = Py[i]
                # Testing full merge again
                Vx[i] = Ux
                Vy[i] = Uy
            else:
                Vx[i] = Ux
                Vy[i] = Uy
