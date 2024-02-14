# -*- coding: utf-8 -*-

"""
Cubic spline 


Author: Atsushi Sakai(@Atsushi_twi)

"""
import math
import numpy as np
import bisect
from scipy.spatial import KDTree


def dot_prod(vec_a, vec_b):
    return vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1]


def cross_prod(vec_a, vec_b):
    return vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0]

# 自然边界，端点的二阶导数为0. S''(x0) = S''(x_{nx-1}) = 0
class Spline:
    """
    Cubic Spline class
    """

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)
        # calc coefficient a
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        """
        Calc position

        if t is outside of the input x, return None

        """

        if t < self.x[0]:
            return self.calc(self.x[0])
        elif t > self.x[-1]:
            return self.calc(self.x[-1])

        i = self.__search_index(t)
        i -= 1 if i >= len(self.b) else 0
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t):
        """
        Calc first derivative

        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return self.calcd(self.x[0])
        elif t > self.x[-1]:
            return self.calcd(self.x[-1])

        i = self.__search_index(t)
        i -= 1 if i >= len(self.b) else 0
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        """
        Calc second derivative
        """

        if t < self.x[0]:
            return self.calcdd(self.x[0])
        elif t > self.x[-1]:
            return self.calcdd(self.x[-1])

        i = self.__search_index(t)
        i -= 1 if i >= len(self.b) else 0
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result
        
    def calcddd(self, t):
        """
        Calc tripple derivative
        """

        if t < self.x[0]:
            return self.calcddd(self.x[0])
        elif t > self.x[-1]:
            return self.calcddd(self.x[-1])

        i = self.__search_index(t)
        i -= 1 if i >= len(self.b) else 0
        result = 6.0 * self.d[i]
        return result

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        return B

# 固定边界条件，给定端点的一阶导数。S0'(x0) = A,S_{nx-2}'(x_{nx-1}) = B
class Clamped_Spline:
    def __init__(self, x, y, a0, bn_1):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient a
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h,a0,bn_1)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)


    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        for i in range(1,self.nx-1):
            A[i, i] = 2.0 * (h[i] + h[i - 1])
            A[i - 1, i] = h[i-1]
            A[i, i - 1] = h[i-1]

        A[0, 0] = 2*h[0]
        A[self.nx - 2, self.nx - 1] = h[-1]
        A[self.nx - 1, self.nx - 2] = h[-1]
        A[self.nx - 1, self.nx - 1] = 2*h[-1]
        # print(A)
        return A

    def __calc_B(self, h, a0, bn_1):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)

        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        B[0] = 3.0 * ((self.a[1] - self.a[0]) / \
                h[0] - a0)
        B[-1] =  3.0 * (bn_1 - (self.a[-1] - self.a[-2]) / \
                h[-1])
        return B


    def calc(self, t):
        """
        Calc position

        if t is outside of the input x, return None

        """

        if t < self.x[0]:
            return self.calc(self.x[0])
        elif t > self.x[-1]:
            return self.calc(self.x[-1])

        i = self.__search_index(t)
        i -= 1 if i >= len(self.b) else 0
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t):
        """
        Calc first derivative

        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return self.calcd(self.x[0])
        elif t > self.x[-1]:
            return self.calcd(self.x[-1])

        i = self.__search_index(t)
        i -= 1 if i >= len(self.b) else 0
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        """
        Calc second derivative
        """

        if t < self.x[0]:
            return self.calcdd(self.x[0])
        elif t > self.x[-1]:
            return self.calcdd(self.x[-1])

        i = self.__search_index(t)
        i -= 1 if i >= len(self.b) else 0
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result
    def calcddd(self, t):
        """
        Calc tripple derivative
        """

        if t < self.x[0]:
            return self.calcddd(self.x[0])
        elif t > self.x[-1]:
            return self.calcddd(self.x[-1])

        i = self.__search_index(t)
        i -= 1 if i >= len(self.b) else 0
        result = 6.0 * self.d[i]
        return result

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1


class Spline2D:
    """
    2D Cubic Spline class

    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

        # self.x = x
        # self.y = y

        
        self.sample_points = list()
        self.sample_ss = list()
        for s in np.arange(0.0, self.length(), 0.5):
            x, y = self.calc_position(s)
            self.sample_points.append((x, y))
            self.sample_ss.append(s)
        self.kd_tree = KDTree(self.sample_points)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def length(self):
        return float(self.s[-1]) if len(self.s) > 0 else 0.0

    def calc_position(self, s):
        """
        calc position
        """
        if s < 0:
            yaw = self.calc_yaw(0) + math.pi
            x, y = self.calc_position(0)
            x += abs(s) * math.cos(yaw)
            y += abs(s) * math.sin(yaw)
            return x, y
        elif s > self.length():
            yaw = self.calc_yaw(self.length())
            x, y = self.calc_position(self.length())
            x += abs(s - self.length()) * math.cos(yaw)
            y += abs(s - self.length()) * math.sin(yaw)
            return x, y

        x = self.sx.calc(s)
        y = self.sy.calc(s)
        return x, y

    def calc_curvature(self, s):
        """
        calc curvature
        """
        if s < 0:
            return self.calc_curvature(0)
        elif s > self.length():
            return self.calc_curvature(self.length())
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_dcurvature(self, s):
        """
        calc dcurvature/ds
        """
        if s < 0:
            return self.calc_dcurvature(0)
        elif s > self.length():
            return self.calc_dcurvature(self.length())

        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dddx = self.sx.calcddd(s)

        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        dddy = self.sy.calcddd(s)

        a = dx*ddy-dy*ddx
        b = dx*dddy-dy*dddx
        c = dx*ddx+dy*ddy
        d = dx*dx+dy*dy
        return (b*d-3.0*a*c)/(d*d*d)

    # 对于cubic spline来说约束了分段之间的一阶导和二阶导相等，所以曲率是连续的，为了求最大曲率需要将分段的曲率都求出来
    def calc_max_curvature(self):
        max_curvature = 0
        for s in self.s:
            max_curvature = max(max_curvature,abs(self.calc_curvature(s)))
        return max_curvature

    def calc_yaw(self, s):
        """
        calc yaw
        """
        if s < 0:
            yaw = self.calc_yaw(0)
            return yaw
        elif s > self.length():
            yaw = self.calc_yaw(self.length())
            return yaw
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)
        return yaw

    def xy_to_sd_by_kdtree(self, x, y):
        query_point = (x, y)
        dist, index = self.kd_tree.query(query_point)

        if index <= 0:
            start_point = self.calc_position(index)
            start_yaw = self.calc_yaw(0)
            vec_a = [x - start_point[0], y - start_point[1]]
            vec_b = [math.cos(start_yaw), math.sin(start_yaw)]
            d = cross_prod(vec_b, vec_a)
            s = dot_prod(vec_a, vec_b)
            return s, d
        elif index >= len(self.sample_ss) - 1:
            end_point = self.calc_position(self.length())
            end_yaw = self.calc_yaw(self.length())
            vec_a = [x - end_point[0], y - end_point[1]]
            vec_b = [math.cos(end_yaw), math.sin(end_yaw)]
            d = cross_prod(vec_b, vec_a)
            s = dot_prod(vec_a, vec_b) + self.length()
            return s, d

        def normalize(vec):
            vec_length = math.hypot(vec[0], vec[1])
            if vec_length >= 1e-3:
                vec = [vec[0] / vec_length, vec[1] / vec_length]
            else:
                vec = [0, 0]
            return vec
        last_point = self.sample_points[index - 1]
        next_point = self.sample_points[index + 1]
        point = self.sample_points[index]
        vec_a = [x - last_point[0], y - last_point[1]]
        vec_b = normalize([point[0] - last_point[0], point[1] - last_point[1]])
        last_d = cross_prod(vec_b, vec_a)
        last_s = dot_prod(vec_b, vec_a) + self.sample_ss[index - 1]
        vec_a = [x - point[0], y - point[1]]
        vec_b = normalize([next_point[0] - point[0], next_point[1] - point[1]])
        next_d = cross_prod(vec_b, vec_a)
        next_s = dot_prod(vec_b, vec_a) + self.sample_ss[index]
        if abs(next_d) <= abs(last_d):
            return next_s, next_d
        else:
            return last_s, last_d

    def xy_to_sd(self, x, y):
        min_dist = float('inf')
        min_s = -float('inf')
        for s in np.arange(0.0, self.length(), 0.01):
            ref_x, ref_y = self.calc_position(s)
            dist = math.hypot(ref_x - x, ref_y - y)
            if dist < min_dist:
                min_dist, min_s = dist, s
        ref_x, ref_y = self.calc_position(self.length())
        dist = math.hypot(ref_x - x, ref_y - y)
        if dist < min_dist:
            min_dist, min_s = dist, self.length()

        if min_s <= 1e-6:
            start_point = self.calc_position(0)
            start_yaw = self.calc_yaw(0)
            vec_a = [x - start_point[0], y - start_point[1]]
            vec_b = [math.cos(start_yaw), math.sin(start_yaw)]
            d = cross_prod(vec_b, vec_a)
            s = dot_prod(vec_a, vec_b)
            return s, d
        elif min_s >= self.length() - 1e-6:
            end_point = self.calc_position(self.length())
            end_yaw = self.calc_yaw(self.length())
            vec_a = [x - end_point[0], y - end_point[1]]
            vec_b = [math.cos(end_yaw), math.sin(end_yaw)]
            d = cross_prod(vec_b, vec_a)
            s = dot_prod(vec_a, vec_b) + self.length()
            return s, d
        yaw = self.calc_yaw(min_s)
        ref_x, ref_y = self.calc_position(min_s)
        cross_pd = math.cos(yaw) * (y - ref_x) - \
            math.sin(yaw) * (x - ref_y)
        d = abs(min_dist) if cross_pd > 0.0 else -abs(min_dist)
        return min_s, d

    def sd_to_xy(self, s, d):
        x_ref, y_ref = self.calc_position(s)
        yaw_ref = self.calc_yaw(s)
        x = x_ref + d * math.cos(yaw_ref + math.pi * 0.5)
        y = y_ref + d * math.sin(yaw_ref + math.pi * 0.5)
        return x, y


class Spline2D_Clamped:
    """
    2D Cubic Spline class

    """

    def __init__(self, x, y, yaw0, yawn_1):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        # self.sy = Spline(self.s, y)
        self.sy = Clamped_Spline(self.s, y, self.sx.calcd(0) * math.tan(yaw0), self.sx.calcd(self.s[-1]) * math.tan(yawn_1)) 

        self.x = x
        self.y = y
        
        self.sample_points = list()
        self.sample_ss = list()
        for s in np.arange(0.0, self.length(), 0.5):
            x, y = self.calc_position(s)
            self.sample_points.append((x, y))
            self.sample_ss.append(s)
        self.kd_tree = KDTree(self.sample_points)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def length(self):
        return float(self.s[-1]) if len(self.s) > 0 else 0.0

    def calc_position(self, s):
        """
        calc position
        """
        if s < 0:
            yaw = self.calc_yaw(0) + math.pi
            x, y = self.calc_position(0)
            x += abs(s) * math.cos(yaw)
            y += abs(s) * math.sin(yaw)
            return x, y
        elif s > self.length():
            yaw = self.calc_yaw(self.length())
            x, y = self.calc_position(self.length())
            x += abs(s - self.length()) * math.cos(yaw)
            y += abs(s - self.length()) * math.sin(yaw)
            return x, y
        x = self.sx.calc(s)
        y = self.sy.calc(s)
        return x, y

    def calc_curvature(self, s):
        """
        calc curvature
        """
        if s < 0:
            return self.calc_curvature(0)
        elif s > self.length():
            return self.calc_curvature(self.length())
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_dcurvature(self, s):
        """
        calc dcurvature/ds
        """
        if s < 0:
            return self.calc_dcurvature(0)
        elif s > self.length():
            return self.calc_dcurvature(self.length())

        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dddx = self.sx.calcddd(s)

        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        dddy = self.sy.calcddd(s)

        a = dx*ddy-dy*ddx
        b = dx*dddy-dy*dddx
        c = dx*ddx+dy*ddy
        d = dx*dx+dy*dy
        return (b*d-3.0*a*c)/(d*d*d)

    def calc_yaw(self, s):
        """
        calc yaw
        """
        if s < 0:
            yaw = self.calc_yaw(0)
            return yaw
        elif s > self.length():
            yaw = self.calc_yaw(self.length())
            return yaw
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)
        return yaw

    def xy_to_sd_by_kdtree(self, x, y):
        query_point = (x, y)
        dist, index = self.kd_tree.query(query_point)
        if index <= 0:
            start_point = self.calc_position(index)
            start_yaw = self.calc_yaw(0)
            vec_a = [x - start_point[0], y - start_point[1]]
            vec_b = [math.cos(start_yaw), math.sin(start_yaw)]
            d = cross_prod(vec_b, vec_a)
            s = dot_prod(vec_a, vec_b)
            return s, d
        elif index >= len(self.sample_ss) - 1:
            end_point = self.calc_position(self.length())
            end_yaw = self.calc_yaw(self.length())
            vec_a = [x - end_point[0], y - end_point[1]]
            vec_b = [math.cos(end_yaw), math.sin(end_yaw)]
            d = cross_prod(vec_b, vec_a)
            s = dot_prod(vec_a, vec_b) + self.length()
            return s, d

        def normalize(vec):
            vec_length = math.hypot(vec[0], vec[1])
            if vec_length >= 1e-3:
                vec = [vec[0] / vec_length, vec[1] / vec_length]
            else:
                vec = [0, 0]
            return vec
        last_point = self.sample_points[index - 1]
        next_point = self.sample_points[index + 1]
        point = self.sample_points[index]
        vec_a = [x - last_point[0], y - last_point[1]]
        vec_b = normalize([point[0] - last_point[0], point[1] - last_point[1]])
        last_d = cross_prod(vec_b, vec_a)
        last_s = dot_prod(vec_b, vec_a) + self.sample_ss[index - 1]
        vec_a = [x - point[0], y - point[1]]
        vec_b = normalize([next_point[0] - point[0], next_point[1] - point[1]])
        next_d = cross_prod(vec_b, vec_a)
        next_s = dot_prod(vec_b, vec_a) + self.sample_ss[index]
        if abs(next_d) <= abs(last_d):
            return next_s, next_d
        else:
            return last_s, last_d

    def xy_to_sd(self, x, y):
        min_dist = float('inf')
        min_s = -float('inf')
        for s in np.arange(0.0, self.length(), 0.01):
            ref_x, ref_y = self.calc_position(s)
            dist = math.hypot(ref_x - x, ref_y - y)
            if dist < min_dist:
                min_dist, min_s = dist, s
        ref_x, ref_y = self.calc_position(self.length())
        dist = math.hypot(ref_x - x, ref_y - y)
        if dist < min_dist:
            min_dist, min_s = dist, self.length()

        if min_s <= 1e-6:
            start_point = self.calc_position(0)
            start_yaw = self.calc_yaw(0)
            vec_a = [x - start_point[0], y - start_point[1]]
            vec_b = [math.cos(start_yaw), math.sin(start_yaw)]
            d = cross_prod(vec_b, vec_a)
            s = dot_prod(vec_a, vec_b)
            return s, d
        elif min_s >= self.length() - 1e-6:
            end_point = self.calc_position(self.length())
            end_yaw = self.calc_yaw(self.length())
            vec_a = [x - end_point[0], y - end_point[1]]
            vec_b = [math.cos(end_yaw), math.sin(end_yaw)]
            d = cross_prod(vec_b, vec_a)
            s = dot_prod(vec_a, vec_b) + self.length()
            return s, d
        yaw = self.calc_yaw(min_s)
        ref_x, ref_y = self.calc_position(min_s)
        cross_pd = math.cos(yaw) * (y - ref_x) - \
            math.sin(yaw) * (x - ref_y)
        d = abs(min_dist) if cross_pd > 0.0 else -abs(min_dist)
        return min_s, d

    def sd_to_xy(self, s, d):
        x_ref, y_ref = self.calc_position(s)
        yaw_ref = self.calc_yaw(s)
        x = x_ref + d * math.cos(yaw_ref + math.pi * 0.5)
        y = y_ref + d * math.sin(yaw_ref + math.pi * 0.5)
        return x, y


if __name__ == "__main__":
    xs = [0, 1, 2, 3]
    ys = [0, 1, 2, 3]
    test_curve = Spline2D(xs, ys)
    s, d = test_curve.xy_to_sd(0, 10)
    print(s, d)
    print(test_curve.xy_to_sd_by_kdtree(0, 10))
    s, d = test_curve.xy_to_sd(-10, 0)
    print(s, d)
    print(test_curve.xy_to_sd_by_kdtree(-10, 0))
    s, d = test_curve.xy_to_sd(0, -10)
    print(s, d)
    print(test_curve.xy_to_sd_by_kdtree(0, -10))
    s, d = test_curve.xy_to_sd(10, 0)
    print(s, d)
    print(test_curve.xy_to_sd_by_kdtree(10, 0))

    x, y = test_curve.sd_to_xy(-1, 0)
    print(x, y)
    x, y = test_curve.sd_to_xy(0, 2)
    print(x, y)
    x, y = test_curve.sd_to_xy(0, -2)
    print(x, y)
    x, y = test_curve.sd_to_xy(10, 0)
    print(x, y)

    print(test_curve.xy_to_sd(1, 1))
    print(test_curve.xy_to_sd_by_kdtree(1, 1))
