# -*- coding: utf-8 -*-

import math
import numpy as np
import bisect
from scipy.spatial import KDTree
import scipy.linalg as spla

import matplotlib.pyplot as plt



def calc_2d_spline_interpolation(sp,xycanv, num=100):
    """
    Calc 2d spline course with interpolation
    :param x: interpolated x positions
    :param y: interpolated y positions
    :param num: number of path points
    :return:
        - x     : x positions
        - y     : y positions
        - yaw   : yaw angle list
        - k     : curvature list
        - s     : Path length from start point
    """
    u = np.linspace(sp.sx.u[0], sp.sx.u[-1], num+1)[:-1]
    print(u)
    u = np.append(u,sp.sx.u[-1])
    r_x, r_y, r_yaw, r_k = [], [], [], []
    r_dk = []
    r_dyaw = []

    dx = []
    ddx = []
    dddx = []
    dy = []
    ddy = []
    dddy = []
    v = []
    a = []
    j = []
    for i_u in u:
        ix, iy = sp.calc_position_u(i_u)
        r_x.append(ix)
        r_y.append(iy)
        r_yaw.append(sp.calc_yaw_u(i_u))
        # r_k.append(sp.calc_curvature_u(i_u))
        # r_dk.append(sp.calc_dcurvature_u(i_u))

        d = sp.calcd(i_u)
        dd = sp.calcdd(i_u)
        ddd = sp.calcddd(i_u)
        dx.append(d[0])
        dy.append(d[1])
        v.append(math.hypot(d[0],d[1]))
        ddx.append(dd[0])
        ddy.append(dd[1])
        a.append(math.hypot(dd[0],dd[1]))
        dddx.append(ddd[0])
        dddy.append(ddd[1])
        j.append(math.hypot(ddd[0],ddd[1]))


    travel = np.cumsum([np.hypot(dx, dy) for dx, dy in zip(np.diff(r_x), np.diff(r_y))]).tolist()
    travel = np.concatenate([[0.0], travel])

    xycanv.plot(sp.sx.ctrl_points,sp.sy.ctrl_points ,"b", label = "",lw=2)#  "-r"
    xycanv.plot(r_x, r_y,"g", label = "b样条轨迹",lw=2)#  "-r"
    xycanv.grid(True)
    xycanv.axis("equal")
    xycanv.set_xlabel('x[m]')
    xycanv.set_ylabel('y[m]')
    # ax.set_title('episode')
    xycanv.legend()

    # plt.subplots()
    # # plt.plot(u*sp.sx.dt*sp.sx.m, r_yaw, "g", label = "",lw=2)
    # plt.plot(u, r_yaw, "g", label = "",lw=2)

    # plt.xlabel("t")
    # plt.ylabel("yaw[rad]")

    # # plt.subplots()
    # # # plt.plot(u*sp.sx.dt*sp.sx.m, r_k, "g", label = "",lw=2)
    # # plt.plot(u, r_k, "g", label = "",lw=2)

    # # plt.xlabel("t")
    # # plt.ylabel("kappa")

    # # plt.subplots()
    # # # plt.plot(u*sp.sx.dt*sp.sx.m, r_dk, "g", label = "",lw=2)
    # # plt.plot(u, r_dk, "g", label = "",lw=2)
    # # plt.xlabel("t")
    # # plt.ylabel("dkappa")

    # plt.subplots()
    # # plt.plot(u*sp.sx.dt*sp.sx.m, r_yaw, "g", label = "",lw=2)
    # plt.plot(u,dx, "g", label = "",lw=2)
    # plt.ylabel("vx")
    # plt.xlabel("t")

    # plt.subplots()
    # plt.plot(u,dy, "g", label = "",lw=2)
    # plt.ylabel("vy")
    # plt.xlabel("t")

    # plt.subplots()
    # # plt.plot(u*sp.sx.dt*sp.sx.m, r_yaw, "g", label = "",lw=2)
    # plt.plot(u,ddx, "g", label = "",lw=2)
    # plt.ylabel("ax")
    # plt.xlabel("t")

    # plt.subplots()
    # plt.plot(u,ddy, "g", label = "",lw=2)
    # plt.ylabel("ay")
    # plt.xlabel("t")

    # plt.subplots()
    # plt.plot(u, dddy, "g", label = "",lw=2)
    # plt.xlabel("t")
    # plt.ylabel("jy")

    # plt.subplots()
    # plt.plot(u, dddx, "g", label = "",lw=2)
    # plt.xlabel("t")
    # plt.ylabel("jx")

    # plt.subplots()
    # plt.plot(u, v, "g", label = "",lw=2)
    # plt.xlabel("t")
    # plt.ylabel("v")

    # plt.subplots()
    # plt.plot(u, a, "g", label = "",lw=2)
    # plt.xlabel("t")
    # plt.ylabel("a")

    # plt.subplots()
    # plt.plot(u, j, "g", label = "",lw=2)
    # plt.xlabel("t")
    # plt.ylabel("j")
    # plt.show()
    return r_x, r_y, r_yaw, r_k, r_dk , travel

def calc_2d_picewise_spline_interpolation(sp,xycanv, num=100):
    """
    Calc 2d spline course with interpolation
    :param x: interpolated x positions
    :param y: interpolated y positions
    :param num: number of path points
    :return:
        - x     : x positions
        - y     : y positions
        - yaw   : yaw angle list
        - k     : curvature list
        - s     : Path length from start point
    """
    u = np.linspace(0., sp.start_time[-1], num+1)[:-1]
    u = np.append(u,sp.start_time[-1])
    r_x, r_y, r_yaw, r_k = [], [], [], []
    r_dk = []
    r_dyaw = []
    dx = []
    ddx = []
    dddx = []
    dy = []
    ddy = []
    dddy = []
    for i_u in u:
        ix, iy = sp.calc_position(i_u)
        r_x.append(ix)
        r_y.append(iy)
        r_yaw.append(sp.calc_yaw(i_u))
        d = sp.calc_v(i_u)
        dd = sp.calc_a(i_u)
        ddd = sp.calc_j(i_u)
        dx.append(d[0])
        dy.append(d[1])
        ddx.append(dd[0])
        ddy.append(dd[1])
        dddx.append(ddd[0])
        dddy.append(ddd[1])
        # r_k.append(sp.calc_curvature(i_u))
        # r_dk.append(sp.calc_dcurvature(i_u))
    

    travel = np.cumsum([np.hypot(dx, dy) for dx, dy in zip(np.diff(r_x), np.diff(r_y))]).tolist()
    travel = np.concatenate([[0.0], travel])

    for i in range(len(sp.splines)):
        xycanv.plot(sp.splines[i].sx.ctrl_points,sp.splines[i].sy.ctrl_points ,"b", label = "",lw=2)#  "-r"
    xycanv.plot(r_x, r_y,"g", label = "",lw=2)#  "-r"
    xycanv.grid(True)
    xycanv.axis("equal")
    xycanv.set_xlabel('x[m]')
    xycanv.set_ylabel('y[m]')
    # ax.set_title('episode')
    xycanv.legend()

    plt.subplots()
    # plt.plot(u*sp.sx.dt*sp.sx.m, r_yaw, "g", label = "",lw=2)
    plt.plot(u, r_yaw, "g", label = "",lw=2)

    plt.xlabel("t")
    plt.ylabel("yaw[rad]")

    # plt.subplots()
    # # plt.plot(u*sp.sx.dt*sp.sx.m, r_k, "g", label = "",lw=2)
    # plt.plot(u, r_k, "g", label = "",lw=2)

    # plt.xlabel("t")
    # plt.ylabel("kappa")

    # plt.subplots()
    # # plt.plot(u*sp.sx.dt*sp.sx.m, r_dk, "g", label = "",lw=2)
    # plt.plot(u, r_dk, "g", label = "",lw=2)
    # plt.xlabel("t")
    # plt.ylabel("dkappa")

    plt.subplots()
    # plt.plot(u*sp.sx.dt*sp.sx.m, r_yaw, "g", label = "",lw=2)
    plt.plot(dx, dy, "g", label = "",lw=2)

    plt.xlabel("vx")
    plt.ylabel("vy")

    plt.subplots()
    # plt.plot(u*sp.sx.dt*sp.sx.m, r_yaw, "g", label = "",lw=2)
    plt.plot(ddx, ddy, "g", label = "",lw=2)

    plt.xlabel("ax")
    plt.ylabel("ay")

    plt.subplots()
    # plt.plot(u*sp.sx.dt*sp.sx.m, r_yaw, "g", label = "",lw=2)
    plt.plot(dddx, dddy, "g", label = "",lw=2)

    plt.xlabel("jx")
    plt.ylabel("jy")

    plt.show()

    return r_x, r_y, r_yaw, r_k, r_dk , travel

def calc_2dsplit_spline_interpolation(spx,spy, num=100):
    u = np.linspace(spx.u[0], spx.u[-1], num+1)[:-1]
    u = np.append(u,sp.sx.u[-1])
    cuxs = []
    cuys = []
    dcuxs = []
    dcuys = []
    ddcuxs = []
    ddcuys = []
    dddcuxs = []
    dddcuys = []

    for i_u in u:
        cux = spx.calc(i_u)
        cuxs.append(cux)
        cuy = spy.calc(i_u)
        cuys.append(cuy)

        dcux = spx.calcd(i_u)
        dcuxs.append(dcux)
        dcuy = spy.calcd(i_u)
        dcuys.append(dcuy)

        ddcux = spx.calcdd(i_u)
        ddcuxs.append(ddcux)
        ddcuy = spy.calcdd(i_u)
        ddcuys.append(ddcuy)

        dddcux = spx.calcddd(i_u)
        dddcuxs.append(dddcux)
        dddcuy = spy.calcddd(i_u)
        dddcuys.append(dddcuy)


    plt.figure()
    plt.plot(cuxs, cuys,"g", label = "",lw=2)#  "-r"
    plt.plot(spx.ctrl_points, spx.ctrl_points,"b", label = "",lw=2)#  "-r"
    xycanv.grid(True)
    xycanv.axis("equal")
    xycanv.set_xlabel('x[m]')
    xycanv.set_ylabel('y[m]')

    plt.subplots()
    plt.plot(dcuxs, dcuys, "g", label = "d",lw=2)
    plt.plot(spx.v_bspline.ctrl_points, spx.v_bspline.ctrl_points,"b", label = "",lw=2)#  "-r"
    plt.xlabel("t")
    plt.ylabel("v")
    plt.legend()

    plt.subplots()
    plt.plot(ddcuxs, ddcuys, "g", label = "dd",lw=2)
    plt.plot(spx.a_bspline.ctrl_points, spx.a_bspline.ctrl_points,"b", label = "",lw=2)#  "-r"


    plt.subplots()
    plt.plot(ddcuxs, dddcuys, "g", label = "ddd",lw=2)
    plt.plot(spx.j_bspline.ctrl_points, spx.j_bspline.ctrl_points,"b", label = "",lw=2)#  "-r"

    plt.show()


# 1 一维画图改

def calc_1d_spline_interpolation(sp, num=100):
    u = np.linspace(sp.u[0], sp.u[-1], num+1)[:-1]
    u = np.append(u,sp.u[-1])

    cuxs = []
    dcuxs = []
    dcuys = []
    ddcuxs = []
    dddcuxs = []

    for i_u in u:
        cux = sp.calc(i_u)
        cuxs.append(cux)

        dcux = sp.calcd(i_u)
        dcuxs.append(dcux)

        ddcux = sp.calcdd(i_u)
        ddcuxs.append(ddcux)

        dddcux = sp.calcddd(i_u)
        dddcuxs.append(dddcux)


    plt.subplots()
    plt.plot(u, cuxs, "g", label = "",lw=2)


    ctrl_u = []
    for i in range(len(sp.ctrl_points)):
        ctrl_u.append(float(i)*sp.T/(len(sp.ctrl_points)-1))
    print(sp.u[sp.p:sp.m-sp.p+1])
    print(sp.ctrl_points)


    plt.plot(ctrl_u, sp.ctrl_points,"b", label = "",lw=2)#  "-r"
    # plt.plot(sp.u[sp.p:sp.m-sp.p], sp.ctrl_points,"b", label = "",lw=2)#  "-r"

    plt.xlabel("t")
    plt.ylabel("x")
    plt.legend()

    plt.subplots()
    plt.plot(u, dcuxs, "g", label = "",lw=2)
    ctrl_u = []
    for i in range(len(sp.v_bspline.ctrl_points)):
        ctrl_u.append(float(i)*sp.T/(len(sp.v_bspline.ctrl_points)-1))

    plt.plot(ctrl_u, sp.v_bspline.ctrl_points,"b", label = "",lw=2)#  "-r"
    plt.xlabel("t")
    plt.ylabel("dx")
    plt.legend()

    plt.subplots()
    plt.plot(u, ddcuxs, "g", label = "",lw=2)
    ctrl_u = []
    for i in range(len(sp.a_bspline.ctrl_points)):
        ctrl_u.append(float(i)*sp.T/(len(sp.a_bspline.ctrl_points)-1))

    plt.plot(ctrl_u, sp.a_bspline.ctrl_points,"b", label = "",lw=2)#  "-r"
    plt.xlabel("t")
    plt.ylabel("ddx")
    plt.legend()

    plt.subplots()
    plt.plot(u, dddcuxs, "g", label = "",lw=2)
    ctrl_u = []
    for i in range(len(sp.j_bspline.ctrl_points)):
        ctrl_u.append(float(i)*sp.T/(len(sp.j_bspline.ctrl_points)-1))

    plt.plot(ctrl_u, sp.j_bspline.ctrl_points,"b", label = "",lw=2)#  "-r"
    plt.xlabel("t")
    plt.ylabel("dddx")
    plt.legend()

    plt.show()


def calc_1d_cubic_spline_interpolation(sp, num=100):
    u = np.linspace(sp.u[0], sp.u[-1], num+1)[:-1]
    u = np.append(u,sp.u[-1])

    cuxs = []
    dcuxs = []
    dcuys = []
    ddcuxs = []
    dddcuxs = []

    for i_u in u:
        cux = sp.calc(i_u)
        cuxs.append(cux)

        # dcux = sp.calcd(i_u)
        # dcuxs.append(dcux)

        # ddcux = sp.calcdd(i_u)
        # ddcuxs.append(ddcux)

        # dddcux = sp.calcddd(i_u)
        # dddcuxs.append(dddcux)



    plt.subplots()
    plt.plot(u, cuxs, "g", label = "",lw=2)


    ctrl_u = []
    for i in range(len(sp.ctrl_points[2:-2])):
        ctrl_u.append(float(i)*sp.T/(len(sp.ctrl_points[2:-2])-1))

    plt.plot(ctrl_u, sp.ctrl_points[2:sp.m-1],"b", label = "",lw=2)#  "-r"

    # for i in range(len(sp.ctrl_points)):
    #     ctrl_u.append(float(i)*sp.T/(len(sp.ctrl_points)-1))

    # plt.plot(ctrl_u, sp.ctrl_points,"b", label = "",lw=2)#  "-r"

    plt.xlabel("t")
    plt.ylabel("x")
    plt.legend()
    plt.show()



def calc_2d_cubic_spline_interpolation(sp, num=100):

    u = np.linspace(sp.sx.u[0], sp.sx.u[-1], num+1)[:-1]
    u = np.append(u,sp.sx.u[-1])
    r_x, r_y, r_yaw, r_k = [], [], [], []
    r_dk = []
    r_dyaw = []
    for i_u in u:
        ix, iy = sp.calc_position_u(i_u)
        r_x.append(ix)
        r_y.append(iy)



    plt.subplots()

    plt.plot(r_x, r_y,"g", label = "",lw=2)#  "-r"
    plt.plot(sp.sx.ctrl_points,sp.sy.ctrl_points ,"b", label = "",lw=2)#  "-r"
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    # ax.set_title('episode')
    plt.legend()


    plt.show()

    return r_x, r_y, r_yaw, r_k, r_dk 





def U_quasi_uniform(n,p,T): 
    """准均匀B样条的节点向量计算
    首末值定义为 0 和 1
    Args:
        n (_type_, optional): n表示控制点个数-1，控制点共n+1个. Defaults to None.
        p (_type_, optional): B样条阶数p次曲线. Defaults to None.

    Returns:
        _type_: _description_
    """
    # 准均匀B样条的节点向量计算，共n+1个控制顶点，k-1次B样条，k阶
    NodeVector = [0. for i in range(n+p+1+1)] # m = n+p+1段 n+p+2个端点
    piecewise = n - p + 1  # B样条曲线的段数[up,un+1) = 共n+1-p+1个端点 n+1-p段:控制点个数-次数
    
    if piecewise == 1:  # 只有一段曲线时，n = p-1
        for i in range(1,p+1 +1):
            NodeVector[n+i] = float(T)  # 末尾重复度p
    else:
        for i in range(n-p):  # 中间段内节点均匀分布：两端共2p个节点，中间还剩(n+p+1+1-2p=n-p+1+1）个节点
            NodeVector[p+i+1] = NodeVector[p+i]+ float(T)/piecewise # [p+1,n]
        for i in range(1,p+1 +1):
            NodeVector[n+i] = float(T)  # 末尾重复度p

        # NodeVector[n + 1:n + p + 2] = float(T)  # 末尾重复度p
    return NodeVector

# https://blog.csdn.net/qq_35635374/article/details/121926398
def points2uniform_bspline_ctrlp(dt,start_d12,end_d12,points):
    #points按执行顺序排列
    k = len(points)
    # 方程数大于变量数 超定方程组
    # A = np.zeros((k+4, k+2))
    # b = np.zeros(k+4)
    A = np.zeros((k+2, k+2))
    b = np.zeros(k+2)
    for i in range(k):
        b[i] = points[i]
        A[i,i] = 1./6.
        A[i,i+1] = 2./3.
        A[i,i+2] = 1./6.
    b[k] = start_d12[0] 
    b[k+1] = end_d12[0] 
    # b[k+2] = start_d12[1] 
    # b[k+3] = end_d12[1] 

    A[k,0] = -1./2./dt
    A[k,1] = 0
    A[k,2] = 1./2./dt

    A[k+1,-3] = -1./2./dt
    A[k+1,-2] = 0
    A[k+1,-1] = 1./2./dt


    # A[k+2,0] = 1/dt/dt
    # A[k+2,1] = -2/dt/dt
    # A[k+2,2] = 1/dt/dt

    # A[k+3,-3] = 1/dt/dt
    # A[k+3,-2] = -2/dt/dt
    # A[k+3,-1] = 1/dt/dt


    print(A)
    print(b)
    print(points)
    # linalg解不了非方，可以揍0成方但是会出现奇异阵 无解或无限解
    # https://andreask.cs.illinois.edu/cs357-s15/public/demos/06-qr-applications/Solving%20Least-Squares%20Problems.html
    # 推荐通过QR分解后再求解
    Q, R = np.linalg.qr(A, mode="complete")
    x = spla.solve_triangular(R[:k+2], Q.T[:k+2].dot(b), lower=False)

    # Q, R = np.linalg.qr(A)
    # x = spla.solve_triangular(R, Q.T.dot(b), lower=False)

    x = np.linalg.solve(A,b)
    # 方程组必存在最小二乘解，且a是方程组的最小二乘解的充要条件是a是RT·R·a=RT·y的解。
    # x = np.linalg.lstsq(A, b)[0]
    # print(x)
    # x = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
    return x



# 凸优化不需要给出初解，约束初终态，约束其他控制点在两个box的交集中，T大概算一个匀速的分段时间
# 决策网络输入压缩点云当前状态输出终态的位置和速度，控制网络输入压缩点云以及 轨迹上当前状态投影点附近区域的跟踪点状态 当前状态输出dvx,dvy,dw,
# 对于控制网络最好是将整条轨迹输入，让网络自己决定后续跟踪哪个点，但是轨迹总点数或者说控制点数都不是固定的

# m = n+p+1 , 除去前后重复控制点p个0,1，计入一个0,1[up,un+1]，区间共n+1-p段,n+1为控制点数，p为次数
# 所以 要求 控制点数 >= p，总共至少p个控制点，即至少是2个box


class quniform_clamped_bspline(object):
    def __init__(self,ctrl_points,degrees,T):
        self.ctrl_points = ctrl_points
        self.p = degrees
        self.n = len(ctrl_points) - 1
        self.m = self.n + self.p + 1 # 区间段数
        self.T = float(T)
        self.u = U_quasi_uniform(self.n,self.p,self.T)
        
        # self.u = [0. for i in range(self.m+1)]
        
        # for i in range(len(self.u)):
        #     if i <= self.p:
        #         self.u[i] = (i - self.p)*dt
        #     else:
        #         self.u[i] = self.u[i-1] + self.dt

        dt = self.T/self.m
        # for i in range(len(self.u)):
        #     if i <= self.p:
        #         self.u[i] = (i - self.p)*dt
        #     elif i > self.p and i <= self.m - self.p:
        #         self.u[i] = self.u[i-1] + dt
        #     elif i > self.m - self.p :
        #         self.u[i] = self.u[i-1] + dt
        # for i in range(len(self.u)):
        #     if i <= self.p:
        #         self.u[i] = (i-self.p)*dt
        #     else:
        #         self.u[i] = self.u[i-1] + dt
            # self.u[i] = i*dt

        # print("T:",self.u)
        # for i in range(len(self.u)):
        #     if i <= self.p:
        #         self.u[i] = 0.
        #     elif i > self.p and i <= self.m - self.p:
        #         self.u[i] = self.u[i-1] + 1./(self.m - 2*self.p)*self.T
        #     elif i > self.m - self.p :
        #         self.u[i] = self.T
        # print(self.u)
        self.v_ctrl_points = []
        self.a_ctrl_points = []
        self.j_ctrl_points = []
        self.v_bspline = None
        self.a_bspline = None
        self.j_bspline = None

        # self.calc_derivative_spline()
    # 计算u落在哪个分段上
    def search_index(self, u):
        """
        search data segment index
        """
        # 最后一个<u
        # j = bisect.bisect_left(self.u, u) # 第一个>= lower_bound
        # bisect.bisect_right #第一个>   upper_bound
        # j -= 1 if j >= len(self.u) else 0

        # j = bisect.bisect_left(self.u, u) - 1 # 最后一个< ,lo = self.p
        j = bisect.bisect_right(self.u, u,lo = self.p,hi = self.m - self.p) - 1 # 最后一个<= ,lo = self.p

        j -= 1 if j >= len(self.u) else 0

        return j
    
    def get_v_spline(self):
        # The derivative of a b-spline is also a b-spline, its order become p_-1
        # control point Qi = p_*(Pi+1-Pi)/(ui+p_+1-ui+1)
        if self.p-1 < 0:
            return

        for i in range(len(self.ctrl_points)-1):
            self.v_ctrl_points.append(self.p*(self.ctrl_points[i+1] - self.ctrl_points[i])/\
            (self.u[i+self.p+1] - self.u[i+1]))

        self.v_bspline = quniform_clamped_bspline(self.v_ctrl_points,self.p-1,self.T)

    def get_a_spline(self):
        if self.v_bspline is None:
            self.get_v_spline()

        self.v_bspline.get_v_spline()
        self.a_ctrl_points = self.v_bspline.v_ctrl_points
        self.a_bspline = self.v_bspline.v_bspline
    
    def get_j_spline(self):
        if self.a_bspline is None:
            self.get_a_spline()

        self.a_bspline.get_v_spline()
        self.j_ctrl_points = self.a_bspline.v_ctrl_points
        self.j_bspline = self.a_bspline.v_bspline

    def calc(self,u):# deBoor 迭代形式  u时间[0,T]
    # http://www.whudj.cn/?p=535
    # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/de-Boor.html
        # u = t/self.dt
        ub = min(max(self.u[self.p],u),self.u[self.m - self.p])
        k = self.search_index(ub) # 确定u所在区间[uk,uk+1)

        # print(self.u)
        # print(ub,k)
        d = []
        # 确定对于区间[uk,uk+1)有影响的控制点
        for i in range(self.p+1):
            d.append(self.ctrl_points[k - self.p + i])
        # 对控制点凸包进行o此切角
        # Pi,r = (1-alpha{i,r})P{i-1,r-1} + alpha{i,r}P{i,r-1}
        # alpha = (u - ui)/(ui+p+1-r - ui) 注意因为可能存在重复节点，所以规定0/0 = 0

        for r in range(1,self.p+1):
            for i in range(self.p,r-1,-1):
                
                alpha = ub - self.u[i+k-self.p]
                dev = self.u[i+1+k-r] - self.u[i+k-self.p]
                # if dev < 1e-6:
                #     alpha = 0.
                # else:
                #     alpha = alpha / dev
                alpha /= dev
                d[i] = (1-alpha)*d[i-1] + alpha*d[i]

        # for r in range(1,self.p+1):
        #     i = k
        #     j = self.p
        #     while i >= k - self.p + r:

        #         alpha = ub - self.u[i]
        #         dev = self.u[i+self.p+1-r] - self.u[i]
        #         if dev < 1e-6:
        #             alpha = 0.
        #         else:
        #             alpha = alpha / dev

        #         d[j] = (1-alpha)*d[j-1] + alpha*d[j]
        #         i-=1
        #         j-=1


        return d[self.p]

    def calcd(self,u):# deBoor 迭代形式
        if self.v_bspline is None:
            self.get_v_spline()

        return self.v_bspline.calc(u)

    def calcdd(self,u):# deBoor 迭代形式
        if self.a_bspline is None:
            self.get_a_spline()

        return self.a_bspline.calc(u)

    def calcddd(self,u):# deBoor 迭代形式
        if self.j_bspline is None:
            self.get_j_spline()

        return self.j_bspline.calc(u)

class qnonuniform_clamped_bspline(quniform_clamped_bspline):
    def __init__(self,ctrl_points,degrees,time_span):
        self.ctrl_points = ctrl_points
        self.p = degrees
        self.n = len(ctrl_points) - 1
        self.m = self.n + self.p + 1 # 区间段数

        # time_span为每段的dt 共控制点数-多项式次数段

        self.span = time_span
        self.u = [0. for i in range(self.p+1)]
        for i in range(len(time_span)):
            self.u.append(self.u[-1] + time_span[i])
        self.u.extend([self.u[-1] for i in range(self.p)])
        print("T:",self.u)
        self.v_ctrl_points = []
        self.a_ctrl_points = []
        self.j_ctrl_points = []
        self.v_bspline = None
        self.a_bspline = None
        self.j_bspline = None

    def get_v_spline(self):
        # The derivative of a b-spline is also a b-spline, its order become p_-1
        # control point Qi = p_*(Pi+1-Pi)/(ui+p_+1-ui+1)
        if self.p-1 < 0:
            return
        
        # timespan是一个数啊！上面还len(time_span)
        for i in range(len(self.ctrl_points)-1):
            self.v_ctrl_points.append(self.p*(self.ctrl_points[i+1] - self.ctrl_points[i])/\
            (self.u[i+self.p+1] - self.u[i+1]))
        self.v_bspline = qnonuniform_clamped_bspline(self.v_ctrl_points,self.p-1,self.span)

    def get_a_spline(self):
        if self.v_bspline is None:
            self.get_v_spline()

        self.v_bspline.get_v_spline()
        self.a_ctrl_points = self.v_bspline.v_ctrl_points
        self.a_bspline = self.v_bspline.v_bspline
    
    def get_j_spline(self):
        if self.a_bspline is None:
            self.get_a_spline()

        self.a_bspline.get_v_spline()
        self.j_ctrl_points = self.a_bspline.v_ctrl_points
        self.j_bspline = self.a_bspline.v_bspline


class quniform_clamped_bspline2D(object):
    def __init__(self,ctrl_points_x,ctrl_points_y,degrees,T):
        self.sx = quniform_clamped_bspline(ctrl_points_x,degrees,T)
        self.sy = quniform_clamped_bspline(ctrl_points_y,degrees,T)

        self.x = []
        self.y = []
        self.s = []

        self.sample_points = list()
        self.sample_ss = list()
        self.length = -1.
        # for s in np.arange(0.0, self.get_length(), 0.5):
        #     x, y = self.calc_position_s(s)
        #     self.sample_points.append((x, y))
        #     self.sample_ss.append(s)
        # self.kd_tree = KDTree(self.sample_points)

    def get_length(self,num = 100):
        if self.length < 0:
            for i in range(num):
                self.x.append(self.sx.calc(1./num))
                self.y.append(self.sy.calc(1./num))
 
            dx = np.diff(self.x)
            dy = np.diff(self.y)
            ds = np.hypot(dx, dy)
            self.s = [0]
            self.s.extend(np.cumsum(ds))
            self.length = self.s[-1]

        return self.length

    def calc_position_u(self, u): # u--[0,T]
        # 拼接
        if u < 0:
            yaw = self.calc_yaw_u(0) + math.pi
            x, y = self.calc_position_u(0)
            x += abs((u-0)/(self.sx.u[-1] - 0)*self.length) * math.cos(yaw)
            y += abs((u-0)/(self.sx.u[-1] - 0)*self.length) * math.sin(yaw)
            return x, y

        elif u > self.sx.u[-1]:
            yaw = self.calc_yaw_u(self.sx.u[-1]) + math.pi
            x, y = self.calc_position_u(self.sx.u[-1])
            x += abs((u-self.sx.u[-1])/(self.sx.u[-1] - 0)*self.length) * math.cos(yaw)
            y += abs((u-self.sx.u[-1])/(self.sx.u[-1] - 0)*self.length) * math.sin(yaw)
            return x, y

        x = self.sx.calc(u)
        y = self.sy.calc(u)

        return x,y


    def calc_yaw_u(self, u):
        """
        calc yaw
        """
        if u < 0:
            yaw = self.calc_yaw(0)
            return yaw
        elif u > self.sx.u[-1]:
            yaw = self.calc_yaw(self.sx.u[-1])
            return yaw
        dx = self.sx.calcd(u)
        dy = self.sy.calcd(u)
        
        yaw = math.atan2(dy, dx)
        return yaw

    def calc_curvature_u(self, u):
        """
        calc curvature
        """
        if u < 0:
            return self.calc_curvature(0)
        elif u > self.sx.u[-1]:
            return self.calc_curvature(self.sx.u[-1])
        dx = self.sx.calcd(u)
        ddx = self.sx.calcdd(u)
        dy = self.sy.calcd(u)
        ddy = self.sy.calcdd(u)
        # 分母为0
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_dcurvature_u(self, u):
        """
        calc dcurvature/ds
        """
        if u < 0:
            return self.calc_dcurvature(0)
        elif u > self.sx.u[-1]:
            return self.calc_dcurvature(self.sx.u[-1])

        dx = self.sx.calcd(u)
        ddx = self.sx.calcdd(u)
        dddx = self.sx.calcddd(u)

        dy = self.sy.calcd(u)
        ddy = self.sy.calcdd(u)
        dddy = self.sy.calcddd(u)

        a = dx*ddy-dy*ddx
        b = dx*dddy-dy*dddx
        c = dx*ddx+dy*ddy
        d = dx*dx+dy*dy
        return (b*d-3.0*a*c)/(d*d*d)

    def calcd(self,u):# deBoor 迭代形式
        if u < 0:
            return self.calcd(0)
        elif u > self.sx.u[-1]:
            return self.calcd(self.sx.u[-1])

        vx = self.sx.calcd(u)
        vy = self.sy.calcd(u)

        return vx,vy

    def calcdd(self,u):# deBoor 迭代形式
        if u < 0:
            return self.calcdd(0)
        elif u > self.sx.u[-1]:
            return self.calcdd(self.sx.u[-1])

        ax = self.sx.calcdd(u)
        ay = self.sy.calcdd(u)

        return ax,ay

    def calcddd(self,u):# deBoor 迭代形式
        if u < 0:
            return self.calcddd(0)
        elif u > self.sx.u[-1]:
            return self.calcddd(self.sx.u[-1])

        jx = self.sx.calcddd(u)
        jy = self.sy.calcddd(u)

        return jx,jy

    def calc_position_s(self, s):
        pass

class qnonuniform_clamped_bspline2D(quniform_clamped_bspline2D):
    def __init__(self,ctrl_points_x,ctrl_points_y,degrees,time_span):
        self.sx = qnonuniform_clamped_bspline(ctrl_points_x,degrees,time_span)
        self.sy = qnonuniform_clamped_bspline(ctrl_points_y,degrees,time_span)
        self.x = []
        self.y = []
        self.s = []

        self.sample_points = list()
        self.sample_ss = list()
        self.length = -1.

class quniform_piecewise_bspline:
    def __init__(self,ctrl_points_xs,ctrl_points_ys,degrees,start_time):
        self.splines = []
        time_spans = [] # 每条轨迹的时长
        self.start_time = start_time
        for i in range(1,len(start_time)):
            time_spans.append(start_time[i] - start_time[i-1])



        for i in range(len(ctrl_points_xs)):
            self.splines.append(quniform_clamped_bspline2D(ctrl_points_xs[i],ctrl_points_ys[i],degrees,time_spans[i]))

    def find_which_spl(self,t):
        # j = bisect.bisect_left(self.u, u) - 1 # 最后一个< ,lo = self.p
        # bisect_right 第一个>传入数
        j = bisect.bisect_right(self.start_time, t) - 1 # 最后一个<= ,lo = self.p
        if j < 0:
            j = 0
        elif j >= len(self.splines):
            j = len(self.splines) - 1

        return j,t - self.start_time[j]

    def calc_position(self, t): # u--[0,T]
        # 拼接
        j,dt = self.find_which_spl(t)
        x,y = self.splines[j].calc_position_u(dt)
        return x,y
    
    def calc_v(self, t): # u--[0,T]
        # 拼接
        j,dt = self.find_which_spl(t)
        vx,vy = self.splines[j].calcd(dt)
        return vx,vy
    def calc_a(self, t): # u--[0,T]
        # 拼接
        j,dt = self.find_which_spl(t)
        ax,ay = self.splines[j].calcdd(dt)
        return ax,ay
    def calc_j(self, t): # u--[0,T]
        # 拼接
        j,dt = self.find_which_spl(t)
        jx,jy = self.splines[j].calcddd(dt)
        return jx,jy

    def calc_yaw(self, t):
        j,dt = self.find_which_spl(t)
        yaw = self.splines[j].calc_yaw_u(dt)
        return yaw

    def calc_curvature(self, t):
        j,dt = self.find_which_spl(t)
        k = self.splines[j].calc_curvature_u(dt)
        return k

    def calc_dcurvature(self, t):
        j,dt = self.find_which_spl(t)
        dk = self.splines[j].calc_dcurvature_u(dt)
        return dk

# 直接多项式实现
# 两种非均的实现
# 插值要用到的由坐标点反求控制点
class cubic_quniform_bspline:
    def __init__(self,ctrl_points,T):
        # self.n_p1 = len(ctrl_points) + 2*(3-1) # 加上clamped重复的控制点
        # self.m = len(ctrl_points) + 3 # self.n_p1 + 3 # 区间段数
        # self.ctrl_points = [ctrl_points[0],ctrl_points[0]]
        # self.ctrl_points.extend(ctrl_points)
        # self.ctrl_points.append(ctrl_points[-1])
        # self.ctrl_points.append(ctrl_points[-1])
        # self.T = T

        # self.u = [0.,0.,0.,0.] #c 区间端点 clamp起点和终点重复3+1次
        # for i in range(self.m-2*3):
        #     self.u.append(self.u[-1]+float(self.T)/(self.m-2*3))

        # self.u.append(float(self.T))
        # self.u.append(float(self.T))
        # self.u.append(float(self.T))
        
        # # print(self.ctrl_points)
        # print(self.u)

        self.n = len(ctrl_points) - 1
        self.m = self.n + 3 + 1
        self.T = float(T)
        self.u = U_quasi_uniform(self.n,3,self.T)# [0. for i in range(self.m+1)]
        self.ctrl_points = ctrl_points
        print(self.u)
        # self.ctrl_points = [ctrl_points[0],ctrl_points[0]]
        # self.ctrl_points.extend(ctrl_points)
        # self.ctrl_points.append(ctrl_points[-1])
        # self.ctrl_points.append(ctrl_points[-1])
        self.dt = self.T/(self.m - 2*3)
        self.get_ddddddd_ctrl_points()


    # 计算u落在哪个分段上
    # def search_index(self, u):
    #     """
    #     search data segment index
    #     """
    #     # j = bisect.bisect_left(self.u, u) # 第一个>=
    #     # bisect.bisect_right #第一个>
    #     # j -= 1 if j >= len(self.u) else 0

    #     j = bisect.bisect(self.u, u) - 1 # 最后一个<=
    #     j -= 1 if j >= len(self.u) else 0

    #     return j
    def get_ddddddd_ctrl_points(self):
        self.d_ctrl_points = []
        self.dd_ctrl_points = []
        self.ddd_ctrl_points = []
        for i in range(1,len(self.ctrl_points)):
            self.d_ctrl_points.append((self.ctrl_points[i] - self.ctrl_points[i-1])/self.dt)
        dt2 = self.dt*self.dt
        for i in range(2,len(self.ctrl_points)):
            self.dd_ctrl_points.append((self.ctrl_points[i] - 2*self.ctrl_points[i-1] + self.ctrl_points[i-2])/dt2)
        dt3 = dt2*self.dt
        for i in range(3,len(self.ctrl_points)):
            self.ddd_ctrl_points.append((self.ctrl_points[i] - 3*self.ctrl_points[i-1] + 3*self.ctrl_points[i-2] - self.ctrl_points[i-3])/dt3)
        
        self.vdt = self.T/(self.m - 2*3)
        self.adt = self.T/(self.m - 1*3)
        self.jdt = self.T/(self.m - 0*3)

        self.uv = U_quasi_uniform(len(self.d_ctrl_points)-1,2,self.T)# [0. for i in range(self.m+1)]
        self.ua = U_quasi_uniform(len(self.dd_ctrl_points)-1,1,self.T)# [0. for i in range(self.m+1)]
        self.uj = U_quasi_uniform(len(self.ddd_ctrl_points)-1,0,self.T)# [0. for i in range(self.m+1)]


    def calc(self,t):
        # u -- [0,1]
        # u - u_{i+3}

        # u = t/float(self.T)
        # u_inn = 3 + u * (self.n + 1 - 3) # u_inn from 3 to n
        # interval = int(math.floor(u_inn))
        # # print(u_inn,interval)
        # t = u_inn - interval
        # if interval == self.n + 1 and abs(t)<1e-6:
        #     interval = self.n
        #     t = 1.
        # t2 = t*t
        # t3 = t2*t
        # cu = 0
        # cu += self.ctrl_points[interval-3]*(1-t)**3
        # cu += self.ctrl_points[interval-3+1]*(3*t3 -6*t2 + 4)
        # cu += self.ctrl_points[interval-3+2]*(-3*t3 + 3*t2 + 3*t +1)
        # cu += self.ctrl_points[interval-3+3]*t3


        u = min(max(self.u[3],t),self.u[-3-1])
        # k = self.search_index(u) # 确定u所在区间[uk,uk+1)
        k = bisect.bisect_right(self.u, u,lo = 3,hi = self.m - 3) - 1 # 最后一个<= ,lo = self.p
        k -= 1 if k >= len(self.u) else 0

        # print(u,k)
        # print(self.u)
        # print(self.ctrl_points)
        # print(self.dt)
        t = (u - self.u[k])/self.dt # uk--ui+3

        t2 = t*t
        t3 = t2*t
        cu = 0
        cu += self.ctrl_points[k-3]*(1-t)**3
        cu += self.ctrl_points[k-3+1]*(3*t3 -6*t2 + 4)
        cu += self.ctrl_points[k-3+2]*(-3*t3 + 3*t2 + 3*t +1)
        cu += self.ctrl_points[k-3+3]*t3
        
        # cu += self.ctrl_points[k-3]*(-t3 + 3*t2 - 3*t + 1)
        # cu += self.ctrl_points[k-3+1]*(3*t3 -6*t2 + 3*t)
        # cu += self.ctrl_points[k-3+2]*(-3*t3 + 3*t2)
        # cu += self.ctrl_points[k-3+3]*t3

        # cu += self.ctrl_points[k%(self.n+1)]*(1-t)**3
        # cu += self.ctrl_points[(k+1)%(self.n+1)]*(3*t3 -6*t2 + 4)
        # cu += self.ctrl_points[(k+2)%(self.n+1)]*(-3*t3 + 3*t2 + 3*t +1)
        # cu += self.ctrl_points[(k+3)%(self.n+1)]*t3

        # t = u - self.u[i] # (u - self.u[i])/1.

        return cu/6.

    def cald(self,t):
        u = min(max(self.uv[2],t),self.uv[-2-1])
        k = bisect.bisect_right(self.uv, u,lo = 2,hi = self.m - 2) - 1 # 最后一个<= ,lo = self.p
        k -= 1 if k >= len(self.uv) else 0
        t = (u - self.uv[k])/self.vdt # uk--ui+3



        t2 = t*t
        dcu = 0
        dcu += self.d_ctrl_points[k-2]*(t2-2*t+1)
        dcu += self.d_ctrl_points[k-2+1]*(-2*t2+2*t+1)
        dcu += self.d_ctrl_points[k-2+2]*t2


        return dcu/2.

    def caldd(self,t):


        u = min(max(self.ua[1],t),self.ua[-1-1])
        k = bisect.bisect_right(self.ua, u,lo = 1,hi = self.m - 1) - 1 # 最后一个<= ,lo = self.p
        k -= 1 if k >= len(self.ua) else 0
        t = (u - self.ua[k])/self.adt # uk--ui+3

        ddcu = 0
        ddcu += self.dd_ctrl_points[k-1]*(1-t)
        ddcu += self.dd_ctrl_points[k-1+1]*t

        return ddcu

    def calddd(self,t):
        u = min(max(self.uj[0],t),self.uj[-1])
        k = bisect.bisect_right(self.uj, u,lo = 0,hi = self.m) - 1 # 最后一个<= ,lo = self.p
        k -= 1 if k >= len(self.uj) else 0

        dddcu = self.ddd_ctrl_points[k]

        return dddcu

class cubic_quniform_bspline2D:
    def __init__(self,ctrl_points_x,ctrl_points_y,T):
        self.sx = cubic_quniform_bspline(ctrl_points_x,T)
        self.sy = cubic_quniform_bspline(ctrl_points_y,T)

    def calc_position_u(self, u): # u--[0,T]
        # 拼接
        # if u < 0:
        #     yaw = self.calc_yaw(0) + math.pi
        #     x, y = self.calc_position(0)
        #     x += abs(u*self.length) * math.cos(yaw)
        #     y += abs(u*self.length) * math.sin(yaw)
        #     return x, y

        # elif u > self.sx.u[-1]:
        #     yaw = self.calc_yaw(self.sx.u[-1]) + math.pi
        #     x, y = self.calc_position(self.sx.u[-1])
        #     x += abs((u-1)*self.length) * math.cos(yaw)
        #     y += abs((u-1)*self.length) * math.sin(yaw)
        #     return x, y

        x = self.sx.calc(u)
        y = self.sy.calc(u)

        return x,y




# if __name__ == "__main__":
#     # ctr_x = [-2,0,1,1.5,-2]
#     # ctr_y = [0,2,1.6,1,0,5]
#     ctr_x = [-2,0,1,1.5,2,3,2.5,1]
#     ctr_y = [0,2,1.6,1,0,-2.5,-2,0]

#     # qx = cubic_quniform_bspline(ctr_x,0)
#     # qy = cubic_quniform_bspline(ctr_y,0)

#     # q = quniform_clamped_bspline2D(ctr_x,ctr_y,3,0.2)
#     # calc_2d_spline_interpolation(q,plt.subplots()[1])

#     # qx = cubic_quniform_bspline(ctr_x,1)
#     # calc_1d_cubic_spline_interpolation(qx)

#     # q = cubic_quniform_bspline2D(ctr_x,ctr_y,1)
#     # calc_2d_cubic_spline_interpolation(q)

#     # qy = quniform_clamped_bspline(ctr_y,3,0.2)
#     # calc_2dsplit_spline_interpolation(qx,qy)
#     # print(q.u)

#     # qx = quniform_clamped_bspline(ctr_x,3,2)
#     # calc_1d_spline_interpolation(qx)


#     x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
#     y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]
#     T = 4.
#     dt = T/(len(x)-1)  # [m] distance of each interpolated points
#     ctr_x = points2uniform_bspline_ctrlp(dt,(0,0),(-1,0),x)
#     ctr_y = points2uniform_bspline_ctrlp(dt,(0,0),(-1,0),y)
#     print(ctr_x)
#     print(ctr_y)

#     # ctr_x = [-2,0,1,1.5,2,3,2.5,1]
#     # ctr_y = [0,2,1.6,1,0,-2.5,-2,0]
#     ctr_x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
#     ctr_y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]

#     # q = quniform_clamped_bspline2D(ctr_x,ctr_y,3,T)
#     # q = cubic_quniform_bspline2D(ctr_x,ctr_y,T)
#     # time_span = [0.,1.5,1.6,2,4.] # [up,un+1] 共控制点 - 多项式次数 + 1个  
#     time_span = [0.1,0.1,2.5,1.4] # time_span为每段的dt 共控制点数-多项式次数段 = 4段

#     q = qnonuniform_clamped_bspline2D(ctr_x,ctr_y,2,time_span)
#     fig,canv = plt.subplots()
#     canv.plot(x, y, "xb")
#     calc_2d_spline_interpolation(q,canv,500)

from pylab import mpl
mpl.rcParams["font.sans-serif"]=["AR PL UMing CN"]
mpl.rcParams["axes.unicode_minus"] = False
# plt.rc("font",family='AR PL UMing CN')
import random

if __name__ == "__main__":
    # ctr_x = [-2,0,1,1.5,-2]
    # ctr_y = [0,2,1.6,1,0,5]
    ctr_x = [-1.9,0,1,1.5,2,2.9,2.5,1]
    ctr_y = [0,1.9,1.6,1,0,-2.5,-1.9,0]
    boundx = [-2,-2,3,3,-2]
    boundy = [-3,2,2,-3,-3]
    # qx = cubic_quniform_bspline(ctr_x,0)
    # qy = cubic_quniform_bspline(ctr_y,0)

    q = quniform_clamped_bspline2D(ctr_x,ctr_y,3,3)
    # calc_2d_spline_interpolation(q,plt.subplots()[1])
    fig,canv = plt.subplots()

    for i in range(100):
        ctr_x = []
        ctr_y = []
        for i in range(8):
            x = -2 + 5*random.random()
            y = -3 + 5*random.random()
            ctr_x.append(x)
            ctr_y.append(y)

        canv.plot(ctr_x, ctr_y, "ob",label='控制点')
        canv.plot(boundx,boundy ,"r", label = "约束凸多边形",lw=2)#  "-r"
    
        q = quniform_clamped_bspline2D(ctr_x,ctr_y,3,3)

        calc_2d_spline_interpolation(q,canv,500)

        plt.pause(1.5)
        # canv.clf()
        fig.clf()
    

    # qx = cubic_quniform_bspline(ctr_x,1)
    # calc_1d_cubic_spline_interpolation(qx)

    # q = cubic_quniform_bspline2D(ctr_x,ctr_y,1)
    # calc_2d_cubic_spline_interpolation(q)

    # qy = quniform_clamped_bspline(ctr_y,3,0.2)
    # calc_2dsplit_spline_interpolation(qx,qy)
    # print(q.u)

    # qx = quniform_clamped_bspline(ctr_x,3,2)
    # calc_1d_spline_interpolation(qx)


    # x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    # y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]
    # T = 4.
    # dt = T/(len(x)-1)  # [m] distance of each interpolated points
    # ctr_x = points2uniform_bspline_ctrlp(dt,(0,0),(-1,0),x)
    # ctr_y = points2uniform_bspline_ctrlp(dt,(0,0),(-1,0),y)
    # print(ctr_x)
    # print(ctr_y)

    # # ctr_x = [-2,0,1,1.5,2,3,2.5,1]
    # # ctr_y = [0,2,1.6,1,0,-2.5,-2,0]
    # ctr_x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    # ctr_y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]

    # # q = quniform_clamped_bspline2D(ctr_x,ctr_y,3,T)
    # # q = cubic_quniform_bspline2D(ctr_x,ctr_y,T)
    # # time_span = [0.,1.5,1.6,2,4.] # [up,un+1] 共控制点 - 多项式次数 + 1个  
    # time_span = [0.1,0.1,2.5,1.4] # time_span为每段的dt 共控制点数-多项式次数段 = 4段

    # q = qnonuniform_clamped_bspline2D(ctr_x,ctr_y,2,time_span)
    # fig,canv = plt.subplots()
    # canv.plot(x, y, "xb")
    # calc_2d_spline_interpolation(q,canv,500)
