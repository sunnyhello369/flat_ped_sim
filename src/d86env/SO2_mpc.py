#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import matplotlib.pyplot as plt
import time
# sys.path.append(r"/home/chen/desire_10086/flat_ped_sim/src/d86env/casadi-linux-py37-v3.5.5-64bit")
# from casadi import *

"""
这5个类都是实现二维运动的模型预测控制(MPC)规划,主要区别如下:

1. SO2MPC

建立如下优化问题:

min  cost(x,u) 

s.t. x_{k+1} = f(x_k, u_k) # 系统动力学方程
     
x和u都作为优化变量。

变量数目多,需要同时优化整个状态序列x和控制序列u,计算量大。

2. SO2MPC_u

min cost(x(u), u)  

s.t. x_{k+1} = f(x_k, u_k)

仅u作为优化变量。状态x通过方程积分获得,不需优化。

降低变量维数,提高计算效率。

3. SO2MPC_dense

引入权重矩阵,构建二次型代价函数:

min 0.5 * (x-x_ref)^T * Q * (x-x_ref) + 0.5 * u^T * R * u

s.t. x_{k+1} = f(x_k, u_k)

使用稠密矩阵运算表达代价函数和约束条件。

计算效率较SO2MPC有一定提升。

4. SO2MPC_dense2 

在3的基础上进一步简化,仅关注部分状态量约束和控制量平滑。

5. SO2MPC_dense3

只保留连接两点的简化问题:

min 0.5 * u^T * R * u

s.t. x_N = x_goal

大幅降低问题复杂度,计算最高效。

以上逐步简化表达式,降低维数,使计算效率不断提升。需根据实际需求选择合适方法。

总结:

- 从SO2MPC到SO2MPC_dense3逐步简化模型,降低优化维数,提高计算效率
- 需根据实际需要在效率和表达能力上做权衡
- 实时性要求高可采用后面几种,对优化效果要求高可采用前两种
"""

class SO2MPC:
    def __init__(self

    ,N  # 规划的时间步数,即划分的轨迹点数

    ,control_period # 控制周期,即每次规划的时间步长

    # 速度、加速度、jerk的权重,用于构建代价函数
    ,v_weight=35.
    ,a_weight=35.
    ,j_weight=35.

    # 轨迹跟踪误差的权重
    # ,ref_track_weight = 15000000.
    ,ref_track_weight = 300000.

    # v_limit、a_limit、j_limit: 速度、加速度、jerk的限制
    ,v_limit=3.
    ,a_limit=3.
    ,j_limit=4.):
        # time span = 3s
        # control period = 0.05s -- 20hz
        # 轨迹点数 = time span/control period = 3/0.05 = 60
        # 状态 x,y,vx,vy,ax,ay
        # 控制量 jx,jy,jz
        # 状态和控制量都作为被优化变量  9+4 = 13
        # optim num = 13*60 = 780 优化变量数太多
        self.control_period = control_period # dt

        self.N = N # 轨迹点数
        self.cost = 0.
        self.v_weight = v_weight
        self.a_weight = a_weight
        self.j_weight = j_weight
        self.ref_track_weight = ref_track_weight

        self.lbpcp = [] # 被优化变量下界 [box1 spline[(p1x,p1y),(),...],box2 spline[]]
        self.ubpcp = [] # 被优化变量下=上界 [box1 spline[(p1x,p1y),(),...],box2 spline[]]
        self.other_constraints = [] # 约束，比如点之间的连续性约束
        self.lbc = [] # 约束的下界
        self.ubc = [] # 约束的上界

        self.v_limit = v_limit
        self.a_limit = a_limit
        self.j_limit = j_limit

        # 存储规划的位置、速度、加速度、jerk轨迹
        self.p = []
        self.v = []
        self.a = []
        self.j = []


        # # 初态为硬约束，不写入
        # for i in range(self.N):
        #     self.p.append((SX.sym('x_' + str(i)),SX.sym('y_'+ str(i))))
        #     self.lbpcp.append((-30.,-30.))
        #     self.ubpcp.append((30.,30.))
        #     self.v.append((SX.sym('vx_' + str(i)),SX.sym('vy_'+ str(i))))
        #     self.lbpcp.append((-v_limit,-v_limit))
        #     self.ubpcp.append((v_limit,v_limit))
        #     self.a.append((SX.sym('ax_' + str(i)),SX.sym('ay_'+ str(i))))
        #     self.lbpcp.append((-a_limit,-a_limit))
        #     self.ubpcp.append((a_limit,a_limit))
        #     self.j.append((SX.sym('jx_' + str(i)),SX.sym('jy_'+ str(i))))
        #     self.lbpcp.append((-j_limit,-j_limit))
        #     self.ubpcp.append((j_limit,j_limit)) 

    def get_position_constraints(self,convex_polygon_vertex):
        # 初态和终态位置
        # 初态和终态控制点为常量，应该剔除出优化

        # 点在凸多边形内部约束，顺时针将顶点和点所成直线叉乘，每个面积都为正>0写成Ax<b形式
        for which_point in range(len(self.p)):
            for i in range(1,len(convex_polygon_vertex)):
                    # v1xv0 消除交叉项后的结果
                    self.other_constraints.append(convex_polygon_vertex[i-1][0]*convex_polygon_vertex[i][1] -
                    convex_polygon_vertex[i-1][1]*convex_polygon_vertex[i][0] +
                    (convex_polygon_vertex[i][0] - convex_polygon_vertex[i-1][0])*self.p[which_point][1]+
                    (convex_polygon_vertex[i-1][1] - convex_polygon_vertex[i][1])*self.p[which_point][0])
                    # self.lbc.append(0)
                    # self.ubc.append(float('inf'))
                    self.lbc.append(-float('inf'))
                    self.ubc.append(0.)

    def get_model_constraints(self,start):
        # 三阶积分模型
        # p_k+1        1      dt       dt^2/2            p_k                      dt^3/6
        # v_k+1   =   0        1            dt        *      v_k                       dt^2/2    * j_k
        # a_k+1        0        0              1                a_k                           dt 

        self.other_constraints = []
        self.lbc = []
        self.ubc = []
        # 稀疏形式MPC 想 x,u都作为被优化变量
        # start--j0--> p0,v0,a0 --j1--> .... --jN-1-->pN-1,vN-1,aN-1
        print(self.j)
        px = start[0][0] + start[1][0]*self.control_period + start[2][0]*self.control_period**2/2. + self.j[0][0]*self.control_period**3/6.
        py = start[0][1] + start[1][1]*self.control_period + start[2][1]*self.control_period**2/2. + self.j[0][1]*self.control_period**3/6.
        self.other_constraints.append(px - self.p[0][0])
        self.lbc.append(0.)
        self.ubc.append(0.)
        self.other_constraints.append(py - self.p[0][1])
        self.lbc.append(0.)
        self.ubc.append(0.)
    
        vx = start[1][0] + start[2][0]*self.control_period + self.j[0][0]*self.control_period**2/2.
        vy = start[1][1] + start[2][1]*self.control_period +  self.j[0][1]*self.control_period**2/2.
        self.other_constraints.append(vx - self.v[0][0])
        self.lbc.append(0.)
        self.ubc.append(0.)
        self.other_constraints.append(vy - self.v[0][1])
        self.lbc.append(0.)
        self.ubc.append(0.)

        ax = start[2][0] + self.j[0][0]*self.control_period
        ay = start[2][1] +  self.j[0][1]*self.control_period
        self.other_constraints.append(ax - self.a[0][0])
        self.lbc.append(0.)
        self.ubc.append(0.)
        self.other_constraints.append(ay - self.a[0][1])
        self.lbc.append(0.)
        self.ubc.append(0.)

        for i in range(1,self.N):
            px = self.p[i-1][0] + self.v[i-1][0]*self.control_period + self.a[i-1][0]*self.control_period**2/2. + self.j[i][0]*self.control_period**3/6.
            py = self.p[i-1][1] + self.v[i-1][1]*self.control_period + self.a[i-1][1]*self.control_period**2/2. + self.j[i][1]*self.control_period**3/6.
            self.other_constraints.append(px - self.p[i][0])
            self.lbc.append(0.)
            self.ubc.append(0.)
            self.other_constraints.append(py - self.p[i][1])
            self.lbc.append(0.)
            self.ubc.append(0.)
        
            vx = self.v[i-1][0] + self.a[i-1][0]*self.control_period + self.j[i][0]*self.control_period**2/2.
            vy = self.v[i-1][1] + self.a[i-1][1]*self.control_period +  self.j[i][1]*self.control_period**2/2.
            self.other_constraints.append(vx - self.v[i][0])
            self.lbc.append(0.)
            self.ubc.append(0.)
            self.other_constraints.append(vy - self.v[i][1])
            self.lbc.append(0.)
            self.ubc.append(0.)

            ax = self.a[i-1][0] + self.j[i][0]*self.control_period
            ay = self.a[i-1][1] +  self.j[i][1]*self.control_period
            self.other_constraints.append(ax - self.a[i][0])
            self.lbc.append(0.)
            self.ubc.append(0.)
            self.other_constraints.append(ay - self.a[i][1])
            self.lbc.append(0.)
            self.ubc.append(0.)
    
    def calc_cost(self,ref_traj_info):
        # ref_traj_info -- [which state][p,v,a][x,y]
        self.cost = 0.
        for i in range(self.N):
            # minimize track error
            self.cost += self.ref_track_weight*(ref_traj_info[i][0][0] - self.p[i][0])**2
            self.cost += self.ref_track_weight*(ref_traj_info[i][0][1] - self.p[i][1])**2
            self.cost += self.v_weight*(ref_traj_info[i][1][0] - self.v[i][0])**2
            self.cost += self.v_weight*(ref_traj_info[i][1][1] - self.v[i][1])**2

            # minimize input
            self.cost += self.j_weight*self.j[i][0]**2
            self.cost += self.j_weight*self.j[i][1]**2
            if i == self.N-1:
                self.cost += self.a_weight*self.a[i][0]**2
                self.cost += self.a_weight*self.a[i][1]**2
            else:
                self.cost += self.a_weight*(ref_traj_info[i][2][0] - self.a[i][0])**2
                self.cost += self.a_weight*(ref_traj_info[i][2][1] - self.a[i][1])**2

    def formulate(self,start,ref_traj_info,convex_polygon_vertex = None):
        if convex_polygon_vertex is not None:
            self.get_position_constraints(convex_polygon_vertex)
        self.get_model_constraints(start)
        self.calc_cost(ref_traj_info)

        x = []
        lbx = []
        ubx = []

        for i in range(self.N):
            # p
            x.append(self.p[i][0])
            lbx.append(self.lbpcp[4*i][0])
            ubx.append(self.ubpcp[4*i][0])
            x.append(self.p[i][1])
            lbx.append(self.lbpcp[4*i][1])
            ubx.append(self.ubpcp[4*i][1])
            # v
            x.append(self.v[i][0])
            lbx.append(self.lbpcp[4*i+1][0])
            ubx.append(self.ubpcp[4*i+1][0])
            x.append(self.v[i][1])
            lbx.append(self.lbpcp[4*i+1][1])
            ubx.append(self.ubpcp[4*i+1][1])
            # a
            x.append(self.a[i][0])
            lbx.append(self.lbpcp[4*i+2][0])
            ubx.append(self.ubpcp[4*i+2][0])
            x.append(self.a[i][1])
            lbx.append(self.lbpcp[4*i+2][1])
            ubx.append(self.ubpcp[4*i+2][1])
            # j
            x.append(self.j[i][0])
            lbx.append(self.lbpcp[4*i+3][0])
            ubx.append(self.ubpcp[4*i+3][0])
            x.append(self.j[i][1])
            lbx.append(self.lbpcp[4*i+3][1])
            ubx.append(self.ubpcp[4*i+3][1])

        qp = {'x':vertcat(*x), 'f':self.cost, 'g':vertcat(*self.other_constraints)}
        # Solve with 
        solver = qpsol('solver', 'qpoases', qp, {'sparse':True,'printLevel':'none'} )# printLevel':'none' 

        # Get the optimal solution
        try:
            sol = solver(lbx=lbx, ubx=ubx, lbg=self.lbc, ubg=self.ubc)
        except:
            return False,None,None,None,None,None,None,None,None

        # print(sol)
        # 分段b样条二维控制点
        x_opt = []
        y_opt = []
        vx_opt = []
        vy_opt = []
        ax_opt = []
        ay_opt = []
        jx_opt = []
        jy_opt = [] 

        for i in range(self.N):
            x_opt.append(float(sol['x'][8*i]))
            y_opt.append(float(sol['x'][8*i+1]))
            vx_opt.append(float(sol['x'][8*i+2]))
            vy_opt.append(float(sol['x'][8*i+3]))
            ax_opt.append(float(sol['x'][8*i+4]))
            ay_opt.append(float(sol['x'][8*i+5]))
            jx_opt.append(float(sol['x'][8*i+6]))
            jy_opt.append(float(sol['x'][8*i+7]))
            # print((x_opt[-1],y_opt[-1]),(vx_opt[-1],vy_opt[-1]),(ax_opt[-1],ay_opt[-1]),(jx_opt[-1],jy_opt[-1]))
        p_cost = 0.
        v_cost = 0.
        a_cost = 0.
        j_cost = 0.
        for i in range(self.N):
            # minimize track error
            
            p_cost += (ref_traj_info[i][0][0] - x_opt[i])**2
            p_cost += (ref_traj_info[i][0][1] - y_opt[i])**2
            v_cost += (ref_traj_info[i][1][0] - vx_opt[i])**2
            v_cost += (ref_traj_info[i][1][1] - vy_opt[i])**2
            a_cost += (ref_traj_info[i][2][0] - ax_opt[i])**2
            a_cost += (ref_traj_info[i][2][1] - ay_opt[i])**2
            # minimize input
            j_cost += jx_opt[i]**2
            j_cost += jy_opt[i]**2

        return True,x_opt,y_opt,vx_opt,vy_opt,ax_opt,ay_opt,jx_opt,jy_opt



class SO2MPC_u:
    def __init__(self
    ,N
    ,control_period
    ,v_weight=35.
    ,a_weight=35.
    ,j_weight=35.
    # ,ref_track_weight = 15000000.
    ,ref_track_weight = 300000.
    ,v_limit=3.
    ,a_limit=3.
    ,j_limit=4.):
        # time span = 3s
        # control period = 0.05s -- 20hz
        # 轨迹点数 = time span/control period = 3/0.05 = 60
        # 状态 x,y,vx,vy,ax,ay
        # 控制量 jx,jy,jz
        # 状态和控制量都作为被优化变量  9+4 = 13
        # optim num = 13*60 = 780 优化变量数太多
        self.control_period = control_period # dt

        self.N = N # 轨迹点数
        self.cost = 0.
        self.v_weight = v_weight
        self.a_weight = a_weight
        self.j_weight = j_weight
        self.ref_track_weight = ref_track_weight

        self.lbpcp = [] # 被优化变量下界 [box1 spline[(p1x,p1y),(),...],box2 spline[]]
        self.ubpcp = [] # 被优化变量下=上界 [box1 spline[(p1x,p1y),(),...],box2 spline[]]
        self.other_constraints = [] # 约束，比如点之间的连续性约束
        self.lbc = [] # 约束的下界
        self.ubc = [] # 约束的上界

        self.v_limit = v_limit
        self.a_limit = a_limit
        self.j_limit = j_limit
        self.p = []
        self.v = []
        self.a = []
        self.j = []


        # # 初态为硬约束，不写入
        # for i in range(self.N):
        #     self.j.append((SX.sym('jx_' + str(i)),SX.sym('jy_'+ str(i))))
        #     self.lbpcp.append((-j_limit,-j_limit))
        #     self.ubpcp.append((j_limit,j_limit)) 

    def get_position_constraints(self,convex_polygon_vertex):
        # 初态和终态位置
        # 初态和终态控制点为常量，应该剔除出优化

        # 点在凸多边形内部约束，顺时针将顶点和点所成直线叉乘，每个面积都为正>0写成Ax<b形式
        for which_point in range(len(self.p)):
            for i in range(1,len(convex_polygon_vertex)):
                    # v1xv0 消除交叉项后的结果
                    self.other_constraints.append(convex_polygon_vertex[i-1][0]*convex_polygon_vertex[i][1] -
                    convex_polygon_vertex[i-1][1]*convex_polygon_vertex[i][0] +
                    (convex_polygon_vertex[i][0] - convex_polygon_vertex[i-1][0])*self.p[which_point][1]+
                    (convex_polygon_vertex[i-1][1] - convex_polygon_vertex[i][1])*self.p[which_point][0])
                    # self.lbc.append(0)
                    # self.ubc.append(float('inf'))
                    self.lbc.append(-float('inf'))
                    self.ubc.append(0.)

    def get_model_constraints(self,start=[(1,2),(0,0),(0,0)]):
        """
        构建三阶模型的运动学约束条件

        Parameters
        ----------
        start: 初始状态,包含位置p、速度v、加速度a,例如[(1,2),(3,4),(5,6)]

        Returns
        -------
        self.other_constraints: 约束方程
        self.lbc: 约束下界
        self.ubc: 约束上界
        self.p/v/a: 存储预测的位置/速度/加速度序列
        """
        # 三阶积分模型
        # p_k+1        1      dt       dt^2/2            p_k                      dt^3/6
        # v_k+1   =   0        1            dt        *      v_k                       dt^2/2    * j_k
        # a_k+1        0        0              1                a_k                           dt 

        self.other_constraints = []
        self.lbc = []
        self.ubc = []
        # 稀疏形式MPC 想 x,u都作为被优化变量
        # start--j0--> p0,v0,a0 --j1--> .... --jN-1-->pN-1,vN-1,aN-1
        print("++++++++++++")
        print(start[2][0])
        a=np.array(start)
        print(a.shape)
        print("++++++++++++")
        if len(start) < 3:
            raise ValueError("start must have at least 3 elements for pos/vel/acc")
        px = start[0][0] + start[1][0]*self.control_period + start[2][0]*self.control_period**2/2. + self.j[0][0]*self.control_period**3/6.
        py = start[0][1] + start[1][1]*self.control_period + start[2][1]*self.control_period**2/2. + self.j[0][1]*self.control_period**3/6.
        self.p.append((px,py))
        self.other_constraints.append(px)
        self.lbc.append(-30.)
        self.ubc.append(30.)
        self.other_constraints.append(py)
        self.lbc.append(-30.)
        self.ubc.append(30.)

        # ? differ to paper's discrete model
        vx = start[1][0] + start[2][0]*self.control_period + self.j[0][0]*self.control_period**2/2.
        vy = start[1][1] + start[2][1]*self.control_period +  self.j[0][1]*self.control_period**2/2.
        self.v.append((vx,vy))
        self.other_constraints.append(vx)
        self.lbc.append(-self.v_limit)
        self.ubc.append(self.v_limit)
        self.other_constraints.append(vy)
        self.lbc.append(-self.v_limit)
        self.ubc.append(self.v_limit)

        ax = start[2][0] + self.j[0][0]*self.control_period
        ay = start[2][1] +  self.j[0][1]*self.control_period
        self.a.append((ax,ay))
        self.other_constraints.append(ax)
        self.lbc.append(-self.a_limit)
        self.ubc.append(self.a_limit)
        self.other_constraints.append(ay)
        self.lbc.append(-self.a_limit)
        self.ubc.append(self.a_limit)


        for i in range(1,self.N):
            px = self.p[i-1][0] + self.v[i-1][0]*self.control_period + self.a[i-1][0]*self.control_period**2/2. + self.j[i][0]*self.control_period**3/6.
            py = self.p[i-1][1] + self.v[i-1][1]*self.control_period + self.a[i-1][1]*self.control_period**2/2. + self.j[i][1]*self.control_period**3/6.
            self.p.append((px,py))
            self.other_constraints.append(px)
            self.lbc.append(-30.)
            self.ubc.append(30.)
            self.other_constraints.append(py)
            self.lbc.append(-30.)
            self.ubc.append(30.)
        
            vx = self.v[i-1][0] + self.a[i-1][0]*self.control_period + self.j[i][0]*self.control_period**2/2.
            vy = self.v[i-1][1] + self.a[i-1][1]*self.control_period +  self.j[i][1]*self.control_period**2/2.
            self.v.append((vx,vy))
            self.other_constraints.append(vx)
            self.lbc.append(-self.v_limit)
            self.ubc.append(self.v_limit)
            self.other_constraints.append(vy)
            self.lbc.append(-self.v_limit)
            self.ubc.append(self.v_limit)

            ax = self.a[i-1][0] + self.j[i][0]*self.control_period
            ay = self.a[i-1][1] +  self.j[i][1]*self.control_period
            self.a.append((ax,ay))
            self.other_constraints.append(ax)
            self.lbc.append(-self.a_limit)
            self.ubc.append(self.a_limit)
            self.other_constraints.append(ay)
            self.lbc.append(-self.a_limit)
            self.ubc.append(self.a_limit)

    def calc_cost_ref(self,ref_traj_info):
        # ref_traj_info -- [which state][p,v,a][x,y]
        self.cost = 0.
        for i in range(self.N):
            # minimize track error
            self.cost += self.ref_track_weight*(ref_traj_info[i][0][0] - self.p[i][0])**2
            self.cost += self.ref_track_weight*(ref_traj_info[i][0][1] - self.p[i][1])**2
            self.cost += self.v_weight*(ref_traj_info[i][1][0] - self.v[i][0])**2
            self.cost += self.v_weight*(ref_traj_info[i][1][1] - self.v[i][1])**2

            # minimize input
            self.cost += self.j_weight*self.j[i][0]**2
            self.cost += self.j_weight*self.j[i][1]**2
            if i == self.N-1:
                self.cost += self.a_weight*self.a[i][0]**2
                self.cost += self.a_weight*self.a[i][1]**2
            else:
                self.cost += self.a_weight*(ref_traj_info[i][2][0] - self.a[i][0])**2
                self.cost += self.a_weight*(ref_traj_info[i][2][1] - self.a[i][1])**2

    def calc_cost_specify(self,specify):
        # specify -- [which state][p,v,a][x,y]
        self.cost = 0.
        for i in range(self.N):
            # minimize input
            self.cost += self.j_weight*self.j[i][0]**2
            self.cost += self.j_weight*self.j[i][1]**2
            self.cost += self.a_weight*self.a[i][0]**2
            self.cost += self.a_weight*self.a[i][1]**2

        self.cost += self.v_weight*self.v[-1][0]**2
        self.cost += self.v_weight*self.v[-1][1]**2

        # minimize track error
        self.cost += self.ref_track_weight*(specify[0] - self.p[-1][0])**2
        self.cost += self.ref_track_weight*(specify[1] - self.p[-1][1])**2

    def formulate(self,start,specify = None,ref_traj_info = None,convex_polygon_vertex = None):
        """
        构建并求解了基于模型预测控制的二维运动规划

        Parameters
        ----------
        start: 初始状态,包含位置、速度、加速度,例如 [(1, 2), (3, 4), (5, 6)]
        specify: 指定的目标位置,比如(10, 20)
        ref_traj_info: 参考轨迹信息,包含位置、速度、加速度,用于跟踪误差最小化
        convex_polygon_vertex: 凸多边形顶点,用于限制运动范围

        Returns
        -------
        成功与否的标志
        位置、速度、加速度、控制jerk的最优轨迹
        """
        if ref_traj_info is None and specify is None:
            return False,None,None,None,None,None,None,None,None

        self.get_model_constraints(start)

        if convex_polygon_vertex is not None:
            self.get_position_constraints(convex_polygon_vertex)
        if ref_traj_info is not None:
            self.calc_cost_ref(ref_traj_info)
        if specify is not None:
            self.calc_cost_specify(specify)


        x = []
        lbx = []
        ubx = []

        for i in range(self.N):
            # j
            x.append(self.j[i][0])
            lbx.append(self.lbpcp[i][0])
            ubx.append(self.ubpcp[i][0])
            x.append(self.j[i][1])
            lbx.append(self.lbpcp[i][1])
            ubx.append(self.ubpcp[i][1])

        qp = {'x':vertcat(*x), 'f':self.cost, 'g':vertcat(*self.other_constraints)}
        # Solve with 
        solver = qpsol('solver', 'qpoases', qp, {'sparse':True,'printLevel':'none'} )# printLevel':'none' 

        # Get the optimal solution
        try:
            sol = solver(lbx=lbx, ubx=ubx, lbg=self.lbc, ubg=self.ubc)
        except:
            return False,None,None,None,None,None,None,None,None

        # print(sol)
        # 分段b样条二维控制点
        jx_opt = [float(sol['x'][0])]
        jy_opt = [float(sol['x'][1])] 
        ax_opt = [start[2][0] + jx_opt[0]*self.control_period]
        ay_opt = [start[2][1] + jy_opt[0]*self.control_period]
        vx_opt = [start[1][0] + start[2][0]*self.control_period + jx_opt[0]*self.control_period**2/2.]
        vy_opt = [start[1][1] + start[2][1]*self.control_period + jy_opt[0]*self.control_period**2/2.]
        x_opt = [start[0][0] + start[1][0]*self.control_period + start[2][0]*self.control_period**2/2. + jx_opt[0]*self.control_period**3/6.]
        y_opt = [start[0][1] + start[1][1]*self.control_period + start[2][1]*self.control_period**2/2. + jy_opt[0]*self.control_period**3/6.]

        for i in range(1,self.N):
            jx_opt.append(float(sol['x'][2*i]))
            jy_opt.append(float(sol['x'][2*i+1]))

            ax = ax_opt[-1] + jx_opt[-1]*self.control_period
            ay = ay_opt[-1] + jy_opt[-1]*self.control_period
            vx = vx_opt[-1] + ax_opt[-1]*self.control_period + jx_opt[-1]*self.control_period**2/2.
            vy = vy_opt[-1] + ay_opt[-1]*self.control_period + jy_opt[-1]*self.control_period**2/2.
            px = x_opt[-1] + vx_opt[-1]*self.control_period + ax_opt[-1]*self.control_period**2/2. + jx_opt[-1]*self.control_period**3/6.
            py = y_opt[-1] + vy_opt[-1]*self.control_period + ay_opt[-1]*self.control_period**2/2. + jy_opt[-1]*self.control_period**3/6.

            x_opt.append(px)
            y_opt.append(py)
            vx_opt.append(vx)
            vy_opt.append(vy)
            ax_opt.append(ax)
            ay_opt.append(ay)

            # print((x_opt[-1],y_opt[-1]),(vx_opt[-1],vy_opt[-1]),(ax_opt[-1],ay_opt[-1]),(jx_opt[-1],jy_opt[-1]))
        # p_cost = 0.
        # v_cost = 0.
        # a_cost = 0.
        # j_cost = 0.
        # for i in range(self.N):
        #     # minimize track error
            
        #     p_cost += (ref_traj_info[i][0][0] - x_opt[i])**2
        #     p_cost += (ref_traj_info[i][0][1] - y_opt[i])**2
        #     v_cost += (ref_traj_info[i][1][0] - vx_opt[i])**2
        #     v_cost += (ref_traj_info[i][1][1] - vy_opt[i])**2
        #     a_cost += (ref_traj_info[i][2][0] - ax_opt[i])**2
        #     a_cost += (ref_traj_info[i][2][1] - ay_opt[i])**2
        #     # minimize input
        #     j_cost += jx_opt[i]**2
        #     j_cost += jy_opt[i]**2

        return True,x_opt,y_opt,vx_opt,vy_opt,ax_opt,ay_opt,jx_opt,jy_opt

import numpy as np
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix
np.set_printoptions(threshold=np.inf)

class SO2MPC_dense:

    """
    Linear model predictive control for a system with linear dynamics and
    linear constraints. This class is fully documented at:

        https://scaron.info/doc/pymanoid/walking-pattern-generation.html#pymanoid.mpc.LinearPredictiveControl
    """
    # x_{i+1} = A x_i + B u_i
    # x_init = x0
    # C_i x_i + Di u_i <= e_i
    
    def __init__(
        self,
        A,
        B,
        state_limit,
        nb_timesteps: int,
        ref_p_index = None,
        smooth_j_weight:float = 1.,
        smooth_a_weight:float = 1.,
        end_p_weight:float = 1.,
        end_v_weight:float = 1.,
        end_a_weight:float = 1.,
        ref_p_weight:float = 1.
    ):
        u_dim = B.shape[1]
        x_dim = A.shape[1]
        self.A = A
        self.B = B
        self.G = None
        self.P = None
        self.U = None
        self.U_dim = u_dim * nb_timesteps
        self.h = None
        self.nb_timesteps = nb_timesteps
        self.q = None
        self.u_dim = u_dim
        self.x_dim = x_dim

        self.ref_p_index = ref_p_index
        self.smooth_j_weight = smooth_j_weight
        self.smooth_a_weight = smooth_a_weight
        self.end_p_weight = end_p_weight
        self.end_v_weight = end_v_weight
        self.ref_p_weight = ref_p_weight
        self.end_a_weight = end_a_weight

        self.vaAN = None
        self.vaAN_1B = None
        self.pAN = None
        self.pAN_1B = None
        self.select1p_pAN = None
        self.select1p_pAN_1B = None
        self.n_state_limit = 4
        self.state_limit = state_limit # v,a,j
        self.lb = -self.state_limit[2]*np.ones(self.U_dim)
        self.ub = self.state_limit[2]*np.ones(self.U_dim)
        self.va_select = np.array([[0.,0.,1.,0.,0.,0.],[0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,1.]])
        self.p_select = np.array([[1.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.],])
        self.v_select = np.array([[0.,0.,1.,0.,0.,0.],[0.,0.,0.,1.,0.,0.],])
        self.a_select = np.array([[0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,1.],])

        self.select_one_p = np.empty((nb_timesteps,2,2*self.nb_timesteps))
        s = np.zeros((2,2*self.nb_timesteps))
        for i in range(self.nb_timesteps):
            s[0][2*i] = 1
            s[1][2*i+1] = 1
            self.select_one_p[i] = s
            s[0][2*i] = 0
            s[1][2*i+1] = 0

        dynamic = []
        for i in range(self.nb_timesteps):
            dynamic.extend([self.state_limit[0],self.state_limit[0],self.state_limit[1],self.state_limit[1]])
        self.h_dynamic_limt = np.array(dynamic)

        self.build()

    def build(self):
        # min_u uTPu + 2qTu
        # Gu <= h
        # Au = b
        # lb<=u<=ub

        self.AN_1B = np.zeros((self.x_dim*self.nb_timesteps, self.U_dim))
        self.AN = np.zeros((self.x_dim*self.nb_timesteps, self.x_dim))
        # va select
        va = np.zeros((self.n_state_limit*self.nb_timesteps, self.x_dim*self.nb_timesteps))
        # p select all
        p = np.zeros((2*self.nb_timesteps, self.x_dim*self.nb_timesteps))
        # v select all
        v = np.zeros((2*self.nb_timesteps, self.x_dim*self.nb_timesteps))
        # a select all
        a = np.zeros((2*self.nb_timesteps, self.x_dim*self.nb_timesteps))
        
        Ak = self.A.copy()
        Ak_1_B = self.B.copy()


        for k in range(self.nb_timesteps):
        
            self.AN[k*self.x_dim:(k+1)*self.x_dim,:] = Ak
            va[k*self.n_state_limit:(k+1)*self.n_state_limit,k*self.x_dim:(k+1)*self.x_dim] = self.va_select

            p[k*2:(k+1)*2,k*self.x_dim:(k+1)*self.x_dim] = self.p_select
            v[k*2:(k+1)*2,k*self.x_dim:(k+1)*self.x_dim] = self.v_select
            a[k*2:(k+1)*2,k*self.x_dim:(k+1)*self.x_dim] = self.a_select

            for j in range(self.nb_timesteps - k):
                self.AN_1B[k*self.x_dim + j*self.x_dim:k*self.x_dim + (j+1)*self.x_dim,j*self.u_dim:(j+1)*self.u_dim] = Ak_1_B
            
            Ak_1_B = self.A @ Ak_1_B
            Ak = self.A @ Ak

        self.vaAN = va @ self.AN
        self.vaAN_1B = va @ self.AN_1B

        self.pAN = p @ self.AN
        self.pAN_1B = p @ self.AN_1B
        self.select1p_pAN = np.empty((self.nb_timesteps,self.select_one_p[0].shape[0],self.pAN.shape[1]))
        self.select1p_pAN_1B = np.empty((self.nb_timesteps,self.select_one_p[0].shape[0],self.pAN_1B.shape[1]))
        for i in range(self.nb_timesteps):
            self.select1p_pAN[i] = self.select_one_p[i] @ self.pAN
            self.select1p_pAN_1B[i] = self.select_one_p[i] @ self.pAN_1B
        
        self.pAN_1BT = self.pAN_1B.T
        vAN_1B = v @ self.AN_1B
        self.vAN_1BT = vAN_1B.T
        aAN_1B = a @ self.AN_1B
        self.aAN_1BT = aAN_1B.T
        
        # COST形式推导
        # 以加速度二次型为例
        # 非ref计算
        # P_a为加速度权重项 对角阵 (2 timestep)X(2 timestep)
        # [ax1,ay1,...,] P_a [ax1,ay1,...,]T = (a (ANx0+AN_1Bu))T P_a a (ANx0+AN_1Bu)
        #  = (ANx0+AN_1Bu)T aTP_aa (ANx0+AN_1Bu)
        #  = (ANx0T + uTAN_1BT) (aTP_aa ANx0 + aTP_aa AN_1Bu )
        #  = ANx0T aTP_aa ANx0 + ANx0T aTP_aa AN_1B u
        #    + uT AN_1BT aTP_aa ANx0 + uT AN_1BT aTP_aa AN_1B u
        # ANx0T aTP_aa AN_1B u和uT AN_1BT aTP_aa ANx0互为转置都是常数，则上式可改为,其中第一项为常数优化时不考虑
        # = uT AN_1BT aTP_aa AN_1B u + 2ANx0T aTP_aa AN_1B u 
        # P = AN_1BT aT P_a a AN_1B ,q = 2ANx0T aTP_aa AN_1B

        # 代价函数 二次型
        min_j_P = self.smooth_j_weight * np.eye(self.U_dim)
        weight_a = self.smooth_a_weight * np.eye(2*self.nb_timesteps)
        # 最小化终态加速度
        weight_a[-2,-2] = self.end_a_weight # ax end
        weight_a[-1,-1] = self.end_a_weight # ay end
        min_a_P = self.aAN_1BT @ weight_a @ aAN_1B
        self.sub_min_a_q = a.T @ weight_a @ aAN_1B # 2aTP_aa AN_1B

        # all ref计算
        # [ax1-axref1,ay1-ayref1,...,] P_a [ax1-axref1,ay1-ayref1,...,]T 
        # a_ref列向量
        #  = [ax1,ay1,..,...]P_a[ax1,ay1,..,...]T - [axref1,..]P_a[ax1,ay1,..]T
        #    -[ax1,ay1,..,...]P_a[axref1,..]T + 常数项
        #  = (ANx0+AN_1Bu)T aTP_aa (ANx0+AN_1Bu) - a_refT P_aa (ANx0+AN_1Bu)
        #     - (ANx0+AN_1Bu)T aTP_a a_ref
        # 其中第一项为非ref情况下的结果= uT AN_1BT aTP_aa AN_1B u + 2ANx0T aTP_aa AN_1B u 
        # 第二项 a_refT P_aa (ANx0+AN_1Bu)
        # = a_refT P_aa ANx0(常数) + a_refT P_aa AN_1Bu
        # 第三项 (ANx0+AN_1Bu)T aTP_a a_ref
        # = (ANx0T+uTAN_1BT) aTP_a a_ref
        # = ANx0T aTP_a a_ref(常数) + uT AN_1BT aTP_a a_ref
        # a_refT P_aa AN_1B u与uT AN_1BT aTP_a a_ref互为转置，都是常数相等
        # 最终
        # P = AN_1BT aTP_aa AN_1B 
        # q = 2ANx0T aTP_aa AN_1B - 2a_refT P_aa AN_1B
        #   = 2(ANx0T aT- a_refT)P_aa AN_1B

        # 指定ref计算
        # [ax3-axref3,ay7-ayref7,...,] P_a_sub [ax3-axref3,ay7-ayref7,...,]T 
        # = (a_sub - aref_sub)T P_a_sub (a_sub - aref_sub)
        # a (ANx0+AN_1Bu) = [ax1,ay1,ax2,ay2,...]T
        # aref_sub = [axref3,ayref3,axref7,ayref7,...]T
        # a_sub = sub a (ANx0+AN_1Bu)
        # 设有n个指定ref点要计算则 sub--(2n,2timestep)
        # [[0,0,0,0,1(第5个 表示选定ax3),0,0,..],
        #  [0,0,0,0,0,1(第6个 表示选定ay3),0,0,..],
        #  ...,
        #  [],
        #  []]  2n行代表n个被选定的参考点
        # = a_subT P_a_sub a_sub - aref_subT P_a_sub a_sub - a_subT P_a_sub aref_sub + 常数
        # 第一项 
        # (a (ANx0+AN_1Bu))T subT P_a_sub sub a (ANx0+AN_1Bu)
        # = (ANx0T+uT AN_1BT) aT subT P_a_sub sub a (ANx0+AN_1Bu)
        # = (ANx0T+uT AN_1BT) ? (ANx0+AN_1Bu)
        # = ANx0T?ANx0(常数) + uT AN_1BT ? ANx0 + ANx0T?AN_1Bu + uT AN_1BT?AN_1Bu
        # = uT AN_1BT?AN_1B u + 2ANx0T?AN_1B u
        # = uT AN_1BT aT subT P_a_sub sub a AN_1B u + 2ANx0T aT subT P_a_sub sub a AN_1B u
        # 第二项 第三项互为转置 都是常数
        # aref_subT P_a_sub a_sub
        # = aref_subT P_a_sub sub a (ANx0+AN_1Bu)
        # = ?(ANx0+AN_1Bu)
        # = ?ANx0(常数) + ?AN_1Bu
        # = aref_subT P_a_sub sub a AN_1B u
        # 最终
        # P = AN_1BT aT subT P_a_sub sub a AN_1B
        # q = 2ANx0T aT subT P_a_sub sub a AN_1B - 2aref_subT P_a_sub sub a AN_1B
        #   = 2(ANx0T aT subT - aref_subT)P_a_sub sub a AN_1B

        # 最小化终态位置误差
        # sub = np.zeros((2*1,2*self.nb_timesteps))
        # sub[0][-2] = 1. # end px
        # sub[1][-1] = 1. # end py
        sub = self.select_one_p[-1]
        P_p_sub  = self.end_p_weight * np.eye(2)   
        sub_p_AN_1B = sub @ self.pAN_1B
        min_track_end_p_P = sub_p_AN_1B.T @ P_p_sub @ sub_p_AN_1B
        # min_track_end_p_P = sub_p_AN_1B.T.dot(sub_p_AN_1B)
        # min_track_end_p_P = sub_p_AN_1B.T @ sub_p_AN_1B
        self.sub_min_track_end_p_q = P_p_sub @ sub_p_AN_1B
        self.endpTsubT = (sub @ p).T

        # 最小化终态速度
        sub = np.zeros((2,2*self.nb_timesteps))
        sub[0][-2] = 1. # end vx
        sub[1][-1] = 1. # end vy
        P_v_sub  = self.end_v_weight * np.eye(2)
        sub_v_AN_1B = sub.dot(v.dot(self.AN_1B))
        min_end_v_P = sub_v_AN_1B.T.dot(P_v_sub.dot(sub_v_AN_1B))
        self.sub_min_end_v_q = P_v_sub.dot(sub_v_AN_1B)
        self.endvTsubT = sub.dot(v).T

        # 最小化指定位置误差
        self.ref_p_index = [6,13,20]
        sub = np.zeros((2*len(self.ref_p_index),2*self.nb_timesteps))
        for k,index in enumerate(self.ref_p_index):
            sub[2*k][2*index-2] = 1.
            sub[2*k+1][2*index-1] = 1.
        P_p_sub  = self.ref_p_weight * np.eye(2*len(self.ref_p_index))
        sub_p_AN_1B = sub.dot(p.dot(self.AN_1B))
        min_track_ref_p_P = sub_p_AN_1B.T.dot(P_p_sub.dot(sub_p_AN_1B))
        self.sub_min_track_ref_p_q = P_p_sub.dot(sub_p_AN_1B)
        self.refpTsubT = sub.dot(p).T

        # self.P = min_j_P + min_a_P + min_track_end_p_P + min_end_v_P
        self.P = min_track_end_p_P # + min_j_P

        self.P_csc = csc_matrix(self.P)

    # x_init = np.array([px],[py],[vx],[vy],[ax],[ay]) 列向量
    def solve(self,x_init,goal_p,ref_p = None,convex_vertex = None):
        # [p v a ,...,pn vn an]T = AN x0 + AN_1B u ,这里的被优化变量x为控制量u
        # [v a,...,vn an]T = va(AN x0 + AN_1B x) = vaANx0 + vaAN_1Bu
        # va select = [ [0 0 1 0 0 0],[0 0 0 1 0 0],[0 0 0 0 1 0],[0 0 0 0 0 1] ]
        # vaAN_1Bx <= [v a,...,vn an]T - vaANx0
        # v a convex(<0)
        G_list = []
        h_list = []
        self.x_init = x_init
        ANx0T = (self.AN @ x_init).T
        vaANx0 = self.vaAN @ x_init
        # v,a limt
        G_list.append(self.vaAN_1B)
        h_list.append(self.h_dynamic_limt-vaANx0)
        G_list.append(-self.vaAN_1B)
        h_list.append(self.h_dynamic_limt+vaANx0)
        # [px py,...,pxn pyn]T = p(AN x0 + AN_1B x) = pANx0 + pAN_1Bu
        # p select = [ [1 0 0 0 0 0],[0 1 0 0 0 0] ]
        # [pxi pyi]T = select1p[i]*p(AN x0 + AN_1B x) = select1p[i]pANx0 + select1p[i]pAN_1Bu
  

        # convex_vertex = [(x1,y1),(x2,y2),....]
        # [px1 py1 ...]T = p(ANx0+AN_1Bu)
        # [px1 py1]T = self.select_one_p[1] p(ANx0+AN_1Bu)
        # 单个点约束在凸多边形内部
        # [[y1-y2,x2-x1],..,[yi-yi+1,xi+1-xi],...] [px1 py1]T <= [[y1x2-x1y2],...,[yixi+1-xiyi+1],...]
        # [[y1-y2,x2-x1],..,[yi-yi+1,xi+1-xi],...] select_one_pj p(ANx0+AN_1Bu) <= [[y1x2-x1y2],...,[yixi+1-xiyi+1],...]
        # [[y1-y2,x2-x1],..,[yi-yi+1,xi+1-xi],...] select_one_pj p ANx0 + 
        # [[y1-y2,x2-x1],..,[yi-yi+1,xi+1-xi],...] select_one_pj p AN_1Bu <= [[y1x2-x1y2],...,[yixi+1-xiyi+1],...]
   
        # [[y1-y2,x2-x1],..,[yi-yi+1,xi+1-xi],...] select_one_pj p AN_1B u <= 
        # [[y1x2-x1y2],...,[yixi+1-xiyi+1],...] - [[y1-y2,x2-x1],..,[yi-yi+1,xi+1-xi],...] select_one_pj p ANx0
        if convex_vertex is not None:
            g_sub = np.zeros((len(convex_vertex), 2))
            h_sub = np.zeros((len(convex_vertex)))
            for j in range(len(convex_vertex)-1):
                g_sub[j][0] = convex_vertex[j][1] - convex_vertex[j+1][1]
                g_sub[j][1] = convex_vertex[j+1][0] - convex_vertex[j][0]
                h_sub[j] = convex_vertex[j][1]*convex_vertex[j+1][0] - convex_vertex[j][0]*convex_vertex[j+1][1]
            g_sub[-1][0] = convex_vertex[-1][1] - convex_vertex[0][1]
            g_sub[-1][1] = convex_vertex[0][0] - convex_vertex[-1][0]
            h_sub[-1] = convex_vertex[-1][1]*convex_vertex[0][0] - convex_vertex[-1][0]*convex_vertex[0][1]
            
            for i in range(self.nb_timesteps):
                g_sub_select1ppANx0 = g_sub.dot(self.select1p_pAN[i].dot(x_init))
                g_sub_select1ppAN_1B = g_sub.dot(self.select1p_pAN_1B[i])
                G_list.append(g_sub_select1ppAN_1B)
                h_list.append(h_sub-g_sub_select1ppANx0)

        # 代价函数
        
        # 最小化a q = 2ANx0T aTP_aa AN_1B    sub_min_a_q--2aTP_aa AN_1B
        min_a_q = ANx0T.dot(self.sub_min_a_q)
        
        # 最小化end p误差 q = 2(ANx0T pT subT - pref_subT)P_p_sub sub p AN_1B
        # sub_min_track_end_p_q -- 2P_p_sub sub p AN_1B
        pref_subT = np.array([goal_p[0],goal_p[1]])
        sub_p_end_q = ANx0T @ self.endpTsubT - pref_subT
        min_track_end_p_q = sub_p_end_q @ self.sub_min_track_end_p_q

        # 最小化end v
        endv_subT = np.array([0.,0.])
        sub_v_end_q = ANx0T.dot(self.endpTsubT) - endv_subT
        min_end_v_q = sub_v_end_q.dot(self.sub_min_end_v_q)

        # self.q =  min_a_q + min_track_end_p_q + min_end_v_q
        self.q = min_track_end_p_q

        if ref_p is not None:
            # 最小化部分位置跟踪误差

            refp_subT = []
            for p in ref_p:
                refp_subT.append(p[0])
                refp_subT.append(p[1])
            refp_subT = np.array(refp_subT)
            sub_ref_p_q = ANx0T.dot(self.refpTsubT) - refp_subT
            min_track_ref_p_q = sub_ref_p_q.dot(self.sub_min_track_ref_p_q)
            self.q = self.q + min_track_ref_p_q

        self.G = np.vstack(G_list)
        self.h = np.hstack(h_list)
        # self.G_csc = csc_matrix(self.G)

        # P = self.P_csc if True else self.P
        # G = self.G_csc if True else self.G
        # try:
        #     self.U = solve_qp(self.P, self.q, self.G, self.h,lb=self.lb,ub=self.ub, solver="quadprog",verbose = False)
        # except:
        #     return False
        # self.U = solve_qp(self.P, self.q.T, self.G, self.h,lb=self.lb,ub=self.ub, solver="quadprog",verbose = False)
        self.U = solve_qp(self.P, self.q, self.G, self.h,lb=self.lb,ub=self.ub, solver="osqp",verbose = False)
        
        if self.U is None:
            return False
        return True

    @property
    def states(self):
        assert self.U is not None, "you need to solve() the MPC problem first"
        # X = np.zeros((self.nb_timesteps + 1, self.x_dim))
        p = []
        v = []
        a = []
        j = []

        X = self.AN.dot(self.x_init) + self.AN_1B.dot(self.U)
        for k in range(self.nb_timesteps):
            j.append((self.U[2*k],self.U[2*k+1]))
            p.append((X[6*k],X[6*k+1]))
            v.append((X[6*k+2],X[6*k+3]))
            a.append((X[6*k+4],X[6*k+5]))

        return p,v,a,j



class SO2MPC_dense2:

    # min_x 0.5xTPx + qTx
    # 将自己的代价函数转为此形态

    def __init__(
        self,
        A,
        B,
        state_limit,
        nb_timesteps: int,
        smooth_j_weight:float = 1.,
        end_weight:float = 1.,
        ref_v_weight:float = 1.,
        ref_a_weight:float = 1.,
        ref_p_weight:float = 1.
    ):

        u_dim = B.shape[1]  # 2
        x_dim = A.shape[1]  # 6
        self.A = A
        self.B = B
        self.G = None
        self.P = None
        self.U = None
        self.U_dim = u_dim * nb_timesteps
        self.h = None
        self.nb_timesteps = nb_timesteps
        self.q = None
        self.u_dim = u_dim
        self.x_dim = x_dim

        self.x_init = None
        self.ref_p_index = None
        self.smooth_j_weight = smooth_j_weight
        self.end_weight = end_weight
        self.ref_v_weight = ref_v_weight
        self.ref_p_weight = ref_p_weight
        self.ref_a_weight = ref_a_weight
        self.ref = None

        self.AN = None
        self.vaAN = None
        self.vaAN_1B = None
        self.pAN = None
        self.pAN_1B = None
        self.select1p_pAN = None
        self.select1p_pAN_1B = None
        self.n_state_limit = 4
        self.state_limit = state_limit # v,a,j
        self.lb = -self.state_limit[2]*np.ones(self.U_dim)
        self.ub = self.state_limit[2]*np.ones(self.U_dim)
        self.va_select = np.array([[0.,0.,1.,0.,0.,0.],[0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,1.]])
        self.p_select = np.array([[1.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.],])
        self.v_select = np.array([[0.,0.,1.,0.,0.,0.],[0.,0.,0.,1.,0.,0.],])
        self.a_select = np.array([[0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,1.],])


        dynamic = []
        for i in range(self.nb_timesteps):
            dynamic.extend([self.state_limit[0],self.state_limit[0],self.state_limit[1],self.state_limit[1]])
        self.h_dynamic_limt = np.array(dynamic).T
        self.build()

    def build(self):
        self.AN = np.zeros((self.nb_timesteps*self.x_dim,self.x_dim))
        self.AN_1B = np.zeros((self.nb_timesteps*self.x_dim,self.U_dim))

        A_k = self.A.copy()
        Ak_1B = self.B.copy()
        
        pi = 0
        vi = 0
        ai = 0

        self.ref_p_index = [1,self.nb_timesteps]
        self.ref_v_index = [self.nb_timesteps]
        self.ref_a_index = [i for i in range(1,self.nb_timesteps+1)]
        self.p_track_weight = np.zeros((self.nb_timesteps*self.x_dim,self.nb_timesteps*self.x_dim))# X = [px1,py1,vx1,vy1,ax1,ay1,px2,py2,vx2,vy2,ax2,ay2,...] X pw XT
        self.v_track_weight = np.zeros((self.nb_timesteps*self.x_dim,self.nb_timesteps*self.x_dim))# X = [px1,py1,vx1,vy1,ax1,ay1,px2,py2,vx2,vy2,ax2,ay2,...] X pw XT
        self.a_track_weight = np.zeros((self.nb_timesteps*self.x_dim,self.nb_timesteps*self.x_dim))# X = [px1,py1,vx1,vy1,ax1,ay1,px2,py2,vx2,vy2,ax2,ay2,...] X pw XT
        self.va = np.zeros((self.nb_timesteps*4,self.nb_timesteps*self.x_dim))
        self.one_p_select = np.zeros((self.nb_timesteps,2,self.nb_timesteps*self.x_dim))


        for k in range(self.nb_timesteps):
            self.AN[k*self.x_dim:(k+1)*self.x_dim,:] = A_k
            self.va[k*4:(k+1)*4,k*self.x_dim:(k+1)*self.x_dim] = self.va_select
            self.one_p_select[k][0][k*self.x_dim] = 1.
            self.one_p_select[k][1][k*self.x_dim + 1] = 1.

            if pi < len(self.ref_p_index) and k == self.ref_p_index[pi] - 1:
                # if k == self.nb_timesteps - 1:
                #     w = self.end_weight
                # else:
                #     w = 1.
                w = self.end_weight
                # 权重阵中为0，则cost计算中不会考虑此项
                self.p_track_weight[k*self.x_dim,k*self.x_dim] = w*self.ref_p_weight
                self.p_track_weight[k*self.x_dim+1,k*self.x_dim+1] = w*self.ref_p_weight

                pi += 1

            if vi<len(self.ref_v_index) and k == self.ref_v_index[vi] - 1:
                # if k == self.nb_timesteps - 1:
                #     w = self.end_weight
                # else:
                #     w = 1.
                w = 1.
                # 权重阵中为0，则cost计算中不会考虑此项
                self.v_track_weight[k*self.x_dim+2,k*self.x_dim+2] = w*self.ref_v_weight
                self.v_track_weight[k*self.x_dim+2+1,k*self.x_dim+2+1] = w*self.ref_v_weight

                vi += 1

            if ai<len(self.ref_a_index) and k == self.ref_a_index[ai] - 1:
                # if k == self.nb_timesteps - 1:
                #     w = self.end_weight
                # else:
                #     w = 1.
                w = 1.
                # 权重阵中为0，则cost计算中不会考虑此项
                self.a_track_weight[k*self.x_dim+4,k*self.x_dim+4] = w*self.ref_a_weight
                self.a_track_weight[k*self.x_dim+4+1,k*self.x_dim+4+1] = w*self.ref_a_weight

                ai += 1

            for j in range(self.nb_timesteps - k):
                self.AN_1B[k*self.x_dim + j*self.x_dim:k*self.x_dim + (j+1)*self.x_dim,j*self.u_dim:(j+1)*self.u_dim] = Ak_1B
            
            A_k = self.A @ A_k 
            Ak_1B = self.A @ Ak_1B

        self.x_weight = self.p_track_weight + self.v_track_weight + self.a_track_weight # XT w X
   
        self.select1p_AN = np.empty((self.nb_timesteps,self.one_p_select[0].shape[0],self.AN.shape[1]))
        self.select1p_AN_1B = np.empty((self.nb_timesteps,self.one_p_select[0].shape[0],self.AN_1B.shape[1]))
        for i in range(self.nb_timesteps):
            self.select1p_AN[i] = self.one_p_select[i] @ self.AN
            self.select1p_AN_1B[i] = self.one_p_select[i] @ self.AN_1B

        # ([px1-,py1,vx1,vy1,ax1,ay1,px2,py2,vx2,vy2,ax2,ay2,...,]-refT) x_weight ([px1,py1,vx1,vy1,ax1,ay1,px2,py2,vx2,vy2,ax2,ay2,...,]-ref)T
        #   (X - ref)T x_weight (X - ref)
        # = XT x_weight X - refT x_weight X - XT x_weight ref + refT x_weight ref
        # refT x_weight ref为常数  refT x_weight X 和 XT x_weight ref 互为转置 都是标量
        # = XT x_weight X - 2refT x_weight X
        #  X = AN X0 + AN_1B U代入
        # = (X0TANT + UT AN_1BT)x_weight(AN X0 + AN_1B U) - 2refT x_weight(AN X0 + AN_1B U)
        # = X0TANT x_weight AN X0(常数) + UT AN_1BT x_weight AN X0 + X0TANT x_weight AN_1B U +
        #   UT AN_1BT x_weight AN_1B U -2refT x_weight AN X0 (常数) - 2refT x_weight AN_1B U
        # = UT AN_1BT x_weight AN_1B U + 2(X0TANT - refT)x_weight AN_1B U

        min_j_P = self.smooth_j_weight * np.eye(self.U_dim) # 最小化控制量
        self.P = self.AN_1B.T @ self.x_weight @ self.AN_1B + min_j_P
        self.va_g = self.va @ self.AN_1B
        self.ref_sub_q = self.x_weight @ self.AN_1B
        
        # # ref_P = 0.5*(ref_P + ref_P.transpose()) + np.eye(self.U_dim)*norm
        norm = np.linalg.norm(self.P,2) * 1e-6
        self.P = self.P + np.eye(self.U_dim)*norm

        self.i = 0



    def solve(self,x_init,X_ref,convex_vertex = None):
        self.ref = X_ref
        G_list = []
        h_list = []
        self.x_init = x_init
        ANX0 = self.AN @ x_init
        vaANX0 = self.va @ ANX0
        # [vx1,vy1,ax1,ay1,...] = va X  <= [v limt,v limt,a limit,a limit,...]T
        # va(AN X0 + AN_1B U) <= va_limit
        # va AN_1B U <= va_limit - vaAN X0

        G_list.append(self.va_g)
        h_list.append(self.h_dynamic_limt-vaANX0)
        G_list.append(-self.va_g)
        h_list.append(self.h_dynamic_limt+vaANX0)

        # [px py,...,pxn pyn]T = p(AN x0 + AN_1B x) = pANx0 + pAN_1Bu
        # p select = [ [1 0 0 0 0 0],[0 1 0 0 0 0] ]
        # [pxi pyi]T = select1p[i]*p(AN x0 + AN_1B x) = select1p[i]pANx0 + select1p[i]pAN_1B
        # convex_vertex = [(x1,y1),(x2,y2),....]
        # [px1 py1 ...]T = p(ANx0+AN_1Bu)
        # [px1 py1]T = self.select_one_p[1] p(ANx0+AN_1Bu)
        # 单个点约束在凸多边形内部
        # [[y1-y2,x2-x1],..,[yi-yi+1,xi+1-xi],...] [px1 py1]T <= [[y1x2-x1y2],...,[yixi+1-xiyi+1],...]
        # [[y1-y2,x2-x1],..,[yi-yi+1,xi+1-xi],...] select_one_pj(ANx0+AN_1Bu) <= [[y1x2-x1y2],...,[yixi+1-xiyi+1],...]
        # [[y1-y2,x2-x1],..,[yi-yi+1,xi+1-xi],...] select_one_pj ANx0 + 
        # [[y1-y2,x2-x1],..,[yi-yi+1,xi+1-xi],...] select_one_pj AN_1Bu <= [[y1x2-x1y2],...,[yixi+1-xiyi+1],...]
   
        # [[y1-y2,x2-x1],..,[yi-yi+1,xi+1-xi],...] select_one_pj p AN_1B u <= 
        # [[y1x2-x1y2],...,[yixi+1-xiyi+1],...] - [[y1-y2,x2-x1],..,[yi-yi+1,xi+1-xi],...] select_one_pj p ANx0
        if convex_vertex is not None:
            g_sub = np.zeros((len(convex_vertex), 2))
            h_sub = np.zeros((len(convex_vertex)))
            for j in range(len(convex_vertex)-1):
                g_sub[j][0] = convex_vertex[j][1] - convex_vertex[j+1][1]
                g_sub[j][1] = convex_vertex[j+1][0] - convex_vertex[j][0]
                h_sub[j] = convex_vertex[j][1]*convex_vertex[j+1][0] - convex_vertex[j][0]*convex_vertex[j+1][1]
            g_sub[-1][0] = convex_vertex[-1][1] - convex_vertex[0][1]
            g_sub[-1][1] = convex_vertex[0][0] - convex_vertex[-1][0]
            h_sub[-1] = convex_vertex[-1][1]*convex_vertex[0][0] - convex_vertex[-1][0]*convex_vertex[0][1]
            
            for i in range(self.nb_timesteps):
                g_sub_select1pANx0 = g_sub @ self.select1p_AN[i] @ x_init
                g_sub_select1pAN_1B = g_sub @ self.select1p_AN_1B[i]
                G_list.append(g_sub_select1pAN_1B)
                h_list.append(h_sub-g_sub_select1pANx0)
 
        ref_q = (ANX0 - X_ref).T @ self.ref_sub_q
        self.G = np.vstack(G_list)
        self.h = np.hstack(h_list)

        # self.U = solve_qp(self.P, ref_q, self.G,self.h,lb=self.lb,ub=self.ub, solver="quadprog",verbose = False)
        # self.U = solve_qp(self.P, ref_q, self.G,self.h,lb=self.lb,ub=self.ub, solver="osqp",verbose = False)
        # 'cvxopt', 'ecos', 'highs', 'osqp', 'proxqp', 'quadprog', 'scs'
        self.U = None
        try:
            self.U = solve_qp(self.P, ref_q, self.G,self.h,lb=self.lb,ub=self.ub, solver="quadprog",verbose = False)
        except:
            self.U = None
            return False
        
        if self.U is None:
            # print(x_init)
            # print(convex_vertex)
            return False

        return True

    @property
    def states(self):
        assert self.U is not None, "you need to solve() the MPC problem first"
        # X = np.zeros((self.nb_timesteps + 1, self.x_dim))
        p = []
        v = []
        a = []
        j = []
        cost = 0.
        X = self.AN.dot(self.x_init) + self.AN_1B.dot(self.U)
        for k in range(self.nb_timesteps):
            cost += self.smooth_j_weight * (self.U[2*k]**2 + self.U[2*k+1]**2)
            # 100为pva约束外的值 一定取不到 使用101代表不做约束的参考值
            if self.ref[6*k+4] < 100:
                cost += self.ref_a_weight * (X[6*k+4] - self.ref[6*k+4])**2 
            if self.ref[6*k+5] < 100:
                cost += self.ref_a_weight * (X[6*k+5] - self.ref[6*k+5])**2

            if self.ref[6*k+2] < 100:
                cost += self.ref_v_weight * (X[6*k+2] - self.ref[6*k+2])**2 
            if self.ref[6*k+3] < 100:
                cost += self.ref_v_weight * (X[6*k+3] - self.ref[6*k+3])**2

            if k == self.nb_timesteps - 1:
                w = self.end_weight
            else:
                w = 1.

            if self.ref[6*k] < 100:
                cost += w*self.ref_p_weight * (X[6*k] - self.ref[6*k])**2 
            if self.ref[6*k+1] < 100:
                cost += w*self.ref_p_weight * (X[6*k+1] - self.ref[6*k+1])**2

            j.append((self.U[2*k],self.U[2*k+1]))
            p.append((X[6*k],X[6*k+1]))
            v.append((X[6*k+2],X[6*k+3]))
            a.append((X[6*k+4],X[6*k+5]))

        return p,v,a,j,cost

class SO2MPC_dense3:

    # min_x 0.5xTPx + qTx
    # 将自己的代价函数转为此形态

    def __init__(
        self,
        A,
        B,
        state_limit,
        nb_timesteps: int,
        smooth_a_weight:float = 1.,
        end_weight:float = 1.,
        ref_v_weight:float = 1.,
        ref_p_weight:float = 1.
    ):

        u_dim = B.shape[1]
        x_dim = A.shape[1]
        self.A = A
        self.B = B
        self.G = None
        self.P = None
        self.U = None
        self.U_dim = u_dim * nb_timesteps
        self.h = None
        self.nb_timesteps = nb_timesteps
        self.q = None
        self.u_dim = u_dim
        self.x_dim = x_dim

        self.x_init = None
        self.ref_p_index = None
        self.end_weight = end_weight
        self.ref_v_weight = ref_v_weight
        self.ref_p_weight = ref_p_weight
        self.smooth_a_weight = smooth_a_weight
        self.ref = None

        self.AN = None
        self.vaAN = None
        self.vaAN_1B = None
        self.pAN = None
        self.pAN_1B = None
        self.select1p_pAN = None
        self.select1p_pAN_1B = None
        self.n_state_limit = 4
        self.state_limit = state_limit # v,a
        self.lb = -self.state_limit[1]*np.ones(self.U_dim)
        self.ub = self.state_limit[1]*np.ones(self.U_dim)
        self.p_select = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],])
        self.v_select = np.array([[0.,0.,1.,0.],[0.,0.,0.,1.],])


        dynamic = []
        for i in range(self.nb_timesteps):
            dynamic.extend([self.state_limit[0],self.state_limit[0]])
        self.h_dynamic_limt = np.array(dynamic).T
        self.build()

    def build(self):
        self.AN = np.zeros((self.nb_timesteps*self.x_dim,self.x_dim))
        self.AN_1B = np.zeros((self.nb_timesteps*self.x_dim,self.U_dim))

        A_k = self.A.copy()
        Ak_1B = self.B.copy()
        
        pi = 0
        vi = 0
        ai = 0

        self.ref_p_index = [self.nb_timesteps]
        self.ref_v_index = [self.nb_timesteps]
        self.p_track_weight = np.zeros((self.nb_timesteps*self.x_dim,self.nb_timesteps*self.x_dim))# X = [px1,py1,vx1,vy1,ax1,ay1,px2,py2,vx2,vy2,ax2,ay2,...] X pw XT
        self.v_track_weight = np.zeros((self.nb_timesteps*self.x_dim,self.nb_timesteps*self.x_dim))# X = [px1,py1,vx1,vy1,ax1,ay1,px2,py2,vx2,vy2,ax2,ay2,...] X pw XT
        self.a_track_weight = np.zeros((self.nb_timesteps*self.x_dim,self.nb_timesteps*self.x_dim))# X = [px1,py1,vx1,vy1,ax1,ay1,px2,py2,vx2,vy2,ax2,ay2,...] X pw XT
        self.one_p_select = np.zeros((self.nb_timesteps,2,self.nb_timesteps*self.x_dim))
        self.v = np.zeros((2*self.nb_timesteps,self.nb_timesteps*self.x_dim))

        for k in range(self.nb_timesteps):
            self.AN[k*self.x_dim:(k+1)*self.x_dim,:] = A_k
            self.one_p_select[k][0][k*self.x_dim] = 1.
            self.one_p_select[k][1][k*self.x_dim + 1] = 1.
            self.v[k*2:(k+1)*2,k*self.x_dim:(k+1)*self.x_dim] = self.v_select

            if pi<len(self.ref_p_index) and k == self.ref_p_index[pi] - 1:
                if k == self.nb_timesteps - 1:
                    w = self.end_weight
                else:
                    w = 1.
                # 权重阵中为0，则cost计算中不会考虑此项
                self.p_track_weight[k*self.x_dim,k*self.x_dim] = w*self.ref_p_weight
                self.p_track_weight[k*self.x_dim+1,k*self.x_dim+1] = w*self.ref_p_weight

                pi += 1

            if vi<len(self.ref_v_index) and k == self.ref_v_index[vi] - 1:
                # if k == self.nb_timesteps - 1:
                #     w = self.end_weight
                # else:
                #     w = 1.
                w = 1.
                # 权重阵中为0，则cost计算中不会考虑此项
                self.v_track_weight[k*self.x_dim+2,k*self.x_dim+2] = w*self.ref_v_weight
                self.v_track_weight[k*self.x_dim+2+1,k*self.x_dim+2+1] = w*self.ref_v_weight

                vi += 1

            for j in range(self.nb_timesteps - k):
                self.AN_1B[k*self.x_dim + j*self.x_dim:k*self.x_dim + (j+1)*self.x_dim,j*self.u_dim:(j+1)*self.u_dim] = Ak_1B
            
            A_k = self.A @ A_k 
            Ak_1B = self.A @ Ak_1B

        self.x_weight = self.p_track_weight + self.v_track_weight # XT w X
   
        self.select1p_AN = np.empty((self.nb_timesteps,self.one_p_select[0].shape[0],self.AN.shape[1]))
        self.select1p_AN_1B = np.empty((self.nb_timesteps,self.one_p_select[0].shape[0],self.AN_1B.shape[1]))
        for i in range(self.nb_timesteps):
            self.select1p_AN[i] = self.one_p_select[i] @ self.AN
            self.select1p_AN_1B[i] = self.one_p_select[i] @ self.AN_1B

        # ([px1-,py1,vx1,vy1,ax1,ay1,px2,py2,vx2,vy2,ax2,ay2,...,]-refT) x_weight ([px1,py1,vx1,vy1,ax1,ay1,px2,py2,vx2,vy2,ax2,ay2,...,]-ref)T
        #   (X - ref)T x_weight (X - ref)
        # = XT x_weight X - refT x_weight X - XT x_weight ref + refT x_weight ref
        # refT x_weight ref为常数  refT x_weight X 和 XT x_weight ref 互为转置 都是标量
        # = XT x_weight X - 2refT x_weight X
        #  X = AN X0 + AN_1B U代入
        # = (X0TANT + UT AN_1BT)x_weight(AN X0 + AN_1B U) - 2refT x_weight(AN X0 + AN_1B U)
        # = X0TANT x_weight AN X0(常数) + UT AN_1BT x_weight AN X0 + X0TANT x_weight AN_1B U +
        #   UT AN_1BT x_weight AN_1B U -2refT x_weight AN X0 (常数) - 2refT x_weight AN_1B U
        # = UT AN_1BT x_weight AN_1B U + 2(X0TANT - refT)x_weight AN_1B U

        min_a_P = self.smooth_a_weight * np.eye(self.U_dim) # 最小化控制量
        self.P = self.AN_1B.T @ self.x_weight @ self.AN_1B + min_a_P
        self.v_g = self.v @ self.AN_1B
        self.ref_sub_q = self.x_weight @ self.AN_1B
        
        # # ref_P = 0.5*(ref_P + ref_P.transpose()) + np.eye(self.U_dim)*norm
        norm = np.linalg.norm(self.P,2) * 1e-6
        self.P = self.P + np.eye(self.U_dim)*norm

        self.i = 0



    def solve(self,x_init,X_ref,convex_vertex = None):
        self.ref = X_ref
        G_list = []
        h_list = []
        self.x_init = x_init
        ANX0 = self.AN @ x_init
        vANX0 = self.v @ ANX0
        # [vx1,vy1,ax1,ay1,...] = va X  <= [v limt,v limt,a limit,a limit,...]T
        # va(AN X0 + AN_1B U) <= va_limit
        # va AN_1B U <= va_limit - vaAN X0

        G_list.append(self.v_g)
        h_list.append(self.h_dynamic_limt-vANX0)
        G_list.append(-self.v_g)
        h_list.append(self.h_dynamic_limt+vANX0)

        # [px py,...,pxn pyn]T = p(AN x0 + AN_1B x) = pANx0 + pAN_1Bu
        # p select = [ [1 0 0 0 0 0],[0 1 0 0 0 0] ]
        # [pxi pyi]T = select1p[i]*p(AN x0 + AN_1B x) = select1p[i]pANx0 + select1p[i]pAN_1B
        # convex_vertex = [(x1,y1),(x2,y2),....]
        # [px1 py1 ...]T = p(ANx0+AN_1Bu)
        # [px1 py1]T = self.select_one_p[1] p(ANx0+AN_1Bu)
        # 单个点约束在凸多边形内部
        # [[y1-y2,x2-x1],..,[yi-yi+1,xi+1-xi],...] [px1 py1]T <= [[y1x2-x1y2],...,[yixi+1-xiyi+1],...]
        # [[y1-y2,x2-x1],..,[yi-yi+1,xi+1-xi],...] select_one_pj(ANx0+AN_1Bu) <= [[y1x2-x1y2],...,[yixi+1-xiyi+1],...]
        # [[y1-y2,x2-x1],..,[yi-yi+1,xi+1-xi],...] select_one_pj ANx0 + 
        # [[y1-y2,x2-x1],..,[yi-yi+1,xi+1-xi],...] select_one_pj AN_1Bu <= [[y1x2-x1y2],...,[yixi+1-xiyi+1],...]
   
        # [[y1-y2,x2-x1],..,[yi-yi+1,xi+1-xi],...] select_one_pj p AN_1B u <= 
        # [[y1x2-x1y2],...,[yixi+1-xiyi+1],...] - [[y1-y2,x2-x1],..,[yi-yi+1,xi+1-xi],...] select_one_pj p ANx0
        if convex_vertex is not None:
            g_sub = np.zeros((len(convex_vertex), 2))
            h_sub = np.zeros((len(convex_vertex)))
            for j in range(len(convex_vertex)-1):
                g_sub[j][0] = convex_vertex[j][1] - convex_vertex[j+1][1]
                g_sub[j][1] = convex_vertex[j+1][0] - convex_vertex[j][0]
                h_sub[j] = convex_vertex[j][1]*convex_vertex[j+1][0] - convex_vertex[j][0]*convex_vertex[j+1][1]
            g_sub[-1][0] = convex_vertex[-1][1] - convex_vertex[0][1]
            g_sub[-1][1] = convex_vertex[0][0] - convex_vertex[-1][0]
            h_sub[-1] = convex_vertex[-1][1]*convex_vertex[0][0] - convex_vertex[-1][0]*convex_vertex[0][1]
            
            for i in range(self.nb_timesteps):
                g_sub_select1pANx0 = g_sub @ self.select1p_AN[i] @ x_init
                g_sub_select1pAN_1B = g_sub @ self.select1p_AN_1B[i]
                G_list.append(g_sub_select1pAN_1B)
                h_list.append(h_sub-g_sub_select1pANx0)
 
        ref_q = (ANX0 - X_ref).T @ self.ref_sub_q
        self.G = np.vstack(G_list)
        self.h = np.hstack(h_list)

        # self.U = solve_qp(self.P, ref_q, self.G,self.h,lb=self.lb,ub=self.ub, solver="quadprog",verbose = False)
        # self.U = solve_qp(self.P, ref_q, self.G,self.h,lb=self.lb,ub=self.ub, solver="osqp",verbose = False)
        # 'cvxopt', 'ecos', 'highs', 'osqp', 'proxqp', 'quadprog', 'scs'
        self.U = None
        try:
            self.U = solve_qp(self.P, ref_q, self.G,self.h,lb=self.lb,ub=self.ub, solver="quadprog",verbose = False)
        except:
            self.U = None
            return False
        
        if self.U is None:
            # print(x_init)
            # print(convex_vertex)
            return False

        return True

    @property
    def states(self):
        assert self.U is not None, "you need to solve() the MPC problem first"
        # X = np.zeros((self.nb_timesteps + 1, self.x_dim))
        p = []
        v = []
        a = []
        cost = 0.
        X = self.AN.dot(self.x_init) + self.AN_1B.dot(self.U)
        for k in range(self.nb_timesteps):
            cost += self.smooth_a_weight * (self.U[2*k]**2 + self.U[2*k+1]**2)
            # 100为pva约束外的值 一定取不到 使用101代表不做约束的参考值
            if self.ref[4*k+2] < 100:
                cost += self.ref_v_weight * (X[4*k+2] - self.ref[4*k+2])**2 
            if self.ref[4*k+3] < 100:
                cost += self.ref_v_weight * (X[4*k+3] - self.ref[4*k+3])**2

            if k == self.nb_timesteps - 1:
                w = self.end_weight
            else:
                w = 1.

            if self.ref[4*k] < 100:
                cost += w*self.ref_p_weight * (X[4*k] - self.ref[4*k])**2 
            if self.ref[4*k+1] < 100:
                cost += w*self.ref_p_weight * (X[4*k+1] - self.ref[4*k+1])**2

            a.append((self.U[2*k],self.U[2*k+1]))
            p.append((X[4*k],X[4*k+1]))
            v.append((X[4*k+2],X[4*k+3]))


        return p,v,a,cost

# if __name__ == "__main__":
#     dt = 0.1
#     A = np.array([[1.,0.,dt,0.],
#                  [0.,1.,0.,dt],
#                  [0.,0.,1.,0.],
#                  [0.,0.,0.,1.]])
#     B = np.array([[dt**2/2.,0.],
#                  [0.,dt**2/2.],
#                  [dt,0.],
#                  [0.,dt]])

#     convex = []
#     center = (1.543,12.247)
#     r = 10.
#     dnum = 60
#     drad = 2*np.pi/dnum
#     for i in range(dnum):
#         convex.append((center[0]+r*np.cos(i*drad),center[1]+r*np.sin(i*drad)))
#     # convex = [(1.753,12.4),(1.37,12.44),(1.748,12.056)]
#     convex.reverse()

#     # slover = SO2MPC_dense(        
#     #     A,
#     #     B,
#     #     state_limit = [3.,8.,40.],
#     #     nb_timesteps = 20,
#     #     ref_p_index = None,
#     #     smooth_j_weight = 1.,
#     #     smooth_a_weight = 1.,
#     #     # end_p_weight =  15000000.,
#     #     end_p_weight =  2500.,
#     #     end_v_weight = 1.,
#     #     end_a_weight= 1.,
#     #     ref_p_weight = 1.)

#     # end_weight 终态位置误差的放大权重
#     # ref_p_weight 终态位置误差权重，只有p最后一个点会乘end_weight
#     # ref_a_weight  期望最小化所有的nb_timesteps个状态加速度 
#     # smooth_j_weight 期望最小化所有的nb_timesteps个状态加加速度
#     # ref_v_weight  期望最小化终态速度

#     slover = SO2MPC_dense3(        
#         A,
#         B,
#         state_limit = [2.5,2.],
#         # state_limit = [30000.,800000.,400000.],
#         nb_timesteps = 20, # 20*0.1 = 2s 总轨迹时长2s
#         smooth_a_weight = 1.,
#         # end_weight =  15000000.,
#         # end_weight =  1600., # 终端位置160与其他cost各占一半
#         end_weight =  700., # 终端位置160与其他cost各占一半
#         ref_v_weight = 1.,
#         ref_p_weight = 1.)



#     start = np.array([0.7921,13.535,1.,0.0]).T
#     goal = (-3.56, 5.6)

#     ref_traj_info = [[(1.4728878354147859, 12.097789760714159), (0.7514952373161028, -0.8127539427640484), (0.414049363075255, -0.7997176772247541)], [(1.5109763104510379, 12.056173131552171), (0.7719619877344635, -0.8514901108093704), (0.4042386445812106, -0.7491909748594432)], [(1.550074919896432, 12.012684154750449), (0.791882125589424, -0.8876226564566909), (0.3922093549991603, -0.6956412448960212)], [(1.59015362114509, 11.96745661905442), (0.8111483964955724, -0.9210106978648025), (0.37810846108866913, -0.6394792688824075)], [(1.6311771925803087, 11.92063084382462), (0.8296608944054762, -0.9515338922698938), (0.3620829296093018, -0.5811158283665208)], [(1.673105600991466, 11.872352652082853), (0.8473270616096807, -0.9790924359855491), (0.3442797273206234, -0.5209617048962805)], [(1.7158943689909139, 11.822772343558304), (0.8640616887367094, -1.0036070644027482), (0.32484582098219883, -0.45942768001960543)], [(1.759494942430882, 11.772043667733678), (0.8797869147530641, -1.0250190519898674), (0.3039281773535928, -0.39692453528441474)], [(1.8038550578203738, 11.72032279689132), (0.894432226963225, -1.0432902122926795), (0.2816737631943705, -0.33386305223862756)], [(1.848919109742066, 11.667767299159362), (0.9079344610096505, -1.0584028979343525), (0.25822954526409664, -0.27065401243016285)], [(1.8946285182692075, 11.614535111557828), (0.9202378008727766, -1.07036000061545), (0.23374249032233624, -0.20770819740693963)], [(1.9409220963825193, 11.56078351304479), (0.9312937788710187, -1.0791849511139322), (0.2083595651286542, -0.1454363887168772)], [(1.9877364173870913, 11.506668097562487), (0.9410612756607695, -1.0849217192851555), (0.1822277364426154, -0.08424936790789439)], [(2.0350061823292833, 11.45234174708345), (0.9495065202364004, -1.0876348140618717), (0.1554939710237849, -0.02455791652791063)], [(2.082664588221107, 11.397953610429138), (0.9566031706784336, -1.087408706204184), (0.12831169548555418, 0.03327336387874386)], [(2.130643717259124, 11.343648232173585), (0.9623332023834441, -1.0843414785490524), (0.10086017585662158, 0.08906459178209222)], [(2.1788749793371567, 11.28956506253207), (0.9666877962939593, -1.078538476259803), (0.0733251380195121, 0.14268206565574637)], [(2.2272895755535034, 11.23583811733478), (0.9696674196466344, -1.0701117295760791), (0.0458923078567508, 0.19399208397331785)], [(2.27581896252565, 11.18259564377303), (0.9712818259722492, -1.0591799538138458), (0.01874741125086269, 0.2428609452084186)], [(2.324395316704973, 11.129959786145431), (0.9715500550957104, -1.0458685493653852), (-0.007923825915627201, 0.2891549478346602)]]
#     X_ref = []
#     for i in range(len(ref_traj_info)):
#         # cost中不关心的项一律给101
#         X_ref.append(101)
#         X_ref.append(101)
#         X_ref.append(101)
#         X_ref.append(101)
#         # X_ref.append(0.) # 最小化加速度
#         # X_ref.append(0.)
    
#     # X_ref  px1,py1,...,axN,ayN
#     # cost中不关心的项一律给-1
#     X_ref[-1] = 0. # 最小化终态速度
#     X_ref[-2] = 0.
#     X_ref[-3] = goal[1]
#     X_ref[-4] = goal[0]
#     X_ref = np.array(X_ref).T

#     # 360个约束 20步 耗时20ms
#     # 0约束 20步 耗时1ms左右
#     # 180 10ms
#     # 120 7ms
#     # X_ref[-5] = -1.53835324
#     # X_ref[-6] = 0.29218931
#     # convex = [(9.675002098083496, 2.6371435524197295e-07), (5.992757320404053, -4.222295761108398), (-2.858917236328125, -4.026829719543457), (-3.914973258972168, -2.622645616531372), (-4.080860137939453, 7.839923858642578), (9.675002098083496, 2.6371435524197295e-07)]
#     t = 0.
#     start_time = time.time()
#     # for i in range(1000):
#     slover.solve(start,X_ref,convex_vertex=convex)
#     # slover.solve(start,X_ref,convex_vertex=None)
#     p,v,a,cost = slover.states
#     t = t + time.time() - start_time
#     # start_time = time.time()
#     # print(t/1000)
#     print(time.time() - start_time)
#     # print(time.time() - start_time)
#     print(cost)
#     print("p:")
#     print(p)
#     print("v:")
#     print(v)
#     print("a:")
#     print(a)


#     plt.plot(start[0], start[1], 'ro')
#     plt.plot([xy[0] for xy in p], [xy[1] for xy in p], 'r')

#     convex.append(convex[0])
#     plt.plot([p[0] for p in convex], [p[1] for p in convex], 'g')

#     plt.plot(X_ref[-4], X_ref[-3], 'bo')
#     # plt.plot([s[0][0] for s in ref_traj_info], [s[0][1] for s in ref_traj_info], 'b')
#     plt.show()

if __name__ == "__main__":
    dt = 0.1
    A = np.array([[1.,0.,dt,0.,dt**2/2., 0.],
                 [0.,1.,0.,dt,  0.,  dt**2/2.],
                 [0.,0.,1.,0.,  dt,    0.],
                 [0.,0.,0.,1.,   0.,  dt],
                 [0.,0.,0.,0.,   1.,   0.],
                 [0.,0.,0.,0.,   0.,   1.]])
    B = np.array([[dt**3/6.,0.],
                 [0.,dt**3/6.],
                 [dt**2/2.,0.],
                 [0.,dt**2/2.],
                 [dt,0.],
                 [0.,dt]])

    convex = []
    center = (1.543,12.247)
    r = 10.
    dnum = 60
    drad = 2*np.pi/dnum
    for i in range(dnum):
        convex.append((center[0]+r*np.cos(i*drad),center[1]+r*np.sin(i*drad)))
    # convex = [(1.753,12.4),(1.37,12.44),(1.748,12.056)]
    
    # clockwise cross
    convex.reverse()

    # slover = SO2MPC_dense(        
    #     A,
    #     B,
    #     state_limit = [3.,8.,40.],
    #     nb_timesteps = 20,
    #     ref_p_index = None,
    #     smooth_j_weight = 1.,
    #     smooth_a_weight = 1.,
    #     # end_p_weight =  15000000.,
    #     end_p_weight =  2500.,
    #     end_v_weight = 1.,
    #     end_a_weight= 1.,
    #     ref_p_weight = 1.)

    # end_weight 终态位置误差的放大权重
    # ref_p_weight 终态位置误差权重，只有p最后一个点会乘end_weight
    # ref_a_weight  期望最小化所有的nb_timesteps个状态加速度 
    # smooth_j_weight 期望最小化所有的nb_timesteps个状态加加速度
    # ref_v_weight  期望最小化终态速度
    nb_timesteps = 30
    slover = SO2MPC_dense2(        
        A,
        B,
        state_limit = [2.5,4.,4.],
        # state_limit = [30000.,800000.,400000.],
        nb_timesteps = nb_timesteps, # 30*0.1 = 3s 总轨迹时长3s
        smooth_j_weight = 1.,
        # end_weight =  15000000.,
        # end_weight =  1600., # 终端位置160与其他cost各占一半
        end_weight =  20., # 终端位置160与其他cost各占一半
        ref_v_weight = 1.,
        ref_a_weight= 1.,
        ref_p_weight = 1.)


    start = np.array([0.7921,13.535,1.,0.0,1.,0.]).T
    goal = (-3.56, 5.6)

    ref_traj_info = [[(1.4728878354147859, 12.097789760714159), (0.7514952373161028, -0.8127539427640484), (0.414049363075255, -0.7997176772247541)], [(1.5109763104510379, 12.056173131552171), (0.7719619877344635, -0.8514901108093704), (0.4042386445812106, -0.7491909748594432)], [(1.550074919896432, 12.012684154750449), (0.791882125589424, -0.8876226564566909), (0.3922093549991603, -0.6956412448960212)], [(1.59015362114509, 11.96745661905442), (0.8111483964955724, -0.9210106978648025), (0.37810846108866913, -0.6394792688824075)], [(1.6311771925803087, 11.92063084382462), (0.8296608944054762, -0.9515338922698938), (0.3620829296093018, -0.5811158283665208)], [(1.673105600991466, 11.872352652082853), (0.8473270616096807, -0.9790924359855491), (0.3442797273206234, -0.5209617048962805)], [(1.7158943689909139, 11.822772343558304), (0.8640616887367094, -1.0036070644027482), (0.32484582098219883, -0.45942768001960543)], [(1.759494942430882, 11.772043667733678), (0.8797869147530641, -1.0250190519898674), (0.3039281773535928, -0.39692453528441474)], [(1.8038550578203738, 11.72032279689132), (0.894432226963225, -1.0432902122926795), (0.2816737631943705, -0.33386305223862756)], [(1.848919109742066, 11.667767299159362), (0.9079344610096505, -1.0584028979343525), (0.25822954526409664, -0.27065401243016285)], [(1.8946285182692075, 11.614535111557828), (0.9202378008727766, -1.07036000061545), (0.23374249032233624, -0.20770819740693963)], [(1.9409220963825193, 11.56078351304479), (0.9312937788710187, -1.0791849511139322), (0.2083595651286542, -0.1454363887168772)], [(1.9877364173870913, 11.506668097562487), (0.9410612756607695, -1.0849217192851555), (0.1822277364426154, -0.08424936790789439)], [(2.0350061823292833, 11.45234174708345), (0.9495065202364004, -1.0876348140618717), (0.1554939710237849, -0.02455791652791063)], [(2.082664588221107, 11.397953610429138), (0.9566031706784336, -1.087408706204184), (0.12831169548555418, 0.03327336387874386)], [(2.130643717259124, 11.343648232173585), (0.9623332023834441, -1.0843414785490524), (0.10086017585662158, 0.08906459178209222)], [(2.1788749793371567, 11.28956506253207), (0.9666877962939593, -1.078538476259803), (0.0733251380195121, 0.14268206565574637)], [(2.2272895755535034, 11.23583811733478), (0.9696674196466344, -1.0701117295760791), (0.0458923078567508, 0.19399208397331785)], [(2.27581896252565, 11.18259564377303), (0.9712818259722492, -1.0591799538138458), (0.01874741125086269, 0.2428609452084186)], [(2.324395316704973, 11.129959786145431), (0.9715500550957104, -1.0458685493653852), (-0.007923825915627201, 0.2891549478346602)]]
    X_ref = []
    for i in range(nb_timesteps):
        # cost中不关心的项一律给101
        X_ref.append(101)
        X_ref.append(101)
        X_ref.append(101)
        X_ref.append(101)
        X_ref.append(0.) # 最小化加速度
        X_ref.append(0.)
    
    sub_goal = (1.5, 12)
    # X_ref  px1,py1,...,axN,ayN
    # cost中不关心的项一律给-1
    X_ref[0] = sub_goal[0]
    X_ref[1] = sub_goal[1]

    X_ref[-3] = 0. # 最小化终态速度
    X_ref[-4] = 0.
    X_ref[-5] = goal[1]
    X_ref[-6] = goal[0]
    X_ref = np.array(X_ref).T

    # 360个约束 20步 耗时20ms
    # 0约束 20步 耗时1ms左右
    # 180 10ms
    # 120 7ms
    # X_ref[-5] = -1.53835324
    # X_ref[-6] = 0.29218931
    # convex = [(9.675002098083496, 2.6371435524197295e-07), (5.992757320404053, -4.222295761108398), (-2.858917236328125, -4.026829719543457), (-3.914973258972168, -2.622645616531372), (-4.080860137939453, 7.839923858642578), (9.675002098083496, 2.6371435524197295e-07)]
    
    t = 0.
    start_time = time.time()
    for i in range(1000):
        slover.solve(start,X_ref,convex_vertex=convex)
        # slover.solve(start,X_ref,convex_vertex=None)
        p,v,a,j,cost = slover.states
        t = t + time.time() - start_time
        start_time = time.time()
    print('time',t/1000)
    # print(time.time() - start_time)
    print(cost)
    print("p:")
    print(p)
    print("v:")
    print(v)
    print("a:")
    print(a)
    print("j:")
    print(j)

    # plt.plot(start[0], start[1], 'ro')
    # plt.plot([xy[0] for xy in p], [xy[1] for xy in p], 'r')

    # convex.append(convex[0])
    # plt.plot([p[0] for p in convex], [p[1] for p in convex], 'g')

    # plt.plot(X_ref[-6], X_ref[-5], 'bo')
    # plt.plot(X_ref[0], X_ref[1], 'bo')

    # plt.plot([s[0][0] for s in ref_traj_info], [s[0][1] for s in ref_traj_info], 'b')
    # plt.show()

# if __name__ == "__main__":
#     slover = SO2MPC_u(N=20,control_period=0.05
#     ,v_limit=3.
#     ,a_limit=8.
#     ,j_limit=40.)
#     start = [(1,2),(0,0),(0,0)] # 领先
#     # start = [(1.436231017112732, 12.424905776977539), (0.4554591774940491, -0.06441496312618256), (0.0, 0.0)]
#     # start = [(11.639065742492676, 14.49618911743164), (1, -1), (0.0, -0.0)] # 落后

#     # start = [(0.0, 0.), (0., -0.), (0.06825739568400957, -0.3312339563331807)]
#     specify = (2.324395316704973, 11.129959786145431)
#     ref_traj_info = [[(1.4728878354147859, 12.097789760714159), (0.7514952373161028, -0.8127539427640484), (0.414049363075255, -0.7997176772247541)], [(1.5109763104510379, 12.056173131552171), (0.7719619877344635, -0.8514901108093704), (0.4042386445812106, -0.7491909748594432)], [(1.550074919896432, 12.012684154750449), (0.791882125589424, -0.8876226564566909), (0.3922093549991603, -0.6956412448960212)], [(1.59015362114509, 11.96745661905442), (0.8111483964955724, -0.9210106978648025), (0.37810846108866913, -0.6394792688824075)], [(1.6311771925803087, 11.92063084382462), (0.8296608944054762, -0.9515338922698938), (0.3620829296093018, -0.5811158283665208)], [(1.673105600991466, 11.872352652082853), (0.8473270616096807, -0.9790924359855491), (0.3442797273206234, -0.5209617048962805)], [(1.7158943689909139, 11.822772343558304), (0.8640616887367094, -1.0036070644027482), (0.32484582098219883, -0.45942768001960543)], [(1.759494942430882, 11.772043667733678), (0.8797869147530641, -1.0250190519898674), (0.3039281773535928, -0.39692453528441474)], [(1.8038550578203738, 11.72032279689132), (0.894432226963225, -1.0432902122926795), (0.2816737631943705, -0.33386305223862756)], [(1.848919109742066, 11.667767299159362), (0.9079344610096505, -1.0584028979343525), (0.25822954526409664, -0.27065401243016285)], [(1.8946285182692075, 11.614535111557828), (0.9202378008727766, -1.07036000061545), (0.23374249032233624, -0.20770819740693963)], [(1.9409220963825193, 11.56078351304479), (0.9312937788710187, -1.0791849511139322), (0.2083595651286542, -0.1454363887168772)], [(1.9877364173870913, 11.506668097562487), (0.9410612756607695, -1.0849217192851555), (0.1822277364426154, -0.08424936790789439)], [(2.0350061823292833, 11.45234174708345), (0.9495065202364004, -1.0876348140618717), (0.1554939710237849, -0.02455791652791063)], [(2.082664588221107, 11.397953610429138), (0.9566031706784336, -1.087408706204184), (0.12831169548555418, 0.03327336387874386)], [(2.130643717259124, 11.343648232173585), (0.9623332023834441, -1.0843414785490524), (0.10086017585662158, 0.08906459178209222)], [(2.1788749793371567, 11.28956506253207), (0.9666877962939593, -1.078538476259803), (0.0733251380195121, 0.14268206565574637)], [(2.2272895755535034, 11.23583811733478), (0.9696674196466344, -1.0701117295760791), (0.0458923078567508, 0.19399208397331785)], [(2.27581896252565, 11.18259564377303), (0.9712818259722492, -1.0591799538138458), (0.01874741125086269, 0.2428609452084186)], [(2.324395316704973, 11.129959786145431), (0.9715500550957104, -1.0458685493653852), (-0.007923825915627201, 0.2891549478346602)]]
    

#     start_time = time.time()
#     success,x_opt,y_opt,vx_opt,vy_opt,ax_opt,ay_opt,jx_opt,jy_opt = slover.formulate(start,specify=specify)
#     # success,x_opt,y_opt,vx_opt,vy_opt,ax_opt,ay_opt,jx_opt,jy_opt = slover.formulate(start,ref_traj_info=ref_traj_info)

#     print(time.time() - start_time)

#     plt.plot(start[0][0], start[0][1], 'ro')
#     plt.plot(x_opt, y_opt, 'r')

#     plt.plot(ref_traj_info[0][0][0], ref_traj_info[0][0][1], 'bo')
#     plt.plot([s[0][0] for s in ref_traj_info], [s[0][1] for s in ref_traj_info], 'b')
#     plt.show()

# if __name__ == "__main__":
#     slover = SO2MPC(N=20,control_period=0.05
#     ,v_limit=3.
#     ,a_limit=8.
#     ,j_limit=40.)
#     # start = [(0.2, 1.), (0., -0.), (0.06825739568400957, -0.3312339563331807)] # 领先
#     start = [(1.436231017112732, 12.424905776977539), (0.4554591774940491, -0.06441496312618256), (0.0, 0.0)]
#     # start = [(11.639065742492676, 14.49618911743164), (1, -1), (0.0, -0.0)] # 落后

#     # start = [(0.0, 0.), (0., -0.), (0.06825739568400957, -0.3312339563331807)]

#     ref_traj_info = [[(1.4728878354147859, 12.097789760714159), (0.7514952373161028, -0.8127539427640484), (0.414049363075255, -0.7997176772247541)], [(1.5109763104510379, 12.056173131552171), (0.7719619877344635, -0.8514901108093704), (0.4042386445812106, -0.7491909748594432)], [(1.550074919896432, 12.012684154750449), (0.791882125589424, -0.8876226564566909), (0.3922093549991603, -0.6956412448960212)], [(1.59015362114509, 11.96745661905442), (0.8111483964955724, -0.9210106978648025), (0.37810846108866913, -0.6394792688824075)], [(1.6311771925803087, 11.92063084382462), (0.8296608944054762, -0.9515338922698938), (0.3620829296093018, -0.5811158283665208)], [(1.673105600991466, 11.872352652082853), (0.8473270616096807, -0.9790924359855491), (0.3442797273206234, -0.5209617048962805)], [(1.7158943689909139, 11.822772343558304), (0.8640616887367094, -1.0036070644027482), (0.32484582098219883, -0.45942768001960543)], [(1.759494942430882, 11.772043667733678), (0.8797869147530641, -1.0250190519898674), (0.3039281773535928, -0.39692453528441474)], [(1.8038550578203738, 11.72032279689132), (0.894432226963225, -1.0432902122926795), (0.2816737631943705, -0.33386305223862756)], [(1.848919109742066, 11.667767299159362), (0.9079344610096505, -1.0584028979343525), (0.25822954526409664, -0.27065401243016285)], [(1.8946285182692075, 11.614535111557828), (0.9202378008727766, -1.07036000061545), (0.23374249032233624, -0.20770819740693963)], [(1.9409220963825193, 11.56078351304479), (0.9312937788710187, -1.0791849511139322), (0.2083595651286542, -0.1454363887168772)], [(1.9877364173870913, 11.506668097562487), (0.9410612756607695, -1.0849217192851555), (0.1822277364426154, -0.08424936790789439)], [(2.0350061823292833, 11.45234174708345), (0.9495065202364004, -1.0876348140618717), (0.1554939710237849, -0.02455791652791063)], [(2.082664588221107, 11.397953610429138), (0.9566031706784336, -1.087408706204184), (0.12831169548555418, 0.03327336387874386)], [(2.130643717259124, 11.343648232173585), (0.9623332023834441, -1.0843414785490524), (0.10086017585662158, 0.08906459178209222)], [(2.1788749793371567, 11.28956506253207), (0.9666877962939593, -1.078538476259803), (0.0733251380195121, 0.14268206565574637)], [(2.2272895755535034, 11.23583811733478), (0.9696674196466344, -1.0701117295760791), (0.0458923078567508, 0.19399208397331785)], [(2.27581896252565, 11.18259564377303), (0.9712818259722492, -1.0591799538138458), (0.01874741125086269, 0.2428609452084186)], [(2.324395316704973, 11.129959786145431), (0.9715500550957104, -1.0458685493653852), (-0.007923825915627201, 0.2891549478346602)]]


#     start_time = time.time()
#     success,x_opt,y_opt,vx_opt,vy_opt,ax_opt,ay_opt,jx_opt,jy_opt = slover.formulate(start,ref_traj_info)
#     print(time.time() - start_time)

#     plt.plot(start[0][0], start[0][1], 'ro')
#     plt.plot(x_opt, y_opt, 'r')

#     plt.plot(ref_traj_info[0][0][0], ref_traj_info[0][0][1], 'bo')
#     plt.plot([s[0][0] for s in ref_traj_info], [s[0][1] for s in ref_traj_info], 'b')
#     plt.show()