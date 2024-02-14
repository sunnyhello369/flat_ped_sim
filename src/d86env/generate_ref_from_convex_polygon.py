from audioop import reverse
from doctest import FAIL_FAST
import imp
import sys
import time

from matplotlib.pyplot import box, sca
from galaxy2d import Graham_Andrew_Scan,polar2xy,galaxy,galaxy_xyin_360out
# sys.path.append(r"casadi-linux-py27-v3.5.5-64bit")
# sys.path.append(r"casadi-linux-py36-v3.5.5-64bit")

import math
from cubic_bspline import *
from galaxy2d import galaxy

# sys.path.append(r"/home/chen/desire_10086/flat_ped_sim/src/d86env/casadi-linux-py37-v3.5.5-64bit")
# from casadi import *


# 取模，Python中可直接用%，计算模，r = a % b
def mod(a, b):
    c = a // b
    r = a - c * b
    return r

# 取余
def rem(a, b):
    c = int(a / b)
    r = a - c * b
    return r
    
def plot_spline(sp,polygons,start,end):
    figure,xycanv = plt.subplots()
    for poly in polygons:
        polygon_x = [p[0] for p in poly]
        polygon_x.append(poly[0][0])
        polygon_y = [p[1] for p in poly]
        polygon_y.append(poly[0][1])
        
        xycanv.plot(polygon_x, polygon_y, color='r',lw=2)

    xycanv.plot(end[0], end[1], 'ro')
    xycanv.plot(start[0], start[1], 'bo')
    calc_2d_picewise_spline_interpolation(sp,xycanv,num=300)
    # calc_2d_spline_interpolation(sp,xycanv,num=300)


def convex_counter_lock_sort(points):
    n = len(points)
    if n < 3:
        return points
    mass_x = 0
    mass_y = 0
    for p in points:
        mass_x += p[0]
        mass_y += p[1]
    mass_x /= n
    mass_y /= n
    angle = []
    for p in points:
        a = math.atan2(p[1] - mass_y,p[0] - mass_x)
        if a < 0:
            a += 2*math.pi
        angle.append([p,a])
    angle = sorted(angle, key=lambda x:x[1])
    # points = 
    return [p[0] for p in angle]


# 全向移动微分平坦的输出空间为x,y,yaw
# Fast中用的是非clamped曲线实现
# 为了实现交叠区域的必经点约束，还是需要做成pivewise的b样条优化
class qnonuniform_bspline_optim:
    def __init__(self,convex_polygons,start,end,degrees,time_span
    ,v_weight=1.
    ,a_weight=1.
    ,j_weight=1.
    ,v_limit=3.
    ,a_limit=30.
    ,j_limit=300.):

        self.x = [] # 被优化变量 px0,px1,....,pxn,py0,...,py0
        self.cost_aim = [] # 
        self.v_weight = v_weight
        self.a_weight = a_weight
        self.j_weight = j_weight
        self.cost = 0.

        # time_span轨迹段数，总的轨迹点数是len(convex_polygons) + 2,段数是len(convex_polygons) + 1
        self.p = degrees
        # 交叠区域控制点的重复度也为p，要求必须经过这个点，否则还是会出现样条曲线在约束外的情况
        self.u = [0. for i in range(self.p+1)]
        for i in range(len(time_span)):
            self.u.append(self.u[-1] + time_span[i])
        self.u.extend([self.u[-1] for i in range(self.p)])

        # convex_polygons约束多边形，除了初终态，其他控制点都被约束在某个多边形内或两个凸多边形的交叉部分内
        # convex_polygons保存了每个约束多边形
        self.np1 = len(convex_polygons) # 总控制点数量 - 2除去了起点和终点

        print(len(self.u))
        print(self.u)

        self.lbx = [-float('inf') for i in range(2*self.np1)] # 被优化变量下界
        self.ubx = [float('inf') for i in range(2*self.np1)] # 被优化变量下=上界
        self.other_constraints = [] # 约束，比如点之间的连续性约束
        self.lbc = [] # 约束的下界
        self.ubc = [] # 约束的上界
        self.convex_polygons = []
        for poly in convex_polygons:
            self.convex_polygons.append(convex_counter_lock_sort(poly))
        self.start = start # 3[(x,y),(dx,dy),(ddx,ddy)]
        self.end = end
        self.v_limit = v_limit
        self.a_limit = a_limit
        self.j_limit = j_limit

        self.vx = []
        self.ax = []
        self.vy = []
        self.ay = []
        self.jx = []
        self.jy = []
        # qnonuniform_clamped_bspline2D()

    def get_position_constraints(self):
        # 初态和终态位置
        # 初态和终态控制点为常量，应该剔除出优化
        # # 起点
        # self.lbx[0] = self.start[0][0]
        # self.ubx[0] = self.start[0][0]
        # self.lbx[self.np1] = self.start[0][1]
        # self.ubx[self.np1] = self.start[0][1]
        # # 终点
        # self.lbx[self.np1 - 1] = self.end[0][0]
        # self.ubx[self.np1 - 1] = self.end[0][0]
        # self.lbx[2*self.np1 - 1] = self.end[0][1]
        # self.ubx[2*self.np1 - 1] = self.end[0][1]

        # 点在凸多边形内部约束，逆时针将顶点和点所成直线叉乘，每个面积都为正>0写成Ax<b形式


        for j,poly in enumerate(self.convex_polygons):
            for i in range(1,len(poly)):
                # for j in range(self.np1):
                    # v1xv0 消除交叉项后的结果
                self.other_constraints.append(poly[i-1][0]*poly[i][1] -
                poly[i-1][1]*poly[i][0] +
                (poly[i][0] - poly[i-1][0])*self.x[j + self.np1]+
                (poly[i-1][1] - poly[i][1])*self.x[j])

                self.lbc.append(0)
                self.ubc.append(float('inf'))
            # 最后一条边
            self.other_constraints.append(poly[-1][0]*poly[0][1] -
            poly[-1][1]*poly[0][0] +
            (poly[0][0] - poly[-1][0])*self.x[j + self.np1]+
            (poly[-1][1] - poly[0][1])*self.x[j])
            self.lbc.append(0)
            self.ubc.append(float('inf'))

    def get_feasibility_constraints(self):
        # v
        # Vi = p(Qi+1 - Qi)/(ui+p+1 - ui+1)  uniform -- > (Qi+1 - Qi)/dt


        # 初态v
        self.other_constraints.append(self.p*(self.x[0] - self.start[0][0])/(self.u[self.p + 1] - self.u[1]))
        self.vx.append(self.p*(self.x[0] - self.start[0][0])/(self.u[self.p + 1] - self.u[1]))
        self.lbc.append(self.start[1][0])
        self.ubc.append(self.start[1][0])       
        
        self.other_constraints.append(self.p*(self.x[self.np1] - self.start[0][1])/(self.u[self.p + 1] - self.u[1]))
        self.vy.append(self.p*(self.x[self.np1] - self.start[0][1])/(self.u[self.p + 1] - self.u[1]))
        self.lbc.append(self.start[1][1])
        self.ubc.append(self.start[1][1])  

        for i in range(1,self.np1):
            # vx 
            self.other_constraints.append(self.p*(self.x[i] - self.x[i-1])/(self.u[i + self.p + 1] - self.u[i + 1]))
            self.vx.append(self.p*(self.x[i] - self.x[i-1])/(self.u[i + self.p + 1] - self.u[i+1]))
            self.lbc.append(-self.v_limit)
            self.ubc.append(self.v_limit)
            # vy 
            self.other_constraints.append(self.p*(self.x[i+self.np1] - self.x[i+self.np1-1])/(self.u[i + self.p + 1] - self.u[i + 1]))
            self.vy.append(self.p*(self.x[i+self.np1] - self.x[i+self.np1-1])/(self.u[i + self.p + 1] - self.u[i + 1]))
            self.lbc.append(-self.v_limit)
            self.ubc.append(self.v_limit)
        
        # 终态v
        self.other_constraints.append(self.p*(self.end[0][0] - self.x[self.np1 - 1])/(self.u[self.np1 + self.p -1] - self.u[self.np1 - 1]))
        self.vx.append(self.p*(self.end[0][0] - self.x[self.np1 - 1])/(self.u[self.np1 + self.p - 1] - self.u[self.np1 - 1]))
        self.lbc.append(self.end[1][0])
        self.ubc.append(self.end[1][0])       
        
        self.other_constraints.append(self.p*(self.end[0][1] - self.x[-1])/(self.u[self.np1 + self.p - 1] - self.u[self.np1 - 1]))
        self.vy.append(self.p*(self.end[0][1] - self.x[-1])/(self.u[self.np1 + self.p - 1] - self.u[self.np1 - 1]))
        self.lbc.append(self.end[1][1])
        self.ubc.append(self.end[1][1])   



        # Ai = (p-1)(Vi+1 - Vi)/(ui+p+1 - ui+2)  uniform -- > (Vi+1 - Vi)/dt = (Qi+2 - 2Qi+1 + Qi)/dt^2
        # 初态a
        self.other_constraints.append((self.p - 1)*(self.vx[1] - self.vx[0])/(self.u[self.p+1] - self.u[2]))
        self.ax.append((self.p - 1)*(self.vx[1] - self.vx[0])/(self.u[self.p+1] - self.u[2]))
        self.lbc.append(self.start[2][0])
        self.ubc.append(self.start[2][0])       

        self.other_constraints.append((self.p - 1)*(self.vy[1] - self.vy[0])/(self.u[self.p+1] - self.u[2]))
        self.ay.append((self.p - 1)*(self.vy[1] - self.vy[0])/(self.u[self.p+1] - self.u[2]))
        self.lbc.append(self.start[2][1])
        self.ubc.append(self.start[2][1])  

        for i in range(2,len(self.vx)-1):
            # Ax 
            self.other_constraints.append((self.p - 1)*(self.vx[i] - self.vx[i-1])/(self.u[i+self.p] - self.u[i+1]))
            self.ax.append((self.p - 1)*(self.vx[i] - self.vx[i-1])/(self.u[i+self.p] - self.u[i+1]))
            self.lbc.append(-self.a_limit)
            self.ubc.append(self.a_limit)
            # Ay 
            self.other_constraints.append((self.p - 1)*(self.vy[i] - self.vy[i-1])/(self.u[i+self.p] - self.u[i+1]))
            self.ay.append((self.p - 1)*(self.vy[i] - self.vy[i-1])/(self.u[i+self.p] - self.u[i+1]))
            self.lbc.append(-self.a_limit)
            self.ubc.append(self.a_limit)


        # Ji = (p-2)(Ai+1 - Ai)/(ui+p+1 - ui+3)  uniform -- > (Ai+1 - Ai)/dt = (Qi+3 - 3Qi+2 + 3Qi+1 - Qi)/dt^3
        self.other_constraints.append((self.p - 1)*(self.vx[-1] - self.vx[-2])/(self.u[self.np1 + self.p - 2] - self.u[self.np1 - 1]))
        self.ax.append((self.p - 1)*(self.vx[-1] - self.vx[-2])/(self.u[self.np1 + self.p - 2] - self.u[self.np1 - 1]))
        self.lbc.append(self.end[2][0])
        self.ubc.append(self.end[2][0])       
        
        self.other_constraints.append((self.p - 1)*(self.vy[-1] - self.vy[-2])/(self.u[self.np1 + self.p - 2] - self.u[self.np1 - 1]))
        self.ay.append((self.p - 1)*(self.vy[-1] - self.vy[-2])/(self.u[self.np1 + self.p - 2] - self.u[self.np1 - 1]))
        self.lbc.append(self.end[2][1])
        self.ubc.append(self.end[2][1])  

        # jerk
        # 初态jerk都为0

        for i in range(0,len(self.ax)-1):
            # jx 
            self.other_constraints.append((self.p - 2)*(self.ax[i+1] - self.ax[i])/(self.u[i+self.p+1] - self.u[i+3]))
            self.jx.append((self.p - 2)*(self.ax[i+1] - self.ax[i])/(self.u[i+self.p+1] - self.u[i+3]))
            self.lbc.append(-self.j_limit)
            self.ubc.append(self.j_limit)
            # jy 
            self.other_constraints.append((self.p - 2)*(self.ay[i+1] - self.ay[i])/(self.u[i+self.p+1] - self.u[i+3]))
            self.jy.append((self.p - 2)*(self.ay[i+1] - self.ay[i])/(self.u[i+self.p+1] - self.u[i+3]))
            self.lbc.append(-self.j_limit)
            self.ubc.append(self.j_limit)



    def formulate(self):
        for i in range(1,self.np1+1):
            self.x.append(SX.sym('x_' + str(i)))
            self.lbx[i - 1] = -50.
            self.ubx[i - 1] = 50.

        for i in range(1,self.np1+1):
            self.x.append(SX.sym('y_' + str(i)))
            self.lbx[i - 1 + self.np1] = -50.
            self.ubx[i - 1 + self.np1] = 50.
        
        self.get_position_constraints()
        self.get_feasibility_constraints()

        # for i in range(self.n):
        #     print(str(i)+":")
        #     print(self.lbx[i],self.lbx[self.n+i],self.lbx[2*self.n+i])
        #     print(self.ubx[i],self.ubx[self.n+i],self.ubx[2*self.n+i])
       
        # 光滑代价 初态a和终态a为约束无法改变

        for i in range(1,len(self.ax)-1):
            self.cost += self.ax[i]**2
            self.cost_aim.append(self.ax[i]**2)
        for i in range(1,len(self.ay)-1):
            self.cost += self.ay[i]**2
            self.cost_aim.append(self.ay[i]**2)
            
        # for i in range(len(self.jx)):
        #     self.cost += 0.5*self.jx[i]**2
        #     self.cost_aim.append(1*self.jx[i]**2)
        # for i in range(len(self.jy)):
        #     self.cost += 0.5*self.jy[i]**2
        #     self.cost_aim.append(1*self.jy[i]**2)

        # for i in range(len(self.ax)):
        #     self.cost += self.ax[i]**2
        #     self.cost_aim.append(self.ax[i]**2)
        # for i in range(len(self.ay)):
        #     self.cost += self.ay[i]**2
        #     self.cost_aim.append(self.ay[i]**2)

        qp = {'x':vertcat(*self.x), 'f':self.cost, 'g':vertcat(*self.other_constraints)}
        # Solve with 
        solver = qpsol('solver', 'qpoases', qp, {'sparse':True} )# printLevel':'none' 
        # Get the optimal solution
        sol = solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbc, ubg=self.ubc)
        x_opt = []
        y_opt = []
        # x_opt = sol['x']
        for i in range(self.np1): # remove DM attr
            x_opt.append(float(sol['x'][i]))        
        for i in range(self.np1,2*self.np1): # remove DM attr
            y_opt.append(float(sol['x'][i]))  
        print('opt cost = ', sol['f'])

        return x_opt,y_opt


# 每个box内是一条b样条，此b样条的终点一定在交叠区域内或就是规定终点，k个box对应k条b样条
# 每条b样条是均匀b样条，要不然参数太多了

class quniform_one_bspline_optim:
    def __init__(self,one_spline_ctrl_points_num,degrees
    ,fix_end_pos_state = False
    ,v_weight=1.
    ,a_weight=1.
    ,j_weight=1.
    # ,ref_track_weight = 15.
    # ,end_p_weight = 1000.
    # ,end_v_weight = 50.
    # ,end_a_weight = 50.
    # ,end_j_weight = 50. # 50
    ,ref_track_weight = 6.
    ,end_p_weight = 1000.
    ,end_v_weight = 6.
    ,end_a_weight = 2.
    ,end_j_weight = 2. # 50
    ,v_limit=3.
    ,a_limit=3.
    ,j_limit=4.):

        self.control_points = [] 
        self.v_weight = v_weight
        self.a_weight = a_weight
        self.j_weight = j_weight
        self.ref_track_weight = ref_track_weight
        self.end_p_weight = end_p_weight
        self.end_v_weight = end_v_weight
        self.end_a_weight = end_a_weight
        self.end_j_weight = end_j_weight
        self.cost = 0.
        # time_span轨迹段数，总的轨迹点数是len(convex_polygons) + 2,段数是len(convex_polygons) + 1
        self.p = degrees

        self.lbpcp = [] # 被优化变量下界 [box1 spline[(p1x,p1y),(),...],box2 spline[]]
        self.ubpcp = [] # 被优化变量下=上界 [box1 spline[(p1x,p1y),(),...],box2 spline[]]
        self.other_constraints = [] # 约束，比如点之间的连续性约束
        self.lbc = [] # 约束的下界
        self.ubc = [] # 约束的上界

        self.v_limit = v_limit
        self.a_limit = a_limit
        self.j_limit = j_limit
        self.vx = []
        self.ax = []
        self.vy = []
        self.ay = []
        self.jx = []
        self.jy = []
        self.u = U_quasi_uniform(one_spline_ctrl_points_num-1,self.p,1.) # n p 
        self.du = self.u[self.p+1]
        # print(len(self.u))
        # print(self.u)

        self.fix_end_pos_state = fix_end_pos_state # 为True则尾部位置为硬约束

        if fix_end_pos_state:
            # 优化变量排除首固定的3个控制点+尾1个位置控制点 尾v,a,j设为软约束
            for i in range(one_spline_ctrl_points_num-4):
                self.control_points.append((SX.sym('x_' + str(i)),SX.sym('y_'+ str(i))))
                # self.lbpcp[-1].append((-float('inf'),-float('inf')))
                # self.ubpcp[-1].append((float('inf'),float('inf')))
                self.lbpcp.append((-30.,-30.))
                self.ubpcp.append((30.,30.))
        else:
            # 优化变量排除首固定的3个控制点
            for i in range(one_spline_ctrl_points_num-3):
                self.control_points.append((SX.sym('x_' + str(i)),SX.sym('y_'+ str(i))))
                # self.lbpcp[-1].append((-float('inf'),-float('inf')))
                # self.ubpcp[-1].append((float('inf'),float('inf')))
                self.lbpcp.append((-30.,-30.))
                self.ubpcp.append((30.,30.))

    def get_position_constraints(self,convex_polygon_vertex):
        # 初态和终态位置
        # 初态和终态控制点为常量，应该剔除出优化
        # # 起点
        # self.lbx[0] = self.start[0][0]
        # self.ubx[0] = self.start[0][0]
        # self.lbx[self.np1] = self.start[0][1]
        # self.ubx[self.np1] = self.start[0][1]
        # # 终点
        # self.lbx[self.np1 - 1] = self.end[0][0]
        # self.ubx[self.np1 - 1] = self.end[0][0]
        # self.lbx[2*self.np1 - 1] = self.end[0][1]
        # self.ubx[2*self.np1 - 1] = self.end[0][1]

        # 点在凸多边形内部约束，顺时针将顶点和点所成直线叉乘，每个面积都为正>0写成Ax<b形式
        for which_point in range(len(self.control_points)):
            for i in range(1,len(convex_polygon_vertex)):
                    # v1xv0 消除交叉项后的结果
                    self.other_constraints.append(convex_polygon_vertex[i-1][0]*convex_polygon_vertex[i][1] -
                    convex_polygon_vertex[i-1][1]*convex_polygon_vertex[i][0] +
                    (convex_polygon_vertex[i][0] - convex_polygon_vertex[i-1][0])*self.control_points[which_point][1]+
                    (convex_polygon_vertex[i-1][1] - convex_polygon_vertex[i][1])*self.control_points[which_point][0])
                    # self.lbc.append(0)
                    # self.ubc.append(float('inf'))
                    self.lbc.append(-float('inf'))
                    self.ubc.append(0.)
            # 最后一条边
            # self.other_constraints.append(convex_polygon_vertex[-1][0]*convex_polygon_vertex[0][1] -
            # convex_polygon_vertex[-1][1]*convex_polygon_vertex[0][0] +
            # (convex_polygon_vertex[0][0] - convex_polygon_vertex[-1][0])*self.control_points[which_point][1]+
            # (convex_polygon_vertex[-1][1] - convex_polygon_vertex[0][1])*self.control_points[which_point][0])
            # # self.lbc.append(0)
            # # self.ubc.append(float('inf'))
            # self.lbc.append(-float('inf'))
            # self.ubc.append(0.)

        # print(self.other_constraints)

    def get_feasibility_constraints(self,time_span,start,end):
        # v
        # Vi = p(Qi+1 - Qi)/(ui+p+1 - ui+1)  uniform -- > (Qi+1 - Qi)/dt

        # 初态v
        print("********")

        self.vx = []
        self.vy = []

        self.vx.append(self.p*(self.control_points[0][0] - start[0][0])/(time_span*(self.u[self.p + 1] - self.u[1])))
        self.other_constraints.append(self.vx[-1])
        self.lbc.append(start[1][0])
        self.ubc.append(start[1][0])       

        self.vy.append(self.p*(self.control_points[0][1] - start[0][1])/(time_span*(self.u[self.p + 1] - self.u[1])))
        self.other_constraints.append(self.vy[-1])
        self.lbc.append(start[1][1])
        self.ubc.append(start[1][1])  


        for j in range(1,len(self.control_points)):          
            # vx 
            self.vx.append(self.p*(self.control_points[j][0] - self.control_points[j-1][0])/(time_span*(self.u[j + self.p + 1] - self.u[j + 1])))
            self.other_constraints.append(self.vx[-1])
            self.lbc.append(-self.v_limit)
            self.ubc.append(self.v_limit)

            # vy 
            self.vy.append(self.p*(self.control_points[j][1] - self.control_points[j-1][1])/(time_span*(self.u[j + self.p + 1] - self.u[j + 1])))
            self.other_constraints.append(self.vy[-1])
            self.lbc.append(-self.v_limit)
            self.ubc.append(self.v_limit)   

        # 终态v
        self.vx.append(self.p*(end[0][0] - self.control_points[-1][0])/(time_span*(self.u[len(self.control_points) + self.p +1] - self.u[len(self.control_points) + 1])))
        self.other_constraints.append(self.vx[-1])
        # self.lbc.append(end[1][0] - 0.1)
        # self.ubc.append(end[1][0] + 0.1)       
        self.lbc.append(-self.v_limit)
        self.ubc.append(self.v_limit)   

        self.vy.append(self.p*(end[0][1] - self.control_points[-1][1])/(time_span*(self.u[len(self.control_points) + self.p +1] - self.u[len(self.control_points) + 1])))
        self.other_constraints.append(self.vy[-1])
        # self.lbc.append(end[1][1] - 0.1)
        # self.ubc.append(end[1][1] + 0.1)   
        self.lbc.append(-self.v_limit)
        self.ubc.append(self.v_limit)   

        self.lbc[-1] = self.lbc[-1]*0.6
        self.ubc[-1] = self.ubc[-1]*0.6
        self.lbc[-2] = self.lbc[-2]*0.6
        self.ubc[-2] = self.ubc[-2]*0.6

        print(self.vx)
        print(self.vy)
        print("********")

        # Ai = (p-1)(Vi+1 - Vi)/(ui+p+1 - ui+2)  uniform -- > (Vi+1 - Vi)/dt = (Qi+2 - 2Qi+1 + Qi)/dt^2
        # 初态a

        self.ax = []
        self.ay = []

        self.ax.append((self.p - 1)*(self.vx[1] - self.vx[0])/(time_span*(self.u[self.p + 1] - self.u[2])))
        self.other_constraints.append(self.ax[-1])
        self.lbc.append(start[2][0])
        self.ubc.append(start[2][0])    

        self.ay.append((self.p - 1)*(self.vy[1] - self.vy[0])/(time_span*(self.u[self.p + 1] - self.u[2])))
        self.other_constraints.append(self.ay[-1])
        self.lbc.append(start[2][1])
        self.ubc.append(start[2][1])  

        print(len(self.vx))
        print(self.u)
        for j in range(1,len(self.vx)-2):
            # Ax 
            self.ax.append((self.p - 1)*(self.vx[j+1] - self.vx[j])/((time_span*(self.u[j+ self.p + 1] - self.u[j + 2]))))
            self.other_constraints.append( self.ax[-1])
            self.lbc.append(-self.a_limit)
            self.ubc.append(self.a_limit)
            # Ay 
            self.ay.append((self.p - 1)*(self.vy[j+1] - self.vy[j])/((time_span*(self.u[j + self.p + 1] - self.u[j+ 2]))))
            self.other_constraints.append( self.ay[-1])
            self.lbc.append(-self.a_limit)
            self.ubc.append(self.a_limit)

            print(self.u[j+ self.p + 1] ,self.u[j + 2])

        self.ax.append((self.p - 1)*(self.vx[-1] - self.vx[-2])/((time_span*(self.u[len(self.vx)-2 + self.p + 1] - self.u[len(self.vx)-2 + 2]))))
        self.other_constraints.append(self.ax[-1])
        # self.lbc.append(end[2][0] - 0.5)
        # self.ubc.append(end[2][0] + 0.5)  
        self.lbc.append(-self.a_limit)
        self.ubc.append(self.a_limit)

        self.ay.append((self.p - 1)*(self.vy[-1] - self.vy[-2])/((time_span*(self.u[len(self.vy)-2 + self.p + 1] - self.u[len(self.vy)-2 + 2]))))
        self.other_constraints.append(self.ay[-1])
        # self.lbc.append(end[2][1] - 0.5)
        # self.ubc.append(end[2][1] + 0.5)       
        self.lbc.append(-self.a_limit)
        self.ubc.append(self.a_limit)

        self.lbc[-1] = self.lbc[-1]*0.4
        self.ubc[-1] = self.ubc[-1]*0.4
        self.lbc[-2] = self.lbc[-2]*0.4
        self.ubc[-2] = self.ubc[-2]*0.4
        
        print(self.ax)
        print(self.ay)
        print("********")

        # jerk
        # 初态jerk都为0
        # Ji = (p-2)(Ai+1 - Ai)/(ui+p+1 - ui+3)  uniform -- > (Ai+1 - Ai)/dt = (Qi+3 - 3Qi+2 + 3Qi+1 - Qi)/dt^3

        self.jx= []
        self.jy = []
        self.jx.append((self.p - 2)*(self.ax[1] - self.ax[0])/(time_span*(self.u[self.p+1] - self.u[3])))
        self.other_constraints.append(self.jx[-1])
        self.lbc.append(start[3][0])
        self.ubc.append(start[3][0])       

        self.jy.append((self.p - 2)*(self.ay[1] - self.ay[0])/(time_span*(self.u[self.p+1] - self.u[3])))
        self.other_constraints.append(self.jy[-1])
        self.lbc.append(start[3][1])
        self.ubc.append(start[3][1])  

        for j in range(1,len(self.ax)-2):
            # jx 
            self.jx.append((self.p - 2)*(self.ax[j+1] - self.ax[j])/(time_span*(self.u[j+self.p+1] - self.u[j+3])))
            self.other_constraints.append(self.jx[-1])
            self.lbc.append(-self.j_limit)
            self.ubc.append(self.j_limit)

            # jy 
            self.jy.append((self.p - 2)*(self.ay[j+1] - self.ay[j])/(time_span*(self.u[j+self.p+1] - self.u[j+3])))
            self.other_constraints.append(self.jy[-1])
            self.lbc.append(-self.j_limit)
            self.ubc.append(self.j_limit)

        self.jx.append((self.p - 2)*(self.ax[-1] - self.ax[-2])/((time_span*(self.u[len(self.ax)-2 + self.p + 1] - self.u[len(self.ax)-2 + 3]))))
        self.other_constraints.append(self.jx[-1])
        # self.lbc.append(end[3][0] - 3)
        # self.ubc.append(end[3][0] + 3)  
        self.lbc.append(-self.j_limit)
        self.ubc.append(self.j_limit)

        self.jy.append((self.p - 2)*(self.ay[-1] - self.ay[-2])/((time_span*(self.u[len(self.ay)-2 + self.p + 1] - self.u[len(self.ay)-2 + 3]))))
        self.other_constraints.append(self.jy[-1])
        # self.lbc.append(end[3][1] - 3)
        # self.ubc.append(end[3][1] + 3)  
        self.lbc.append(-self.j_limit)
        self.ubc.append(self.j_limit)

        # print(self.jx)
        # print(self.jy)

    def get_feasibility_constraints2(self,time_span,pre_ctrl,end=None):
        # 终态状态全设为软约束,初态硬约束已计算完毕

        # v
        # Vi = p(Qi+1 - Qi)/(ui+p+1 - ui+1)  uniform -- > (Qi+1 - Qi)/dt

        # print("********")
        pre_ctrl_num = len(pre_ctrl)
        self.vx = []
        self.vy = []
        if pre_ctrl_num >= 1:
            for i in range(pre_ctrl_num - 1):
                self.vx.append(self.p*(pre_ctrl[i+1][0] - pre_ctrl[i][0])/(time_span*(self.u[i + self.p + 1] - self.u[i+1])))
                self.vy.append(self.p*(pre_ctrl[i+1][1] - pre_ctrl[i][1])/(time_span*(self.u[i + self.p + 1] - self.u[i+1])))

            self.vx.append(self.p*(self.control_points[0][0] - pre_ctrl[-1][0])/(time_span*(self.u[pre_ctrl_num - 1 +self.p + 1] - self.u[pre_ctrl_num - 1 +1])))
            self.other_constraints.append(self.vx[-1])
            self.lbc.append(-self.v_limit)
            self.ubc.append(self.v_limit)       

            self.vy.append(self.p*(self.control_points[0][1] - pre_ctrl[-1][1])/(time_span*(self.u[pre_ctrl_num - 1 +self.p + 1] - self.u[pre_ctrl_num - 1 +1])))
            self.other_constraints.append(self.vy[-1])
            self.lbc.append(-self.v_limit)
            self.ubc.append(self.v_limit)  


        for j in range(1,len(self.control_points)):          
            # vx 
            self.vx.append(self.p*(self.control_points[j][0] - self.control_points[j-1][0])/(time_span*(self.u[pre_ctrl_num-1 + j + self.p + 1] - self.u[pre_ctrl_num-1 + j + 1])))
            self.other_constraints.append(self.vx[-1])
            self.lbc.append(-self.v_limit)
            self.ubc.append(self.v_limit)

            # vy 
            self.vy.append(self.p*(self.control_points[j][1] - self.control_points[j-1][1])/(time_span*(self.u[pre_ctrl_num-1 + j + self.p + 1] - self.u[pre_ctrl_num-1 + j + 1])))
            self.other_constraints.append(self.vy[-1])
            self.lbc.append(-self.v_limit)
            self.ubc.append(self.v_limit)   

        # 终态v(固定end位置)
        if end is not None and self.fix_end_pos_state:
            self.vx.append(self.p*(end[0][0] - self.control_points[-1][0])/(time_span*(self.u[pre_ctrl_num-1 + len(self.control_points) + self.p +1] - self.u[pre_ctrl_num-1 + len(self.control_points) + 1])))
            self.other_constraints.append(self.vx[-1])   
            self.lbc.append(-self.v_limit)
            self.ubc.append(self.v_limit)   

            self.vy.append(self.p*(end[0][1] - self.control_points[-1][1])/(time_span*(self.u[pre_ctrl_num-1 + len(self.control_points) + self.p +1] - self.u[pre_ctrl_num-1 + len(self.control_points) + 1])))
            self.other_constraints.append(self.vy[-1])
            self.lbc.append(-self.v_limit)
            self.ubc.append(self.v_limit)   

        # vend 收紧
        # self.lbc[-2] = self.lbc[-2]*0.6
        # self.ubc[-2] = self.ubc[-2]*0.6
        # self.lbc[-1] = self.lbc[-1]*0.6
        # self.ubc[-1] = self.ubc[-1]*0.6
        # print(self.vx)
        # print(self.vy)
        # print("********")

        # Ai = (p-1)(Vi+1 - Vi)/(ui+p+1 - ui+2)  uniform -- > (Vi+1 - Vi)/dt = (Qi+2 - 2Qi+1 + Qi)/dt^2
        # 初态a

        self.ax = []
        self.ay = []

        # print(len(self.vx))
        # print(self.u)
        for j in range(len(self.vx)-1):
            # Ax 
            self.ax.append((self.p - 1)*(self.vx[j+1] - self.vx[j])/((time_span*(self.u[j+ self.p + 1] - self.u[j + 2]))))
            self.other_constraints.append( self.ax[-1])
            self.lbc.append(-self.a_limit)
            self.ubc.append(self.a_limit)
            # Ay 
            self.ay.append((self.p - 1)*(self.vy[j+1] - self.vy[j])/((time_span*(self.u[j + self.p + 1] - self.u[j+ 2]))))
            self.other_constraints.append( self.ay[-1])
            self.lbc.append(-self.a_limit)
            self.ubc.append(self.a_limit)

        # a end 收紧
        # self.lbc[-2] = self.lbc[-2]*0.6
        # self.ubc[-2] = self.ubc[-2]*0.6
        # self.lbc[-1] = self.lbc[-1]*0.6
        # self.ubc[-1] = self.ubc[-1]*0.6

        # print(self.ax)
        # print(self.ay)
        # print("********")

        # jerk
        # 初态jerk都为0
        # Ji = (p-2)(Ai+1 - Ai)/(ui+p+1 - ui+3)  uniform -- > (Ai+1 - Ai)/dt = (Qi+3 - 3Qi+2 + 3Qi+1 - Qi)/dt^3

        self.jx= []
        self.jy = []
        for j in range(len(self.ax)-1):
            # jx 
            self.jx.append((self.p - 2)*(self.ax[j+1] - self.ax[j])/(time_span*(self.u[j+self.p+1] - self.u[j+3])))
            self.other_constraints.append(self.jx[-1])
            self.lbc.append(-self.j_limit)
            self.ubc.append(self.j_limit)

            # jy 
            self.jy.append((self.p - 2)*(self.ay[j+1] - self.ay[j])/(time_span*(self.u[j+self.p+1] - self.u[j+3])))
            self.other_constraints.append(self.jy[-1])
            self.lbc.append(-self.j_limit)
            self.ubc.append(self.j_limit)
       
        
    def calc_dynamic_ctrl_points(self,time_span,x_opt,y_opt):
        # 终态状态全设为软约束,初态硬约束已计算完毕

        # v
        # Vi = p(Qi+1 - Qi)/(ui+p+1 - ui+1)  uniform -- > (Qi+1 - Qi)/dt

        # print("********")
        vx = []
        vy = []

        for j in range(len(x_opt)-1):      
            # vx 
            vx.append(self.p*(x_opt[j+1] - x_opt[j])/(time_span*(self.u[j + self.p + 1] - self.u[j + 1])))
            # vy 
            vy.append(self.p*(y_opt[j+1] - y_opt[j])/(time_span*(self.u[j + self.p + 1] - self.u[j + 1])))

        # print(self.vx)
        # print(self.vy)
        # print("********")

        # Ai = (p-1)(Vi+1 - Vi)/(ui+p+1 - ui+2)  uniform -- > (Vi+1 - Vi)/dt = (Qi+2 - 2Qi+1 + Qi)/dt^2
        # 初态a

        ax = []
        ay = []

        # print(len(self.vx))
        # print(self.u)
        for j in range(len(vx)-1):
            # Ax 
            ax.append((self.p - 1)*(vx[j+1] - vx[j])/((time_span*(self.u[j+ self.p + 1] - self.u[j + 2]))))
            # Ay 
            ay.append((self.p - 1)*(vy[j+1] - vy[j])/((time_span*(self.u[j + self.p + 1] - self.u[j+ 2]))))


        # print(self.ax)
        # print(self.ay)
        # print("********")

        # jerk
        # 初态jerk都为0
        # Ji = (p-2)(Ai+1 - Ai)/(ui+p+1 - ui+3)  uniform -- > (Ai+1 - Ai)/dt = (Qi+3 - 3Qi+2 + 3Qi+1 - Qi)/dt^3

        jx= []
        jy = []
        for j in range(len(ax)-1):
            # jx 
            jx.append((self.p - 2)*(ax[j+1] - ax[j])/(time_span*(self.u[j+self.p+1] - self.u[j+3])))
            # jy 
            jy.append((self.p - 2)*(ay[j+1] - ay[j])/(time_span*(self.u[j+self.p+1] - self.u[j+3])))
        return vx,vy,ax,ay,jx,jy

    def calc_cost(self,ref_ctrl_points,end=None):
        self.cost = 0.
        for j in range(len(self.jx)):
            self.cost += self.j_weight*self.jx[j]**2
            self.cost += self.j_weight*self.jy[j]**2
        
        # ref control将end状态的运动信息全部都决定好了，不再需要约束终态了
        for i in range(len(ref_ctrl_points)-1):
            self.cost += self.ref_track_weight*(self.control_points[i][0] - ref_ctrl_points[i][0])**2
            self.cost += self.ref_track_weight*(self.control_points[i][1] - ref_ctrl_points[i][1])**2     
        if not self.fix_end_pos_state:
            # 位置也是软约束
            self.cost += 3*self.ref_track_weight*(self.control_points[-1][0] - ref_ctrl_points[-1][0])**2
            self.cost += 3*self.ref_track_weight*(self.control_points[-1][1] - ref_ctrl_points[-1][1])**2       

        if end is not None:
            # 终态软约束
            if not self.fix_end_pos_state and ref_ctrl_points is None:
                # 位置也是软约束
                self.cost += self.end_p_weight*(self.control_points[-1][0] - end[0][0])**2
                self.cost += self.end_p_weight*(self.control_points[-1][1] - end[0][1])**2        
            self.cost += self.end_v_weight*(self.vx[-1] - end[1][0])**2
            self.cost += self.end_v_weight*(self.vy[-1] - end[1][1])**2
            self.cost += self.end_a_weight*(self.ax[-1] - end[2][0])**2
            self.cost += self.end_a_weight*(self.ay[-1] - end[2][1])**2
            self.cost += (self.end_j_weight - self.j_weight)*self.jx[-1]**2
            self.cost += (self.end_j_weight - self.j_weight)*self.jy[-1]**2
        else:
            # 为了保证下次规划成功 尽可能最小化末端加速度和jerk，否则下次规划初态可能会不满运动学约束
            self.cost += self.end_v_weight*self.vx[-1]**2
            self.cost += self.end_v_weight*self.vy[-1]**2
            self.cost += self.end_a_weight*self.ax[-1]**2
            self.cost += self.end_a_weight*self.ay[-1]**2
            self.cost += (self.end_j_weight - self.j_weight)*self.jx[-1]**2
            self.cost += (self.end_j_weight - self.j_weight)*self.jy[-1]**2

    def bark_calc_cost(self):
        # 刹车规划
        self.cost = 0.
        for j in range(len(self.vx)):
            self.cost += 5.*self.v_weight*self.vx[j]**2
            self.cost += 5.*self.v_weight*self.vy[j]**2
        for j in range(len(self.ax)):
            self.cost += 3.*self.a_weight*self.ax[j]**2
            self.cost += 3.*self.a_weight*self.ay[j]**2
        for j in range(len(self.jx)):
            self.cost += self.j_weight*self.jx[j]**2
            self.cost += self.j_weight*self.jy[j]**2
        

    def formulate(self,time_span,convex_polygon_vertex,start,end=None,ref_ctrl_points=None ):# [(x,y),...,(x,y)]
        self.other_constraints = []
        self.lbc = []
        self.ubc = []

        VIS = False
        # 根据start p,v,a将前3个控制点位置直接确定
        x_opt = [start[0][0]]
        y_opt = [start[0][1]]
        tp1 = time_span*(self.u[self.p + 1] - self.u[1])/self.p
        tp2 = time_span*(self.u[self.p + 2] - self.u[2])/self.p
        tv1 =  time_span*(self.u[self.p + 1] - self.u[2])/(self.p-1.)
        x_opt.append(start[1][0]*tp1 +x_opt[0]) # vx
        y_opt.append(start[1][1]*tp1 +y_opt[0]) # vy
        x_opt.append((start[2][0]*tv1+start[1][0])*tp2+x_opt[1]) # ax
        y_opt.append((start[2][1]*tv1+start[1][1])*tp2+y_opt[1]) # ay

        if convex_polygon_vertex is not None:
            self.get_position_constraints(convex_polygon_vertex)
        self.get_feasibility_constraints2(time_span,[(x_opt[0],y_opt[0]),(x_opt[1],y_opt[1]),(x_opt[2],y_opt[2])],end)
        # self.get_feasibility_constraints2(time_span,[(x_opt[0],y_opt[0]),(x_opt[1],y_opt[1])],end)
        if ref_ctrl_points is not None:
            self.calc_cost(ref_ctrl_points,end)
        else:
            self.bark_calc_cost() # 刹车规划

        x = []
        lbx = []
        ubx = []
        y = []
        lby = []
        uby = []

        for j in range(len(self.control_points)):
            x.append(self.control_points[j][0])
            y.append(self.control_points[j][1])
            lbx.append(self.lbpcp[j][0])
            lby.append(self.lbpcp[j][1])
            ubx.append(self.ubpcp[j][0])
            uby.append(self.ubpcp[j][1])

        x.extend(y)
        lbx.extend(lby)
        ubx.extend(uby)

        qp = {'x':vertcat(*x), 'f':self.cost, 'g':vertcat(*self.other_constraints)}
        # Solve with 
        solver = qpsol('solver', 'qpoases', qp, {'sparse':True,'printLevel':'none' } )#'printLevel':'none' 

        # Get the optimal solution
        try:
            sol = solver(lbx=lbx, ubx=ubx, lbg=self.lbc, ubg=self.ubc)
        except:
            # print("fail")
            # print("polygon:")
            # print(convex_polygon_vertex)
            # print("start:")
            # print(start)
            # print("ref_ctrl_points:")
            # print(ref_ctrl_points)
            return False,None,None,None

        # print(sol)
        # 分段b样条二维控制点 
        for j in range(len(self.control_points)):
            x_opt.append(float(sol['x'][j]))
            y_opt.append(float(sol['x'][j+len(self.control_points)]))
        
        if self.fix_end_pos_state:
            x_opt.append(end[0][0])
            y_opt.append(end[0][1])

        # print('opt cost = ', sol['f'])
        cost_track = 0.
        if ref_ctrl_points is not None:
            for i in range(len(ref_ctrl_points)-1):
                cost_track += (x_opt[2+i] - ref_ctrl_points[i][0])**2
                cost_track += (y_opt[2+i] - ref_ctrl_points[i][1])**2     
            if not self.fix_end_pos_state:
                # 位置也是软约束
                cost_track += (x_opt[-1] - ref_ctrl_points[-1][0])**2
                cost_track += (y_opt[-1] - ref_ctrl_points[-1][1])**2       
        if VIS:
            print("track cost: "+str(cost_track))
            self.print_sol(x_opt,y_opt,end,time_span)

        spline = quniform_clamped_bspline2D(x_opt,y_opt,5,time_span)
        vx,vy,ax,ay,jx,jy = self.calc_dynamic_ctrl_points(time_span,x_opt,y_opt)
        return True,spline,cost_track,(x_opt,y_opt,vx,vy,ax,ay,jx,jy)
        # return x_opt,y_opt

    def print_sol(self,x_opt,y_opt,end,time_span):
        print(x_opt)
        print(y_opt)

        vx = []
        vy = []
        ax = []
        ay = []
        jx = []
        jy = []

        for j in range(len(x_opt)-1):          
            vx.append(self.p*(x_opt[j+1] - x_opt[j])/(time_span*(self.u[j + self.p + 1] - self.u[j + 1])))
            vy.append(self.p*(y_opt[j+1] - y_opt[j])/(time_span*(self.u[j + self.p + 1] - self.u[j + 1])))

        for j in range(len(vx)-1):
            ax.append((self.p - 1)*(vx[j+1] - vx[j])/((time_span*(self.u[j+ self.p + 1] - self.u[j + 2]))))
            ay.append((self.p - 1)*(vy[j+1] - vy[j])/((time_span*(self.u[j + self.p + 1] - self.u[j+ 2]))))

        for j in range(len(ax)-1):
            jx.append((self.p - 2)*(ax[j+1] - ax[j])/(time_span*(self.u[j+self.p+1] - self.u[j+3])))
            jy.append((self.p - 2)*(ay[j+1] - ay[j])/(time_span*(self.u[j+self.p+1] - self.u[j+3])))


        print("x:")
        print(x_opt)
        print("y:")
        print(y_opt)
        print("vx:")
        print(vx)
        print("vy:")
        print(vy)
        print("ax:")
        print(ax)
        print("ay:")
        print(ay)
        print("jx:")
        print(jx)
        print("jy:")
        print(jy)

        cost_j = 0.
        cost_v = 0.
        cost_v_end = 0.
        cost_a_end = 0.
        cost_j_end = 0.
        for j in range(len(jx)):
            cost_j += self.j_weight*jx[j]**2
            cost_j += self.j_weight*jy[j]**2
        for j in range(len(self.vx)):
            cost_v -= vx[j]**2
            cost_v -= vy[j]**2
        # 终态约束
        if end is not None:
            cost_v_end += self.end_v_weight*(vx[-1] - end[1][0])**2
            cost_v_end += self.end_v_weight*(vy[-1] - end[1][1])**2
            cost_a_end += self.end_a_weight*(ax[-1] - end[2][0])**2
            cost_a_end += self.end_a_weight*(ay[-1] - end[2][1])**2
            cost_j_end += self.end_j_weight*(jx[-1] - end[3][0])**2
            cost_j_end += self.end_j_weight*(jy[-1] - end[3][1])**2
        else:
            cost_v_end += self.end_v_weight*(vx[-1])**2
            cost_v_end += self.end_v_weight*(vy[-1])**2
            cost_a_end += self.end_a_weight*(ax[-1])**2
            cost_a_end += self.end_a_weight*(ay[-1])**2
            cost_j_end += self.end_j_weight*(jx[-1])**2
            cost_j_end += self.end_j_weight*(jy[-1])**2
        print("j cost:")
        print(cost_j)

        print("v cost:")
        print(cost_v)
    
        print("v end cost:")
        print(cost_v_end)

        print("a end cost:")
        print(cost_a_end)

        print("j end cost:")
        print(cost_j_end)
        print(cost_j + cost_v_end + cost_a_end + cost_j_end)

class quniform_piecewise_bspline_optim:
    def __init__(self,one_spline_ctrl_points_num,convex_polygons,start,end,degrees,time_span
    ,v_weight=1.
    ,a_weight=1.
    ,j_weight=1.
    ,v_limit=3.5
    ,a_limit=8.
    ,j_limit=3000.):
        if len(convex_polygons) < 2:
            return
        self.piecewise_control_points = [] # [box1 spline[(p1x,p1y),(),...],box2 spline[]] 上段终点控制点与下段起点控制点相同
        self.cost_aim = [] # 
        self.v_weight = v_weight
        self.a_weight = a_weight
        self.j_weight = j_weight
        self.cost = 0.

        # time_span轨迹段数，总的轨迹点数是len(convex_polygons) + 2,段数是len(convex_polygons) + 1
        self.p = degrees
        # 交叠区域控制点的重复度也为p，要求必须经过这个点，否则还是会出现样条曲线在约束外的情况
        
        self.time_span = time_span 
        self.start_time = [0] # 每条b样条的真实起始时间
        for i in range(len(time_span)):
            self.start_time.append(self.start_time[-1]+time_span[i])

        # convex_polygons约束多边形，除了初终态，其他控制点都被约束在某个多边形内或两个凸多边形的交叉部分内
        # convex_polygons保存了每个约束多边形
        # 传入的convex_polygons就是约束空间不含交叠区域，需要自己算
        # convex_polygons [(x,y,w,h),(),...]
        self.constraints_convexs = []
        prev_overlap = None
        ld = (convex_polygons[0][0]-convex_polygons[0][2]/2.,
            convex_polygons[0][1]-convex_polygons[0][3]/2.)
        rd = (convex_polygons[0][0]+convex_polygons[0][2]/2.,
            convex_polygons[0][1]-convex_polygons[0][3]/2.)
        ru = (convex_polygons[0][0]+convex_polygons[0][2]/2.,
            convex_polygons[0][1]+convex_polygons[0][3]/2.)
        lu = (convex_polygons[0][0]-convex_polygons[0][2]/2.,
            convex_polygons[0][1]+convex_polygons[0][3]/2.)
        for i in range(len(convex_polygons)-1):
            # 逆时针，左下开始
            ldn = (convex_polygons[i+1][0]-convex_polygons[i+1][2]/2.,
                convex_polygons[i+1][1]-convex_polygons[i+1][3]/2.)
            rdn = (convex_polygons[i+1][0]+convex_polygons[i+1][2]/2.,
                convex_polygons[i+1][1]-convex_polygons[i+1][3]/2.)
            run = (convex_polygons[i+1][0]+convex_polygons[i+1][2]/2.,
                convex_polygons[i+1][1]+convex_polygons[i+1][3]/2.)
            lun = (convex_polygons[i+1][0]-convex_polygons[i+1][2]/2.,
                convex_polygons[i+1][1]+convex_polygons[i+1][3]/2.)            

            # 初态overlap
            if prev_overlap is not None:
                self.constraints_convexs.append(prev_overlap)
            
            # 第一个在上个overlap,最后一个点在下个overlap都不在这个范围内
            for j in range(1,one_spline_ctrl_points_num-1):
                self.constraints_convexs.append([ld,rd,ru,lu])
            
            # 计算当前box和下个box的交叉部分
            overlap = [(max(ld[0],ldn[0]),max(ld[1],ldn[1])),(min(rd[0],rdn[0]),max(rd[1],rdn[1]))
            ,(min(ru[0],run[0]),min(ru[1],run[1])),(max(lu[0],lun[0]),min(lu[1],lun[1]))]
            # 此box内最后一个状态点的约束
            self.constraints_convexs.append(overlap)
            ld = ldn
            rd = rdn
            ru = run
            lu = lun
            prev_overlap = overlap
        # 最后一个box
        self.constraints_convexs.append(prev_overlap)
        # 第一个在上个overlap,最后一个点在下个overlap都不在这个范围内
        for j in range(1,one_spline_ctrl_points_num-1):
            self.constraints_convexs.append([ld,rd,ru,lu])        


        self.control_points_num = len(self.constraints_convexs) + 2# 总的控制点数
        self.sub_control_points_num = one_spline_ctrl_points_num# 每段的控制点数
        self.spline_num = int(self.control_points_num/self.sub_control_points_num)
        print(self.spline_num)
        self.x_num = len(self.constraints_convexs) - (self.spline_num - 1)# 总的被优化变量数  (-除了第一条外的起点)      
        self.u = U_quasi_uniform(self.sub_control_points_num-1,self.p,1.) # n p 
    

        print(len(self.u))
        print(self.u)

        
        self.lbpcp = [] # 被优化变量下界 [box1 spline[(p1x,p1y),(),...],box2 spline[]]
        self.ubpcp = [] # 被优化变量下=上界 [box1 spline[(p1x,p1y),(),...],box2 spline[]]

        self.other_constraints = [] # 约束，比如点之间的连续性约束
        self.lbc = [] # 约束的下界
        self.ubc = [] # 约束的上界
        self.start = start # 3[(x,y),(dx,dy),(ddx,ddy)]
        self.end = end
        self.v_limit = v_limit
        self.a_limit = a_limit
        self.j_limit = j_limit

        self.vx = []
        self.ax = []
        self.vy = []
        self.ay = []
        self.jx = []
        self.jy = []
        # qnonuniform_clamped_bspline2D()

    def get_position_constraints(self):
        # 初态和终态位置
        # 初态和终态控制点为常量，应该剔除出优化
        # # 起点
        # self.lbx[0] = self.start[0][0]
        # self.ubx[0] = self.start[0][0]
        # self.lbx[self.np1] = self.start[0][1]
        # self.ubx[self.np1] = self.start[0][1]
        # # 终点
        # self.lbx[self.np1 - 1] = self.end[0][0]
        # self.ubx[self.np1 - 1] = self.end[0][0]
        # self.lbx[2*self.np1 - 1] = self.end[0][1]
        # self.ubx[2*self.np1 - 1] = self.end[0][1]

        # 点在凸多边形内部约束，逆时针将顶点和点所成直线叉乘，每个面积都为正>0写成Ax<b形式


        for j,poly in enumerate(self.constraints_convexs):
            which_bspline = math.floor((j+1)/self.sub_control_points_num) # 约束属于第几条样条 整型除法忽略浮点
            if which_bspline == 0:
                which_point = j
            else:
                which_point = (j+1) - which_bspline*self.sub_control_points_num # 属于这条的第几个点
            if which_bspline!=0 and which_point == 0:
                continue # 上条终点 为下条起点 不重复加约束
            print(which_bspline,which_point)
            print(poly)
            print(self.piecewise_control_points[which_bspline][which_point][0],self.piecewise_control_points[which_bspline][which_point][1])


            for i in range(1,len(poly)):
                # v1xv0 消除交叉项后的结果
                self.other_constraints.append(poly[i-1][0]*poly[i][1] -
                poly[i-1][1]*poly[i][0] +
                (poly[i][0] - poly[i-1][0])*self.piecewise_control_points[which_bspline][which_point][1]+
                (poly[i-1][1] - poly[i][1])*self.piecewise_control_points[which_bspline][which_point][0])

                self.lbc.append(0)
                self.ubc.append(float('inf'))
            # 最后一条边
            self.other_constraints.append(poly[-1][0]*poly[0][1] -
            poly[-1][1]*poly[0][0] +
            (poly[0][0] - poly[-1][0])*self.piecewise_control_points[which_bspline][which_point][1]+
            (poly[-1][1] - poly[0][1])*self.piecewise_control_points[which_bspline][which_point][0])
            self.lbc.append(0)
            self.ubc.append(float('inf'))
        print(self.other_constraints)
 

    def get_feasibility_constraints(self):
        # v
        # Vi = p(Qi+1 - Qi)/(ui+p+1 - ui+1)  uniform -- > (Qi+1 - Qi)/dt

        # 初态v
        print("********")

        self.vx.append([])
        self.vy.append([])

        self.other_constraints.append(self.p*(self.piecewise_control_points[0][0][0] - self.start[0][0])/(self.time_span[0]*(self.u[self.p + 1] - self.u[1])))
        self.lbc.append(self.start[1][0])
        self.ubc.append(self.start[1][0])       
        self.vx[-1].append(self.p*(self.piecewise_control_points[0][0][0] - self.start[0][0])/(self.time_span[0]*(self.u[self.p + 1] - self.u[1])))

        self.other_constraints.append(self.p*(self.piecewise_control_points[0][0][1] - self.start[0][1])/(self.time_span[0]*(self.u[self.p + 1] - self.u[1])))
        self.lbc.append(self.start[1][1])
        self.ubc.append(self.start[1][1])  
        self.vy[-1].append(self.p*(self.piecewise_control_points[0][0][1] - self.start[0][1])/(self.time_span[0]*(self.u[self.p + 1] - self.u[1])))

        for i in range(self.spline_num):
            if i == 0:
                d = 1
            else:
                d = 0
                self.vx.append([])
                self.vy.append([])

            for j in range(len(self.piecewise_control_points[i])-1):          
                # vx 
                self.vx[-1].append(self.p*(self.piecewise_control_points[i][j+1][0] - self.piecewise_control_points[i][j][0])/(self.time_span[i]*(self.u[j+d + self.p + 1] - self.u[j+d + 1])))
                self.other_constraints.append(self.p*(self.piecewise_control_points[i][j+1][0] - self.piecewise_control_points[i][j][0])/(self.time_span[i]*(self.u[j+d + self.p + 1] - self.u[j+d + 1])))
                self.lbc.append(-self.v_limit)
                self.ubc.append(self.v_limit)

                # vy 
                self.vy[-1].append(self.p*(self.piecewise_control_points[i][j+1][1] - self.piecewise_control_points[i][j][1])/(self.time_span[i]*(self.u[j+d + self.p + 1] - self.u[j+d + 1])))
                self.other_constraints.append(self.p*(self.piecewise_control_points[i][j+1][1] - self.piecewise_control_points[i][j][1])/(self.time_span[i]*(self.u[j+d + self.p + 1] - self.u[j+d + 1])))
                self.lbc.append(-self.v_limit)
                self.ubc.append(self.v_limit)   
        
        # 终态v
        self.vx[-1].append(self.p*(self.end[0][0] - self.piecewise_control_points[-1][-1][0])/(self.time_span[-1]*(self.u[len(self.piecewise_control_points[-1])-1 + self.p +1] - self.u[len(self.piecewise_control_points[-1])-1 + 1])))
        self.other_constraints.append(self.p*(self.end[0][0] - self.piecewise_control_points[-1][-1][0])/(self.time_span[-1]*(self.u[len(self.piecewise_control_points[-1])-1 + self.p +1] - self.u[len(self.piecewise_control_points[-1])-1 + 1])))
        self.lbc.append(self.end[1][0])
        self.ubc.append(self.end[1][0])       

        self.vy[-1].append(self.p*(self.end[0][1] - self.piecewise_control_points[-1][-1][1])/(self.time_span[-1]*(self.u[len(self.piecewise_control_points[-1])-1 + self.p +1] - self.u[len(self.piecewise_control_points[-1])-1 + 1])))
        self.other_constraints.append(self.p*(self.end[0][1] - self.piecewise_control_points[-1][-1][1])/(self.time_span[-1]*(self.u[len(self.piecewise_control_points[-1])-1 + self.p +1] - self.u[len(self.piecewise_control_points[-1])-1 + 1])))
        self.lbc.append(self.end[1][1])
        self.ubc.append(self.end[1][1])   

        print(self.vx)
        print(self.vy)
        print("********")

        # Ai = (p-1)(Vi+1 - Vi)/(ui+p+1 - ui+2)  uniform -- > (Vi+1 - Vi)/dt = (Qi+2 - 2Qi+1 + Qi)/dt^2
        # 初态a

        self.ax.append([])
        self.ay.append([])

        self.ax[-1].append((self.p - 1)*(self.vx[0][1] - self.vx[0][0])/(self.time_span[0]*(self.u[self.p + 1] - self.u[2])))
        self.other_constraints.append((self.p - 1)*(self.vx[0][1] - self.vx[0][0])/(self.time_span[0]*(self.u[self.p + 1] - self.u[2])))
        self.lbc.append(self.start[2][0])
        self.ubc.append(self.start[2][0])    

        self.ay[-1].append((self.p - 1)*(self.vy[0][1] - self.vy[0][0])/(self.time_span[0]*(self.u[self.p + 1] - self.u[2])))
        self.other_constraints.append((self.p - 1)*(self.vy[0][1] - self.vy[0][0])/(self.time_span[0]*(self.u[self.p + 1] - self.u[2])))
        self.lbc.append(self.start[2][1])
        self.ubc.append(self.start[2][1])  

        for i in range(self.spline_num):
            if i == 0:
                s = 1
            else:
                s = 0
                self.ax.append([])
                self.ay.append([])

            if i == self.spline_num-1:
                e = len(self.vx[i])-2
            else:
                e = len(self.vx[i])-1
            
            for j in range(s,e):
                # Ax 
                self.ax[-1].append((self.p - 1)*(self.vx[i][j+1] - self.vx[i][j])/((self.time_span[i]*(self.u[j+ self.p + 1] - self.u[j + 2]))))
                self.other_constraints.append((self.p - 1)*(self.vx[i][j+1] - self.vx[i][j])/((self.time_span[i]*(self.u[j + self.p + 1] - self.u[j + 2]))))
                self.lbc.append(-self.a_limit)
                self.ubc.append(self.a_limit)
                # Ay 
                self.ay[-1].append((self.p - 1)*(self.vy[i][j+1] - self.vy[i][j])/((self.time_span[i]*(self.u[j + self.p + 1] - self.u[j+ 2]))))
                self.other_constraints.append((self.p - 1)*(self.vy[i][j+1] - self.vy[i][j])/((self.time_span[i]*(self.u[j + self.p + 1] - self.u[j+ 2]))))
                self.lbc.append(-self.a_limit)
                self.ubc.append(self.a_limit)

        self.ax[-1].append((self.p - 1)*(self.vx[-1][-1] - self.vx[-1][-2])/((self.time_span[-1]*(self.u[len(self.vx[-1])-2 + self.p + 1] - self.u[len(self.vx[-1])-2 + 2]))))
        self.other_constraints.append((self.p - 1)*(self.vx[-1][-1] - self.vx[-1][-2])/((self.time_span[-1]*(self.u[len(self.vx[-1])-2 + self.p + 1] - self.u[len(self.vx[-1])-2 + 2]))))
        self.lbc.append(self.end[2][0])
        self.ubc.append(self.end[2][0])  

        self.ay[-1].append((self.p - 1)*(self.vy[-1][-1] - self.vy[-1][-2])/((self.time_span[-1]*(self.u[len(self.vy[-1])-2 + self.p + 1] - self.u[len(self.vy[-1])-2 + 2]))))
        self.other_constraints.append((self.p - 1)*(self.vy[-1][-1] - self.vy[-1][-2])/((self.time_span[-1]*(self.u[len(self.vy[-1])-2 + self.p + 1] - self.u[len(self.vy[-1])-2 + 2]))))
        self.lbc.append(self.end[2][1])
        self.ubc.append(self.end[2][1])       

        print(self.ax)
        print(self.ay)
        print("********")

        # jerk
        # 初态jerk都为0
        # Ji = (p-2)(Ai+1 - Ai)/(ui+p+1 - ui+3)  uniform -- > (Ai+1 - Ai)/dt = (Qi+3 - 3Qi+2 + 3Qi+1 - Qi)/dt^3
        for i in range(self.spline_num):
            self.jx.append([])
            self.jy.append([])
            for j in range(0,len(self.ax[i])-1):
                # jx 
                self.jx[-1].append((self.p - 2)*(self.ax[i][j+1] - self.ax[i][j])/(self.time_span[i]*(self.u[j+self.p+1] - self.u[j+3])))
                self.other_constraints.append((self.p - 2)*(self.ax[i][j+1] - self.ax[i][j])/(self.time_span[i]*(self.u[j+self.p+1] - self.u[j+3])))
                self.lbc.append(-self.j_limit)
                self.ubc.append(self.j_limit)

                # jy 
                self.jy[-1].append((self.p - 2)*(self.ay[i][j+1] - self.ay[i][j])/(self.time_span[i]*(self.u[j+self.p+1] - self.u[j+3])))
                self.other_constraints.append((self.p - 2)*(self.ay[i][j+1] - self.ay[i][j])/(self.time_span[i]*(self.u[j+self.p+1] - self.u[j+3])))
                self.lbc.append(-self.j_limit)
                self.ubc.append(self.j_limit)
        print(self.jx)
        print(self.jy)

    def get_continuity_constraints(self):

        # 上条的终点性质要和下条的一致，保证到jerk都是连续的
        for i in range(self.spline_num-1):
            # self.other_constraints.append(self.piecewise_control_points[i+1][0][0] - self.piecewise_control_points[i][-1][0])
            # self.lbc.append(0)
            # self.ubc.append(0)
            # self.other_constraints.append(self.piecewise_control_points[i+1][0][1] - self.piecewise_control_points[i][-1][1])
            # self.lbc.append(0)
            # self.ubc.append(0)

            self.other_constraints.append(self.vx[i+1][0] - self.vx[i][-1])
            self.lbc.append(0)
            self.ubc.append(0)
            self.other_constraints.append(self.vy[i+1][0] - self.vy[i][-1])
            self.lbc.append(0)
            self.ubc.append(0)

            self.other_constraints.append(self.ax[i+1][0] - self.ax[i][-1])
            self.lbc.append(0)
            self.ubc.append(0)
            self.other_constraints.append(self.ay[i+1][0] - self.ay[i][-1])
            self.lbc.append(0)
            self.ubc.append(0)

            self.other_constraints.append(self.jx[i+1][0] - self.jx[i][-1])
            self.lbc.append(0)
            self.ubc.append(0)
            self.other_constraints.append(self.jy[i+1][0] - self.jy[i][-1])
            self.lbc.append(0)
            self.ubc.append(0)

    def formulate(self):
        # 只加上条的终点，下条的起点不算
        for j in range(1,self.spline_num+1):
            self.piecewise_control_points.append([])
            self.lbpcp.append([])
            self.ubpcp.append([])
            if j == 1:
                s = 2
                e = self.sub_control_points_num + 1
            elif j == self.spline_num:
                s = 2
                e = self.sub_control_points_num
                self.piecewise_control_points[-1].append(self.piecewise_control_points[-2][-1])# 上条终点为本条起点
                self.lbpcp[-1].append(self.lbpcp[-2][-1])
                self.ubpcp[-1].append(self.lbpcp[-2][-1])
            else:
                s = 2
                e = self.sub_control_points_num + 1
                self.piecewise_control_points[-1].append(self.piecewise_control_points[-2][-1])# 上条终点为本条起点
                self.lbpcp[-1].append(self.lbpcp[-2][-1])
                self.ubpcp[-1].append(self.lbpcp[-2][-1])

            for i in range(s,e):
                self.piecewise_control_points[-1].append((SX.sym('x_' +str(j)+ str(i)),
                SX.sym('y_' +str(j)+ str(i))))
                # self.lbpcp[-1].append((-float('inf'),-float('inf')))
                # self.ubpcp[-1].append((float('inf'),float('inf')))
                self.lbpcp[-1].append((-30.,-30.))
                self.ubpcp[-1].append((30.,30.))


        # 每条的起点约束其实可以不加，可以由上条的终点+连续性约束决定
        self.get_position_constraints()
        self.get_feasibility_constraints() # 有问题
        self.get_continuity_constraints()

        # for i in range(self.n):
        #     print(str(i)+":")
        #     print(self.lbx[i],self.lbx[self.n+i],self.lbx[2*self.n+i])
        #     print(self.ubx[i],self.ubx[self.n+i],self.ubx[2*self.n+i])
       
        # 光滑代价 初态a和终态a为约束无法改变
        # for i in range(self.spline_num):# self.spline_num
        #     for j in range(len(self.ax[i])):
        #         self.cost += self.ax[i][j]**2
        #         self.cost_aim.append(self.ax[i][j]**2)
        #     for j in range(len(self.ay[i])):
        #         self.cost += self.ay[i][j]**2
        #         self.cost_aim.append(self.ay[i][j]**2)

        for i in range(self.spline_num):# self.spline_num
            for j in range(len(self.jx[i])):
                self.cost += self.jx[i][j]**2
                self.cost_aim.append(self.jx[i][j]**2)
            for j in range(len(self.jy[i])):
                self.cost += self.jy[i][j]**2
                self.cost_aim.append(self.jy[i][j]**2)

        # for i in range(self.spline_num):# self.spline_num
        #     for j in range(len(self.vx[i])):
        #         self.cost += self.vx[i][j]**2
        #         self.cost_aim.append(self.vx[i][j]**2)


        # for i in range(len(self.jx)):
        #     self.cost += 0.5*self.jx[i]**2
        #     self.cost_aim.append(1*self.jx[i]**2)
        # for i in range(len(self.jy)):
        #     self.cost += 0.5*self.jy[i]**2
        #     self.cost_aim.append(1*self.jy[i]**2)

        # for i in range(len(self.ax)):
        #     self.cost += self.ax[i]**2
        #     self.cost_aim.append(self.ax[i]**2)
        # for i in range(len(self.ay)):
        #     self.cost += self.ay[i]**2
        #     self.cost_aim.append(self.ay[i]**2)


        x = []
        lbx = []
        ubx = []
        y = []
        lby = []
        uby = []
        for i in range(len(self.piecewise_control_points)):
            for j in range(len(self.piecewise_control_points[i])):
                if i!=0 and j==0:
                    continue # 上条的终点就是下条的起点 不需要额外再添加了
                x.append(self.piecewise_control_points[i][j][0])
                y.append(self.piecewise_control_points[i][j][1])
                lbx.append(self.lbpcp[i][j][0])
                lby.append(self.lbpcp[i][j][1])
                ubx.append(self.ubpcp[i][j][0])
                uby.append(self.ubpcp[i][j][1])

        x.extend(y)
        lbx.extend(lby)
        ubx.extend(uby)

        qp = {'x':vertcat(*x), 'f':self.cost, 'g':vertcat(*self.other_constraints)}
        # Solve with 
        solver = qpsol('solver', 'qpoases', qp, {'sparse':True} )# printLevel':'none' 

        # return None,None,None
        # Get the optimal solution
        sol = solver(lbx=lbx, ubx=ubx, lbg=self.lbc, ubg=self.ubc)
        
        print(sol)
        # 分段b样条二维控制点 
        x_opt = [[self.start[0][0]]]
        y_opt = [[self.start[0][1]]]
        for j in range(self.sub_control_points_num-1):
            x_opt[-1].append(float(sol['x'][j]))
            y_opt[-1].append(float(sol['x'][j+self.x_num]))
        x_opt.append([])
        y_opt.append([])

        # 直接把起点终点加进去
        for i in range(1,self.spline_num-1): # remove DM attr
            x_opt[-1].append(x_opt[-2][-1]) # 上条终点作为本条起点
            y_opt[-1].append(y_opt[-2][-1]) # 上条终点作为本条起点
            for j in range(self.sub_control_points_num-1):
                # 起点被省略
                x_opt[-1].append(float(sol['x'][self.sub_control_points_num - 1 + (i-1)*(self.sub_control_points_num-1)+j]))
                y_opt[-1].append(float(sol['x'][self.sub_control_points_num - 1 + (i-1)*(self.sub_control_points_num-1)+j+self.x_num]))

            x_opt.append([])
            y_opt.append([])

        x_opt[-1].append(x_opt[-2][-1]) # 上条终点作为本条起点
        y_opt[-1].append(y_opt[-2][-1]) # 上条终点作为本条起点
        for j in range(self.sub_control_points_num-2):
            x_opt[-1].append(float(sol['x'][self.sub_control_points_num - 1 + (self.spline_num-1-1)*(self.sub_control_points_num-1)+j]))
            y_opt[-1].append(float(sol['x'][self.sub_control_points_num - 1 + (self.spline_num-1-1)*(self.sub_control_points_num-1)+j+self.x_num]))

        x_opt[-1].append(self.end[0][0])
        y_opt[-1].append(self.end[0][1])

        print('opt cost = ', sol['f'])
        print(x_opt)
        print(y_opt)
        
        return x_opt,y_opt,self.start_time





def corridorBuilder2d(origin_x, origin_y, radius,scan):
    safe_ridus = radius
    flip_data = []
    for p in scan:
        dx = p[0] - origin_x
        dy = p[1] - origin_y
        norm2 = math.hypot(dx,dy)
        if norm2 < safe_ridus: # 点云数据中的最小距离
            safe_ridus = norm2
        if norm2 < 1e-6: # 5cm
            continue
        flip_data.append((dx+2*(radius - norm2)*dx/norm2,dy+2*(radius-norm2)*dy/norm2))
    _,vertex_indexs = Graham_Andrew_Scan(flip_data)
    
    isOriginAVertex = False
    OriginIndex = -1
    vertexs = []
    for i in range(len(vertex_indexs)):
        if vertex_indexs[i] == len(scan):
            isOriginAVertex = True
            OriginIndex =   i
            vertexs.append((origin_x,origin_y))
        else:
            vertexs.append(scan[vertex_indexs[i]])    
            
    if isOriginAVertex:
        last_index = (OriginIndex-1)%len(vertexs)
        nxt_index =  (OriginIndex+1)%len(vertexs)
        dx = (scan[vertex_indexs[last_index]][0] + origin_x + scan[vertex_indexs[nxt_index]][0])/3. - origin_x
        dx = (scan[vertex_indexs[last_index]][1] + origin_y + scan[vertex_indexs[nxt_index]][1])/3. - origin_y

        d = math.hypot(dx,dy)
        interior_x = 0.99*safe_ridus*dx/d+ origin_x
        interior_y = 0.99*safe_ridus*dy/d+ origin_y        
    else:
        interior_x = origin_x
        interior_y = origin_y

    _,indexs = Graham_Andrew_Scan(vertexs) # 要求逆时针
    constraints = [] # (a,b,c) a x + b y <= c
    for j in range(len(indexs)):
        jplus1 = (j+1)%len(indexs)
        rayV = vertexs[indexs[jplus1]] - vertexs[indexs[j]]
        normalj = (rayV[1],-rayV[0]) # point to outside
        norm = math.hypot(normalj)
        normalj = (rayV[1]/norm,-rayV[0]/norm)
        ij = indexs[j]

        while ij != indexs[jplus1]:
            c = (vertexs[ij][0] - interior_x)*normalj[0] + (vertexs[ij][1] - interior_y)*normalj[1]
            constraints.append(normalj[0],normalj[1],c)
            ij = (ij+1)%len(vertexs)
    dual_points = []
    for c in constraints:
        dual_points.append((c[0]/c[2],c[1]/c[2]))
    
    dual_verterx =  reverse(Graham_Andrew_Scan(dual_points)[1])# 顺时针
    final_vertex = []
    for i in range(len(dual_verterx)):
        iplus1 = (i+1)%len(dual_verterx)
        rayi =(dual_verterx[iplus1][0] - dual_verterx[i][0],dual_verterx[iplus1][1] - dual_verterx[i][1])
        c = rayi[1]*dual_verterx[i][0] - rayi[0]*dual_verterx[i][1]
        final_vertex.append((interior_x + rayi[1]/c,interior_y - rayi[0]/c))
    output_constraints = []
    for i in range(final_vertex):
        iplus1 = (i+1)%len(final_vertex)
        rayi = (final_vertex[iplus1][0] - final_vertex[i][0],final_vertex[iplus1][1] - final_vertex[i][1])
        c = rayi[1]*final_vertex[i][0] - rayi[0]*final_vertex[i][1]
        output_constraints.append((rayi.y,-rayi.x,c))

    return output_constraints,final_vertex


# class quniform_piecewise_bspline_optim:
#     def __init__(self,one_spline_ctrl_points_num,convex_polygons,start,end,degrees,time_span):
#         if len(convex_polygons) < 2:
#             return
#         self.piecewise_control_points = [] # [box1 spline[(p1x,p1y),(),...],box2 spline[]]
#         self.cost_aim = [] # 
#         self.v_weight = 1.
#         self.a_weight = 1.
#         self.j_weight = 1.
#         self.cost = 0.

#         # time_span轨迹段数，总的轨迹点数是len(convex_polygons) + 2,段数是len(convex_polygons) + 1
#         self.p = degrees
#         # 交叠区域控制点的重复度也为p，要求必须经过这个点，否则还是会出现样条曲线在约束外的情况
        
#         self.time_span = time_span 
#         self.start_time = [0] # 每条b样条的真实起始时间
#         for i in range(len(time_span)):
#             self.start_time.append(self.start_time[-1]+time_span[i])

#         # convex_polygons约束多边形，除了初终态，其他控制点都被约束在某个多边形内或两个凸多边形的交叉部分内
#         # convex_polygons保存了每个约束多边形
#         # 传入的convex_polygons就是约束空间不含交叠区域，需要自己算
#         # convex_polygons [(x,y,w,h),(),...]
#         self.constraints_convexs = []
#         prev_overlap = None
#         ld = (convex_polygons[0][0]-convex_polygons[0][2]/2.,
#             convex_polygons[0][1]-convex_polygons[0][3]/2.)
#         rd = (convex_polygons[0][0]+convex_polygons[0][2]/2.,
#             convex_polygons[0][1]-convex_polygons[0][3]/2.)
#         ru = (convex_polygons[0][0]+convex_polygons[0][2]/2.,
#             convex_polygons[0][1]+convex_polygons[0][3]/2.)
#         lu = (convex_polygons[0][0]-convex_polygons[0][2]/2.,
#             convex_polygons[0][1]+convex_polygons[0][3]/2.)
#         for i in range(len(convex_polygons)-1):
#             # 逆时针，左下开始
#             ldn = (convex_polygons[i+1][0]-convex_polygons[i+1][2]/2.,
#                 convex_polygons[i+1][1]-convex_polygons[i+1][3]/2.)
#             rdn = (convex_polygons[i+1][0]+convex_polygons[i+1][2]/2.,
#                 convex_polygons[i+1][1]-convex_polygons[i+1][3]/2.)
#             run = (convex_polygons[i+1][0]+convex_polygons[i+1][2]/2.,
#                 convex_polygons[i+1][1]+convex_polygons[i+1][3]/2.)
#             lun = (convex_polygons[i+1][0]-convex_polygons[i+1][2]/2.,
#                 convex_polygons[i+1][1]+convex_polygons[i+1][3]/2.)            

#             # 初态overlap
#             if prev_overlap is not None:
#                 self.constraints_convexs.append(prev_overlap)
            
#             # 第一个在上个overlap,最后一个点在下个overlap都不在这个范围内
#             for j in range(1,one_spline_ctrl_points_num-1):
#                 self.constraints_convexs.append([ld,rd,ru,lu])
            
#             # 计算当前box和下个box的交叉部分
#             overlap = [(max(ld[0],ldn[0]),max(ld[1],ldn[1])),(min(rd[0],rdn[0]),max(rd[1],rdn[1]))
#             ,(min(ru[0],run[0]),min(ru[1],run[1])),(max(lu[0],lun[0]),min(lu[1],lun[1]))]
#             # 此box内最后一个状态点的约束
#             self.constraints_convexs.append(overlap)
#             ld = ldn
#             rd = rdn
#             ru = run
#             lu = lun
#             prev_overlap = overlap
#         # 最后一个box
#         self.constraints_convexs.append(prev_overlap)
#         # 第一个在上个overlap,最后一个点在下个overlap都不在这个范围内
#         for j in range(1,one_spline_ctrl_points_num-1):
#             self.constraints_convexs.append([ld,rd,ru,lu])        


#         self.control_points_num = len(self.constraints_convexs) + 2# 总的控制点数
#         self.sub_control_points_num = one_spline_ctrl_points_num# 每段的控制点数
#         self.spline_num = int(self.control_points_num/self.sub_control_points_num)
#         print(self.spline_num)
#         self.x_num = len(self.constraints_convexs)# 总的被优化变量数        
#         self.u = U_quasi_uniform(self.sub_control_points_num-1,self.p,1.) # n p 

#         print(len(self.u))
#         print(self.u)

        
#         self.lbpcp = [] # 被优化变量下界 [box1 spline[(p1x,p1y),(),...],box2 spline[]]
#         self.ubpcp = [] # 被优化变量下=上界 [box1 spline[(p1x,p1y),(),...],box2 spline[]]

#         self.other_constraints = [] # 约束，比如点之间的连续性约束
#         self.lbc = [] # 约束的下界
#         self.ubc = [] # 约束的上界
#         self.start = start # 3[(x,y),(dx,dy),(ddx,ddy)]
#         self.end = end
#         self.v_limit = 3.5
#         self.a_limit = 8.
#         self.j_limit = 3000.

#         self.vx = []
#         self.ax = []
#         self.vy = []
#         self.ay = []
#         self.jx = []
#         self.jy = []
#         # qnonuniform_clamped_bspline2D()

#     def get_position_constraints(self):
#         # 初态和终态位置
#         # 初态和终态控制点为常量，应该剔除出优化
#         # # 起点
#         # self.lbx[0] = self.start[0][0]
#         # self.ubx[0] = self.start[0][0]
#         # self.lbx[self.np1] = self.start[0][1]
#         # self.ubx[self.np1] = self.start[0][1]
#         # # 终点
#         # self.lbx[self.np1 - 1] = self.end[0][0]
#         # self.ubx[self.np1 - 1] = self.end[0][0]
#         # self.lbx[2*self.np1 - 1] = self.end[0][1]
#         # self.ubx[2*self.np1 - 1] = self.end[0][1]

#         # 点在凸多边形内部约束，逆时针将顶点和点所成直线叉乘，每个面积都为正>0写成Ax<b形式


#         for j,poly in enumerate(self.constraints_convexs):
#             which_bspline = math.floor((j+1)/self.sub_control_points_num) # 约束属于第几条样条 整型除法忽略浮点
#             if which_bspline == 0:
#                 which_point = j
#             else:
#                 which_point = (j+1) - which_bspline*self.sub_control_points_num # 属于这条的第几个点

#             print(which_bspline,which_point)
#             print(poly)
#             print(self.piecewise_control_points[which_bspline][which_point][0],self.piecewise_control_points[which_bspline][which_point][1])


#             for i in range(1,len(poly)):
#                 # v1xv0 消除交叉项后的结果
#                 self.other_constraints.append(poly[i-1][0]*poly[i][1] -
#                 poly[i-1][1]*poly[i][0] +
#                 (poly[i][0] - poly[i-1][0])*self.piecewise_control_points[which_bspline][which_point][1]+
#                 (poly[i-1][1] - poly[i][1])*self.piecewise_control_points[which_bspline][which_point][0])

#                 self.lbc.append(0)
#                 self.ubc.append(float('inf'))
#             # 最后一条边
#             self.other_constraints.append(poly[-1][0]*poly[0][1] -
#             poly[-1][1]*poly[0][0] +
#             (poly[0][0] - poly[-1][0])*self.piecewise_control_points[which_bspline][which_point][1]+
#             (poly[-1][1] - poly[0][1])*self.piecewise_control_points[which_bspline][which_point][0])
#             self.lbc.append(0)
#             self.ubc.append(float('inf'))
#         print(self.other_constraints)
 

#     def get_feasibility_constraints(self):
#         # v
#         # Vi = p(Qi+1 - Qi)/(ui+p+1 - ui+1)  uniform -- > (Qi+1 - Qi)/dt

#         # 初态v
#         print("********")

#         self.vx.append([])
#         self.vy.append([])

#         self.other_constraints.append(self.p*(self.piecewise_control_points[0][0][0] - self.start[0][0])/(self.time_span[0]*(self.u[self.p + 1] - self.u[1])))
#         self.lbc.append(self.start[1][0])
#         self.ubc.append(self.start[1][0])       
#         self.vx[-1].append(self.p*(self.piecewise_control_points[0][0][0] - self.start[0][0])/(self.time_span[0]*(self.u[self.p + 1] - self.u[1])))

#         self.other_constraints.append(self.p*(self.piecewise_control_points[0][0][1] - self.start[0][1])/(self.time_span[0]*(self.u[self.p + 1] - self.u[1])))
#         self.lbc.append(self.start[1][1])
#         self.ubc.append(self.start[1][1])  
#         self.vy[-1].append(self.p*(self.piecewise_control_points[0][0][1] - self.start[0][1])/(self.time_span[0]*(self.u[self.p + 1] - self.u[1])))

#         for i in range(self.spline_num):
#             if i == 0:
#                 d = 1
#             else:
#                 d = 0
#                 self.vx.append([])
#                 self.vy.append([])

#             for j in range(len(self.piecewise_control_points[i])-1):          
#                 # vx 
#                 self.vx[-1].append(self.p*(self.piecewise_control_points[i][j+1][0] - self.piecewise_control_points[i][j][0])/(self.time_span[i]*(self.u[j+d + self.p + 1] - self.u[j+d + 1])))
#                 self.other_constraints.append(self.p*(self.piecewise_control_points[i][j+1][0] - self.piecewise_control_points[i][j][0])/(self.time_span[i]*(self.u[j+d + self.p + 1] - self.u[j+d + 1])))
#                 self.lbc.append(-self.v_limit)
#                 self.ubc.append(self.v_limit)

#                 # vy 
#                 self.vy[-1].append(self.p*(self.piecewise_control_points[i][j+1][1] - self.piecewise_control_points[i][j][1])/(self.time_span[i]*(self.u[j+d + self.p + 1] - self.u[j+d + 1])))
#                 self.other_constraints.append(self.p*(self.piecewise_control_points[i][j+1][1] - self.piecewise_control_points[i][j][1])/(self.time_span[i]*(self.u[j+d + self.p + 1] - self.u[j+d + 1])))
#                 self.lbc.append(-self.v_limit)
#                 self.ubc.append(self.v_limit)   
        
#         # 终态v
#         self.vx[-1].append(self.p*(self.end[0][0] - self.piecewise_control_points[-1][-1][0])/(self.time_span[-1]*(self.u[len(self.piecewise_control_points[-1])-1 + self.p +1] - self.u[len(self.piecewise_control_points[-1])-1 + 1])))
#         self.other_constraints.append(self.p*(self.end[0][0] - self.piecewise_control_points[-1][-1][0])/(self.time_span[-1]*(self.u[len(self.piecewise_control_points[-1])-1 + self.p +1] - self.u[len(self.piecewise_control_points[-1])-1 + 1])))
#         self.lbc.append(self.end[1][0])
#         self.ubc.append(self.end[1][0])       

#         self.vy[-1].append(self.p*(self.end[0][1] - self.piecewise_control_points[-1][-1][1])/(self.time_span[-1]*(self.u[len(self.piecewise_control_points[-1])-1 + self.p +1] - self.u[len(self.piecewise_control_points[-1])-1 + 1])))
#         self.other_constraints.append(self.p*(self.end[0][1] - self.piecewise_control_points[-1][-1][1])/(self.time_span[-1]*(self.u[len(self.piecewise_control_points[-1])-1 + self.p +1] - self.u[len(self.piecewise_control_points[-1])-1 + 1])))
#         self.lbc.append(self.end[1][1])
#         self.ubc.append(self.end[1][1])   

#         print(self.vx)
#         print(self.vy)
#         print("********")

#         # Ai = (p-1)(Vi+1 - Vi)/(ui+p+1 - ui+2)  uniform -- > (Vi+1 - Vi)/dt = (Qi+2 - 2Qi+1 + Qi)/dt^2
#         # 初态a

#         self.ax.append([])
#         self.ay.append([])

#         self.ax[-1].append((self.p - 1)*(self.vx[0][1] - self.vx[0][0])/(self.time_span[0]*(self.u[self.p + 1] - self.u[2])))
#         self.other_constraints.append((self.p - 1)*(self.vx[0][1] - self.vx[0][0])/(self.time_span[0]*(self.u[self.p + 1] - self.u[2])))
#         self.lbc.append(self.start[2][0])
#         self.ubc.append(self.start[2][0])    

#         self.ay[-1].append((self.p - 1)*(self.vy[0][1] - self.vy[0][0])/(self.time_span[0]*(self.u[self.p + 1] - self.u[2])))
#         self.other_constraints.append((self.p - 1)*(self.vy[0][1] - self.vy[0][0])/(self.time_span[0]*(self.u[self.p + 1] - self.u[2])))
#         self.lbc.append(self.start[2][1])
#         self.ubc.append(self.start[2][1])  

#         for i in range(self.spline_num):
#             if i == 0:
#                 s = 1
#             else:
#                 s = 0
#                 self.ax.append([])
#                 self.ay.append([])

#             if i == self.spline_num-1:
#                 e = len(self.vx[i])-2
#             else:
#                 e = len(self.vx[i])-1
            
#             for j in range(s,e):
#                 # Ax 
#                 self.ax[-1].append((self.p - 1)*(self.vx[i][j+1] - self.vx[i][j])/((self.time_span[i]*(self.u[j+ self.p + 1] - self.u[j + 2]))))
#                 self.other_constraints.append((self.p - 1)*(self.vx[i][j+1] - self.vx[i][j])/((self.time_span[i]*(self.u[j + self.p + 1] - self.u[j + 2]))))
#                 self.lbc.append(-self.a_limit)
#                 self.ubc.append(self.a_limit)
#                 # Ay 
#                 self.ay[-1].append((self.p - 1)*(self.vy[i][j+1] - self.vy[i][j])/((self.time_span[i]*(self.u[j + self.p + 1] - self.u[j+ 2]))))
#                 self.other_constraints.append((self.p - 1)*(self.vy[i][j+1] - self.vy[i][j])/((self.time_span[i]*(self.u[j + self.p + 1] - self.u[j+ 2]))))
#                 self.lbc.append(-self.a_limit)
#                 self.ubc.append(self.a_limit)

#         self.ax[-1].append((self.p - 1)*(self.vx[-1][-1] - self.vx[-1][-2])/((self.time_span[-1]*(self.u[len(self.vx[-1])-2 + self.p + 1] - self.u[len(self.vx[-1])-2 + 2]))))
#         self.other_constraints.append((self.p - 1)*(self.vx[-1][-1] - self.vx[-1][-2])/((self.time_span[-1]*(self.u[len(self.vx[-1])-2 + self.p + 1] - self.u[len(self.vx[-1])-2 + 2]))))
#         self.lbc.append(self.end[2][0])
#         self.ubc.append(self.end[2][0])  

#         self.ay[-1].append((self.p - 1)*(self.vy[-1][-1] - self.vy[-1][-2])/((self.time_span[-1]*(self.u[len(self.vy[-1])-2 + self.p + 1] - self.u[len(self.vy[-1])-2 + 2]))))
#         self.other_constraints.append((self.p - 1)*(self.vy[-1][-1] - self.vy[-1][-2])/((self.time_span[-1]*(self.u[len(self.vy[-1])-2 + self.p + 1] - self.u[len(self.vy[-1])-2 + 2]))))
#         self.lbc.append(self.end[2][1])
#         self.ubc.append(self.end[2][1])       

#         print(self.ax)
#         print(self.ay)
#         print("********")

#         # jerk
#         # 初态jerk都为0
#         # Ji = (p-2)(Ai+1 - Ai)/(ui+p+1 - ui+3)  uniform -- > (Ai+1 - Ai)/dt = (Qi+3 - 3Qi+2 + 3Qi+1 - Qi)/dt^3
#         for i in range(self.spline_num):
#             self.jx.append([])
#             self.jy.append([])
#             for j in range(0,len(self.ax[i])-1):
#                 # jx 
#                 self.jx[-1].append((self.p - 2)*(self.ax[i][j+1] - self.ax[i][j])/(self.time_span[i]*(self.u[j+self.p+1] - self.u[j+3])))
#                 self.other_constraints.append((self.p - 2)*(self.ax[i][j+1] - self.ax[i][j])/(self.time_span[i]*(self.u[j+self.p+1] - self.u[j+3])))
#                 self.lbc.append(-self.j_limit)
#                 self.ubc.append(self.j_limit)

#                 # jy 
#                 self.jy[-1].append((self.p - 2)*(self.ay[i][j+1] - self.ay[i][j])/(self.time_span[i]*(self.u[j+self.p+1] - self.u[j+3])))
#                 self.other_constraints.append((self.p - 2)*(self.ay[i][j+1] - self.ay[i][j])/(self.time_span[i]*(self.u[j+self.p+1] - self.u[j+3])))
#                 self.lbc.append(-self.j_limit)
#                 self.ubc.append(self.j_limit)
#         print(self.jx)
#         print(self.jy)

#     def get_continuity_constraints(self):

#         # 上条的终点性质要和下条的一致，保证到jerk都是连续的
#         for i in range(self.spline_num-1):
#             self.other_constraints.append(self.piecewise_control_points[i+1][0][0] - self.piecewise_control_points[i][-1][0])
#             self.lbc.append(0)
#             self.ubc.append(0)
#             self.other_constraints.append(self.piecewise_control_points[i+1][0][1] - self.piecewise_control_points[i][-1][1])
#             self.lbc.append(0)
#             self.ubc.append(0)

#             self.other_constraints.append(self.vx[i+1][0] - self.vx[i][-1])
#             self.lbc.append(0)
#             self.ubc.append(0)
#             self.other_constraints.append(self.vy[i+1][0] - self.vy[i][-1])
#             self.lbc.append(0)
#             self.ubc.append(0)

#             self.other_constraints.append(self.ax[i+1][0] - self.ax[i][-1])
#             self.lbc.append(0)
#             self.ubc.append(0)
#             self.other_constraints.append(self.ay[i+1][0] - self.ay[i][-1])
#             self.lbc.append(0)
#             self.ubc.append(0)

#             self.other_constraints.append(self.jx[i+1][0] - self.jx[i][-1])
#             self.lbc.append(0)
#             self.ubc.append(0)
#             self.other_constraints.append(self.jy[i+1][0] - self.jy[i][-1])
#             self.lbc.append(0)
#             self.ubc.append(0)

#     def formulate(self):

#         for j in range(1,self.spline_num+1):
#             self.piecewise_control_points.append([])
#             self.lbpcp.append([])
#             self.ubpcp.append([])
#             if j == 1:
#                 s = 2
#                 e = self.sub_control_points_num + 1
#             elif j == self.spline_num:
#                 s = 1
#                 e = self.sub_control_points_num
#             else:
#                 s = 1
#                 e = self.sub_control_points_num + 1
#             for i in range(s,e):
#                 self.piecewise_control_points[-1].append((SX.sym('x_' +str(j)+ str(i)),
#                 SX.sym('y_' +str(j)+ str(i))))
#                 # self.lbpcp[-1].append((-float('inf'),-float('inf')))
#                 # self.ubpcp[-1].append((float('inf'),float('inf')))
#                 self.lbpcp[-1].append((-30.,-30.))
#                 self.ubpcp[-1].append((30.,30.))


#         # 每条的起点约束其实可以不加，可以由上条的终点+连续性约束决定
#         self.get_position_constraints()
#         self.get_feasibility_constraints() # 有问题
#         self.get_continuity_constraints()

#         # for i in range(self.n):
#         #     print(str(i)+":")
#         #     print(self.lbx[i],self.lbx[self.n+i],self.lbx[2*self.n+i])
#         #     print(self.ubx[i],self.ubx[self.n+i],self.ubx[2*self.n+i])
       
#         # 光滑代价 初态a和终态a为约束无法改变
#         for i in range(self.spline_num):# self.spline_num
#             for j in range(len(self.ax[i])):
#                 self.cost += self.ax[i][j]**2
#                 self.cost_aim.append(self.ax[i][j]**2)
#             for j in range(len(self.ay[i])):
#                 self.cost += self.ay[i][j]**2
#                 self.cost_aim.append(self.ay[i][j]**2)

#         # for i in range(self.spline_num):# self.spline_num
#         #     for j in range(len(self.vx[i])):
#         #         self.cost += self.vx[i][j]**2
#         #         self.cost_aim.append(self.vx[i][j]**2)


#         # for i in range(len(self.jx)):
#         #     self.cost += 0.5*self.jx[i]**2
#         #     self.cost_aim.append(1*self.jx[i]**2)
#         # for i in range(len(self.jy)):
#         #     self.cost += 0.5*self.jy[i]**2
#         #     self.cost_aim.append(1*self.jy[i]**2)

#         # for i in range(len(self.ax)):
#         #     self.cost += self.ax[i]**2
#         #     self.cost_aim.append(self.ax[i]**2)
#         # for i in range(len(self.ay)):
#         #     self.cost += self.ay[i]**2
#         #     self.cost_aim.append(self.ay[i]**2)


#         x = []
#         lbx = []
#         ubx = []
#         y = []
#         lby = []
#         uby = []
#         for i in range(len(self.piecewise_control_points)):
#             for j in range(len(self.piecewise_control_points[i])):
#                 x.append(self.piecewise_control_points[i][j][0])
#                 y.append(self.piecewise_control_points[i][j][1])
#                 lbx.append(self.lbpcp[i][j][0])
#                 lby.append(self.lbpcp[i][j][1])
#                 ubx.append(self.ubpcp[i][j][0])
#                 uby.append(self.ubpcp[i][j][1])

#         x.extend(y)
#         lbx.extend(lby)
#         ubx.extend(uby)

#         qp = {'x':vertcat(*x), 'f':self.cost, 'g':vertcat(*self.other_constraints)}
#         # Solve with 
#         solver = qpsol('solver', 'qpoases', qp, {'sparse':True} )# printLevel':'none' 

#         # return None,None,None
#         # Get the optimal solution
#         sol = solver(lbx=lbx, ubx=ubx, lbg=self.lbc, ubg=self.ubc)
        
#         print(sol)
#         # 分段b样条二维控制点 
#         x_opt = [[self.start[0][0]]]
#         y_opt = [[self.start[0][1]]]
#         for j in range(self.sub_control_points_num-1):
#             x_opt[-1].append(float(sol['x'][j]))
#             y_opt[-1].append(float(sol['x'][j+self.x_num]))
#         x_opt.append([])
#         y_opt.append([])
#         # 直接把起点终点加进去
#         for i in range(1,self.spline_num-1): # remove DM attr

#             for j in range(self.sub_control_points_num):
#                 x_opt[-1].append(float(sol['x'][self.sub_control_points_num - 1 + (i-1)*self.sub_control_points_num+j]))
#                 y_opt[-1].append(float(sol['x'][self.sub_control_points_num - 1 + (i-1)*self.sub_control_points_num+j+self.x_num]))

#             x_opt.append([])
#             y_opt.append([])

#         for j in range(self.sub_control_points_num-1):
#             x_opt[-1].append(float(sol['x'][self.sub_control_points_num - 1 + (self.spline_num-1-1)*self.sub_control_points_num+j]))
#             y_opt[-1].append(float(sol['x'][self.sub_control_points_num - 1 + (self.spline_num-1-1)*self.sub_control_points_num+j+self.x_num]))

#         x_opt[-1].append(self.end[0][0])
#         y_opt[-1].append(self.end[0][1])

#         print('opt cost = ', sol['f'])
#         print(x_opt)
#         print(y_opt)
#         return x_opt,y_opt,self.start_time



# 1 时间分配
# def time_allocation(convex_polygons,max_vel,max_acc,init_state,end_state):
#     points = []
#     points.append((init_state[0][0],init_state[0][1]))
#     for c in convex_polygons:
#         points.append((c[0],c[1]))
#     points.append((end_state[0][0],end_state[0][1]))

#     vel = max_vel*0.6
#     acc = max_acc*0.6
#     times = []
#     t = 0
#     v0 = (0.,0.)
#     for i in range(len(points)-1):

#         d = (points[i+1][0] - points[i][0],points[i+1][1] - points[i][1])
#         d_norm = math.hypot(d[0],d[1])  # 两个box中心之间的距离
#         if i == 0:
#             v0 = (init_state[1][0],init_state[1][1]) # vx,0vy0
        
#         # V0为速度在d方向上的分量
#         V0 = (v0[0]*d[0]+v0[1]*d[1])/d_norm
#         aV0 = abs(V0)
#         acct = abs(vel-V0)/acc # 以acc为加速度从V0加或减速到vel所需要的时间
#         accd = V0*acct + (acc*acct*acct/2.)*(1 if vel > V0 else -1) # 以acc为加速度从V0加或减速到vel所需要位移
#         dcct = vel/acc # 从vel减速到0所需时间
#         dccd = acc*dcct*dcct/2.# 从vel减速到0所需要时间

#         # 梯形时间分配 都是从0开始加速然后减速到0
#         if d_norm < aV0*aV0/(2*acc):# box之间的距离小于以V0减速到0所需要距离，则选择直接减速
#             t1 = 2.*aV0/acc if V0 < 0 else 0.    # 以-V0加速到V0所需要的时间
#             t2 = aV0/acc    # 从V0减速到0所需要的时间
#             t = t1+t2
#         elif d_norm < accd + dccd:  # 距离不够加速到vel再从vel开始减速，所以需要在vel之前开始减速  三角形
#             t1 = 2.*aV0/acc if V0 < 0 else 0.  #  -V0加速到V0所需要时间
#             #  
#             t2 = (-aV0+math.sqrt(aV0*aV0/2. + acc*d_norm ))/acc # 
#             t3 = (aV0 + acc*t2)/acc # V0以acc加速t2的速度，再以acc减速到0所需时间
#             t = t1+t2+t3
#         else:
#             # 典型梯形
#             t1 = acct # 加速时间
#             t2 = (d_norm - accd - dccd)/vel # 匀速时间
#             t3 = dcct # 减速时间
#             t = t1+t2+t3
#         times.append(t)
#     return times

def time_allocation(max_vel,max_acc,init_state,end_state):
    vel = max_vel*0.6 # 巡航速度
    acc = max_acc*0.6

    t = 0
    v0 = (init_state[1][0],init_state[1][1]) # vx,0vy0
    d = (end_state[0][0] - init_state[0][0],end_state[0][1] - init_state[0][1])
    d_norm = math.hypot(d[0],d[1])  # 两个box中心之间的距离
        
    # V0为速度在d方向上的分量
    V0 = (v0[0]*d[0]+v0[1]*d[1])/d_norm
    aV0 = abs(V0)
    acct = abs(vel-V0)/acc # 以acc为加速度从V0加或减速到vel所需要的时间
    accd = V0*acct + (acc*acct*acct/2.)*(1 if vel > V0 else -1) # 以acc为加速度从V0加或减速到vel所需要位移
    dcct = vel/acc # 从vel减速到0所需时间
    dccd = acc*dcct*dcct/2.# 从vel减速到0所需要时间

    # 梯形时间分配 都是从0开始加速然后减速到0
    if d_norm < aV0*aV0/(2*acc):# box之间的距离小于以V0减速到0所需要距离，则选择直接减速
        t1 = 2.*aV0/acc if V0 < 0 else 0.    # 以-V0加速到V0所需要的时间
        t2 = aV0/acc    # 从V0减速到0所需要的时间
        t = t1+t2
    elif d_norm < accd + dccd:  # 距离不够加速到vel再从vel开始减速，所以需要在vel之前开始减速  三角形
        t1 = 2.*aV0/acc if V0 < 0 else 0.  #  -V0加速到V0所需要时间
        t2 = (-aV0+math.sqrt(aV0*aV0/2. + acc*d_norm ))/acc # 
        t3 = (aV0 + acc*t2)/acc # V0以acc加速t2的速度，再以acc减速到0所需时间
        t = t1+t2+t3
    else:
        # 典型梯形
        t1 = acct # 加速时间
        t2 = (d_norm - accd - dccd)/vel # 匀速时间
        t3 = dcct # 减速时间
        t = t1+t2+t3

    return t

def one_axis_time_allocation(curise_v,curise_acc,init_state,end_state):
    # x,y两轴分别计算时间。取较长时间作为分配时间，
    # 如果在较短时间内能够到达期望，则将时间拉长依旧可以达到期望

    # 1 如果存在横向速度或与行驶方向相反的纵向速度都先减速到0
    v0 = (init_state[1][0],init_state[1][1]) # vx,0vy0
    d = (end_state[0][0] - init_state[0][0],end_state[0][1] - init_state[0][1])
    d_norm = math.hypot(d[0],d[1])
    vs = (v0[0]*d[0]+v0[1]*d[1])/d_norm # 初始纵向速度
    vd = (-v0[0]*d[1]+v0[1]*d[0])/d_norm
    tdd =  abs(vd)/curise_acc # 横向减到0的时间
    dd =  abs(vd)*tdd - 0.5*abs(curise_acc)*tdd**2 # 横向减速行驶距离
    # 横向减速后的坐标
    x =  init_state[0][0] - dd * d[1]/d_norm 
    y = init_state[0][1] + dd * d[0]/d_norm
    t = tdd
    if vs < 0:
        tds =  abs(vs)/curise_acc # 纵向减到0的时间
        ds =  abs(vs)*tds - 0.5*abs(curise_acc)*tds**2 # 纵向减速行驶距离
        x += ds*d[0]/d_norm 
        y += ds*d[1]/d_norm
        vs = 0
        t = max(tdd,tds)
    else:
        # 梯形速度策略
        pass
    sx = abs(end_state[0][0] - x)
    sy = abs(end_state[0][1] - y)
    s_norm = math.hypot(sx,sy)
    vx = vs*(end_state[0][0] - x)/s_norm # 初始纵向速度
    vy = vs*(end_state[0][1] - y)/s_norm


    


def galaxy_bspline_optim_test():
    slover = quniform_one_bspline_optim(one_spline_ctrl_points_num=8,degrees=5,fix_end_pos_state=False,v_limit=2.5) # ,j_weight=10.
    closest = 0.3
    farest = 15.
    # scans =closest+ (farest - closest)*np.random.random_sample(360)
    map_id = 386
    path = "/home/tou/Code/navrep/navrep/envs/desire10086/000_scans_robotstates_actions_rewards_dones.npz"
    scans = np.load(path)["scans"][map_id]
    # 减去机器人半径
    scan_x,scan_y = polar2xy(0,0,0,scans)

    figure,xycanv = plt.subplots()
    # xycanv.scatter(scan_x,scan_y,marker='s',s=8,c=(1,0,0,0.5))  
    start_time = time.time()
    convex_vertex,polar_convex_now,_ = galaxy_xyin_360out([p for p in zip(scan_x,scan_y)],radius = farest)
    # convex_vertex = galaxy([p for p in zip(scan_x,scan_y)],radius=farest)
    # convex_vertex.reverse()
    print("galaxy time:"+str(time.time() - start_time))
    # convex_vertex.append(convex_vertex[0])

    convex_vertex = [(0.6309471900337916, 5.4678576206576), (6.871659032901878, -6.810321902025998), (6.841258107979809, -6.8412581079798835), (6.7208197341908225, -6.959612568276427), (6.598334133604652, -7.075847063165494), (6.4738386165219834, -7.189926186493761), (6.347371105483129, -7.301815188655345), (6.218970123717217, -7.411479987176158), (6.088674783407272, -7.518887177096118), (5.956524773775628, -7.6240040411451275), (5.82256034899607, -7.726798559707559), (5.686822315929727, -7.827239420577582), (5.5493520216963494, -7.925296028496011), (5.410191341079531, -8.020938514469993), (5.269382663770316, -8.114137744872023), (5.126968881456298, -8.204865330313398), (4.982993374754794, -8.293093634292925), (4.837499999999986, -8.378795781614452), (4.690533075883279, -8.461945666573673), (4.542137369953516, -8.542517960910105), (4.392358084980059, -8.620488121522488), (4.2412408451843655, -8.695832397944422), (4.088831682341311, -8.76852783957957), (3.9351770217582716, -8.838552302692207), (3.7803236681337222, -8.905884457152364), (-1.8812730774731727, -3.85718141780295), (-1.9222282547555491, -3.77258536676207), (-1.9819130997131489, -3.3871727202676), (-0.3479761495764945, 4.441111406860332), (0.6309471900337916, 5.4678576206576)]

    xycanv.plot([convex_vertex[i][0] for i in range(len(convex_vertex))],[convex_vertex[i][1] for i in range(len(convex_vertex))],c='k')
    # convex_vertex.pop()
    # 期望匀速巡航 当终点在凸多边形内后 将终态直接作为终点 设置终态速度为0
    start_time = time.time()
    start = [(0.,0.),(-2.,1.),(1.,-1),(0.,0.)]
    start = [(-0.31687426921371076, 2.133176362469325), (-0.2314232587814329, -0.3678589463233948), (0.0, 0.0)]
    end = [(3.,-3.),(0.,-2.),(0.,0.),(0.,0.)]
    time_span = 3.
    ref_ctrl_points = [(-1.,0.),(-1.,-1.),(0.,-3.),(1,-3.7),(3.,-3.)] # ,(-1.,-2),(-1.,-3.)
    ref_ctrl_points = [(5.057468960973493, -4.335032739086049), (0.17418175157587498, 0.7021173043485361), (1.1566789634359915, 2.8685955258521725), (0.037542349058514446, 0.9181240189316954), (-0.06552626402374488, 2.2611723968225665)]
    success,spline,cost_track,ctrl = slover.formulate(time_span,convex_vertex,start,end=None,ref_ctrl_points=ref_ctrl_points)

    print("qp time:"+str(time.time() - start_time))
    spline = quniform_clamped_bspline2D(ctrl[0],ctrl[1],5,time_span)
    print([p for p in zip(ctrl[0],ctrl[1])])

    print("pva:")
    print(spline.calc_position_u(0),spline.calcd(0),spline.calcdd(0))
    xycanv.plot(end[0][0], end[0][1], 'ro')
    xycanv.plot(start[0][0], start[0][1], 'bo')
    xycanv.plot([p[0] for p in ref_ctrl_points], [p[1] for p in ref_ctrl_points], 'go')
    calc_2d_spline_interpolation(spline,xycanv,num=300)


if __name__ == "__main__":
    galaxy_bspline_optim_test()

# 分段
# if __name__ == "__main__":

        
#     # 每个box内四个控制点
#     # x,y,w,h来表达一个box
#     convex_polygons = [(0.,0.5,2.,3.),(1.05,2.1,0.5,1.8),(2.,3.2,2.,1.6),(4.4,4.5,3.2,3.)]
#     start = [(-1,1.1),(0.3,0.2),(0,0)]
#     end = [(3.1,3.2),(0.,0),(0,0)]
#     # 3次样条曲线，一段区间影响3+1个控制点，此例共13个控制点每个时间段对应3+1个控制点所以共10个span
#     # 每个box内共4个控制点，第四个是在交叠区域
#     # 还是需要分段b样条
#     # time_span = [3.5,3.5,3.3,3.5,3.7,3.4,3.3,3.5,3.7,3.4]
#     max_vel = 3.
#     max_acc = 8.
#     time_span = time_allocation(convex_polygons,max_vel,max_acc,start,end)
#     print(time_span)
#     # time_span = [3.5,3.4,3.3,3.5]


#     # k个box则有k+1个必经点，则m至少要有(k+1)*(p+1)那么长
#     # ，而m = n+p+1,n+p+1>=(k+1)*(p+1) --> n>=k*(p+1) --> 控制点数量>=k*(p+1)+1
#     # 每个box内的控制点数量 >= (p+1) + 1/k，即每个box内至少p+2个控制点(不重复)
#     # 不希望时间节点中只有重复节点，至少要另一个时间区间
#     # 因为第一个和最后一个box只有一个
#     # for i in range(len(time_span)):
#     #     time_span[i] /= 5.



#     def cross(p0,p1,p2):# p0->p1 x p0->p2
#         v1 = (p1[0]-p0[0],p1[1] - p0[1])
#         v2 = (p2[0]-p0[0],p2[1] - p0[1])
#         return v1[0]*v2[1] - v1[1]*v2[0]

#     o = quniform_piecewise_bspline_optim(6,convex_polygons,start,end,3,time_span,v_limit=max_vel,a_limit=max_acc)

    
#     start_time = time.time()
#     opt_x,opt_y,time_span = o.formulate()
#     end_time = time.time()
#     print(end_time - start_time,'s')
#     q = quniform_piecewise_bspline(opt_x,opt_y,3,time_span)

#     # q = qnonuniform_clamped_bspline2D(ctr_x,ctr_y,3,time_span)
#     # # # q = qnonuniform_clamped_bspline2D(ctr_x,ctr_y,3,[1.,1.,1.])
#     # q.calc_curvature_u(0)
#     # print("v:")
#     # print(time_span)

#     # print(q.sx.v_ctrl_points)
#     # print(q.sy.v_ctrl_points)
#     # # 起点的yaw和dkappa在后续突变是因为，起点的dy=dx=0,ddy=ddx=0,在进行除法时会出现0/0
#     # # 但是dy，dx又不是正好为0而是一个e-十几很小的数导致计算后的数值又看起来是一个停合理的数

#     # print("yaw 0 dx:",q.sx.calcd(0))
#     # print("yaw 0 dy:",q.sy.calcd(0))
#     # print("yaw 0:",q.calc_yaw_u(0))
#     # v = qnonuniform_clamped_bspline2D(q.sx.v_ctrl_points,q.sy.v_ctrl_points,2,time_span)
#     # a = qnonuniform_clamped_bspline2D(q.sx.a_ctrl_points,q.sy.a_ctrl_points,1,time_span)
#     # # plot_spline(v,[],start[1],end[1])
#     # # plot_spline(a,[],start[2],end[2])
#     # plot_spline(q,o.convex_polygons,start[0],end[0])
#     plot_spline(q,o.constraints_convexs,start[0],end[0])
#     plt.show()

# if __name__ == "__main__":
#     # polygon = [(-2,0),(0,2),(1,1.6),(1.5,1),(2,0),(5,5)]
#     # ctr_x = [p[0] for p in polygon]
#     # ctr_y = [p[1] for p in polygon]

#     # convex_polygons = [[(-1.,2.),(-1.,-1.),(1.,-1.),(1.,2.)],
#     # [(0.8,2.),(0.8,1.2),(1.,1.2),(1.,2.)],
#     # [(0.8,3.),(0.8,1.2),(1.3,1.2),(1.3,3)],
#     # [(1.,3.),(1.,2.4),(1.3,2.4),(1.3,3.)],
#     # [(1.,4.),(1.,2.4),(3.,2.4),(3.,4.)],
#     # [(2.8,4.),(2.8,3.),(3.,3.),(3.,4.)],
#     # [(2.8,6.),(2.8,3.),(6.,3.),(6.,6.)]]
#     # # 3次样条曲线，一段区间影响3+1个控制点
#     # time_span = [3.5,3.5,3.3,3.5,3.7,3.4]
    
#     # 每个box内四个控制点
#     convex_polygons = [[(-1.,2.),(-1.,-1.),(1.,-1.),(1.,2.)],
#     [(-1.,2.),(-1.,-1.),(1.,-1.),(1.,2.)],
#     [(0.8,2.),(0.8,1.2),(1.,1.2),(1.,2.)],
#     [(0.8,3.),(0.8,1.2),(1.3,1.2),(1.3,3)],
#     [(0.8,3.),(0.8,1.2),(1.3,1.2),(1.3,3)],
#     [(1.,3.),(1.,2.4),(1.3,2.4),(1.3,3.)],
#     [(1.,4.),(1.,2.4),(3.,2.4),(3.,4.)],
#     [(1.,4.),(1.,2.4),(3.,2.4),(3.,4.)],
#     [(2.8,4.),(2.8,3.),(3.,3.),(3.,4.)],
#     [(2.8,6.),(2.8,3.),(6.,3.),(6.,6.)],
#     [(2.8,6.),(2.8,3.),(6.,3.),(6.,6.)]]
#     # 3次样条曲线，一段区间影响3+1个控制点，此例共13个控制点每个时间段对应3+1个控制点所以共10个span
#     # 每个box内共4个控制点，第四个是在交叠区域
#     # 还是需要分段b样条
#     # time_span = [3.5,3.5,3.3,3.5,3.7,3.4,3.3,3.5,3.7,3.4]
#     time_span = [3.5,3.5,3.3,3.5,3.7,3.4,3.3,3.5,3.7,3.4]



#     # k个box则有k+1个必经点，则m至少要有(k+1)*(p+1)那么长
#     # ，而m = n+p+1,n+p+1>=(k+1)*(p+1) --> n>=k*(p+1) --> 控制点数量>=k*(p+1)+1
#     # 每个box内的控制点数量 >= (p+1) + 1/k，即每个box内至少p+2个控制点(不重复)
#     # 不希望时间节点中只有重复节点，至少要另一个时间区间
#     # 因为第一个和最后一个box只有一个
#     for i in range(len(time_span)):
#         time_span[i] /= 5.

#     start = [(0,0),(0,0),(0,0)]
#     end = [(5,5),(0,0),(0,0)]

#     def cross(p0,p1,p2):# p0->p1 x p0->p2
#         v1 = (p1[0]-p0[0],p1[1] - p0[1])
#         v2 = (p2[0]-p0[0],p2[1] - p0[1])
#         return v1[0]*v2[1] - v1[1]*v2[0]


#     o = qnonuniform_bspline_optim(convex_polygons,start,end,3,time_span)
    
#     start_time = time.time()
#     opt_x,opt_y = o.formulate()
#     end_time = time.time()
#     print(end_time - start_time,'s')

#     ctr_x = [start[0][0]]
#     ctr_x.extend(opt_x)
#     ctr_x.append(end[0][0])

#     ctr_y = [start[0][1]]
#     ctr_y.extend(opt_y)
#     ctr_y.append(end[0][1])

#     q = qnonuniform_clamped_bspline2D(ctr_x,ctr_y,3,time_span)
#     # # q = qnonuniform_clamped_bspline2D(ctr_x,ctr_y,3,[1.,1.,1.])
#     q.calc_curvature_u(0)
#     print("v:")
#     print(time_span)

#     print(q.sx.v_ctrl_points)
#     print(q.sy.v_ctrl_points)
#     # 起点的yaw和dkappa在后续突变是因为，起点的dy=dx=0,ddy=ddx=0,在进行除法时会出现0/0
#     # 但是dy，dx又不是正好为0而是一个e-十几很小的数导致计算后的数值又看起来是一个停合理的数

#     print("yaw 0 dx:",q.sx.calcd(0))
#     print("yaw 0 dy:",q.sy.calcd(0))
#     print("yaw 0:",q.calc_yaw_u(0))
#     v = qnonuniform_clamped_bspline2D(q.sx.v_ctrl_points,q.sy.v_ctrl_points,2,time_span)
#     a = qnonuniform_clamped_bspline2D(q.sx.a_ctrl_points,q.sy.a_ctrl_points,1,time_span)
#     # plot_spline(v,[],start[1],end[1])
#     # plot_spline(a,[],start[2],end[2])
#     plot_spline(q,o.convex_polygons,start[0],end[0])