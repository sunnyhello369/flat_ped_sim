from collections import deque
from hashlib import new
from tkinter.messagebox import NO
from cv2 import divide
import numpy as np
import math

UNKNOW = 0.5
class ocp_local_map(object):
    def __init__(self, scans = None , resolution = 0.3,config_space = 0.3,inflation = 0.6):# 机器人半径0.3 碰撞机器人半径个栅格则能将机器人当做质点处理
        # xy局部地图，以自车位姿作为坐标系原点和坐标系朝向
        # ij占据地图的索引，左下角为0,0
        drad =  2*np.pi/len(scans)
        self.x_min = 100
        self.x_max = -100
        self.y_min = 100
        self.y_max = -100
        self.xy_hits = []
        self.ij_hits = []
        self.resolution = resolution
        self.EXTEND_AREA = 1.
        self.config_space_step = math.ceil(config_space/resolution)
        self.inflate_step = math.ceil(inflation/resolution)
        
        for i in range(len(scans)):
            x = scans[i]*np.cos(i*drad)
            y = scans[i]*np.sin(i*drad)
            self.xy_hits.append((x,y))
            if x < self. x_min:
                self. x_min = x
            if y < self. y_min:
                self. y_min = y  

            if x > self. x_max:
                self. x_max = x
            if y > self. y_max:
                self. y_max = y            

        # self.occupancy =  -1 * np.ones(
        #     (int((self.x_max - self.x_min) / self.resolution), int((self.y_max - self.y_min) / self.resolution))
        # )
        # self.origin_ij = ( int(-self.x_min / self.resolution) , int(-self.y_min / self.resolution) )# 自车坐标系局部地图原点(0,0)所处索引
        self. x_min = round( self.x_min - self.EXTEND_AREA/2.0 )
        self. y_min = round( self.y_min - self.EXTEND_AREA/2.0 )
        self. x_max = round( self.x_max - self.EXTEND_AREA/2.0 )
        self. y_max = round( self.y_max - self.EXTEND_AREA/2.0 )

        self.occupancy = UNKNOW * np.ones(
            (int(round((self.x_max - self.x_min) / self.resolution)), int(round((self.y_max - self.y_min) / self.resolution))) )
        self.origin_ij = ( int(round(-self.x_min / self.resolution)) , int(round(-self.y_min / self.resolution)) )# 自车坐标系局部地图原点(0,0)所处索引

        self.init_flood_fill()
        self.flood_fill()

        for (x,y) in self.xy_hits:
            i,j = self.xy2ij(x,y)
            self.occupancy[i][j] = 1
            self.inflate(i,j)
    
    def init_flood_fill(self):
        # 原点到击中点直线肯定为可通行区域，直接作为泛洪的种子
        # prev_i,prev_j = self.origin_ij[0]-1,self.origin_ij[1] # 为什么-1
        prev_i,prev_j = self.origin_ij[0],self.origin_ij[1]
        for (x,y) in self.xy_hits:
            i,j = self.xy2ij(x,y)
            free_area = ocp_local_map.bresenham((prev_i, prev_j), (i, j))
            for fa in free_area:
                self.occupancy[fa[0]][fa[1]] = 0  # free area 0.0
            prev_i = i
            prev_j = j

    def flood_fill(self):
        q = deque()
        q.appendleft(self.origin_ij)
        # 泛洪将可通行周围的unknow也变为可通行
        # dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        dirs = [(-1,0),(0,-1),(0,1),(1,0)]

        while q:
            i,j = q.pop()
            for dir in dirs:
                if i+dir[0] < 0 or i+dir[0] >= self.occupancy.shape[0] or j+dir[1]<0 or j+dir[1] >= self.occupancy.shape[1]:
                    continue
                if math.isclose(self.occupancy[i+dir[0]][j+dir[1]], UNKNOW, rel_tol=1e-6):
                # if self.occupancy[i+dir[0]][j+dir[1]] == -1:
                    self.occupancy[i+dir[0]][j+dir[1]] = 0
                    q.appendleft((i+dir[0],j+dir[1]))
        # compiled_reverse_raytrace
        # for k in range(len(self.xy_hits)):
        #     self.ij_hits.append(self.xy2ij(*self.xy_hits[k]))
        #     ij = self.ij_hits[-1]
        #     i = ij[0]
        #     j = ij[1]
        #     # length of the ray in increments
        #     d_i = i - self.origin_ij[0]
        #     d_j = j - self.origin_ij[1]
        #     sign_i = 1 if d_i >= 0 else -1
        #     sign_j = 1 if d_j >= 0 else -1
        #     i_len = sign_i * d_i
        #     j_len = sign_j * d_j
        #     if i_len >= j_len:
        #         ray_inc = i_len
        #     else:
        #         ray_inc = j_len
        #     if ray_inc == 0:
        #         continue
        #     # calculate increments
        #     i_inc = d_i * 1. / ray_inc
        #     j_inc = d_j * 1. / ray_inc
        #     # max amount of increments before crossing the grid border
        #     if i_inc == 0:
        #         max_inc = (
        #             math.floor((self.occupancy .shape[1] - 1 - self.origin_ij[1]) / j_inc)
        #             if sign_j >= 0
        #             else math.floor(self.origin_ij[1] / -j_inc)
        #         )
        #     elif j_inc == 0:
        #         max_inc = (
        #             math.floor((self.occupancy .shape[0] - 1 - self.origin_ij[0]) / i_inc)
        #             if sign_i >= 0
        #             else math.floor(self.origin_ij[0] / -i_inc)
        #         )
        #     else:
        #         max_i_inc = (
        #             math.floor((self.occupancy.shape[0] - 1 - self.origin_ij[0]) / i_inc)
        #             if sign_i >= 0
        #             else math.floor(self.origin_ij[0] / -i_inc)
        #         )
        #         max_j_inc = (
        #             math.floor((self.occupancy.shape[1] - 1 - self.origin_ij[1]) / j_inc)
        #             if sign_j >= 0
        #             else math.floor(self.origin_ij[1] / -j_inc)
        #         )
        #         max_inc = max_i_inc if max_i_inc <= max_j_inc else max_j_inc
        #     if ray_inc < max_inc:
        #         max_inc = ray_inc
        #     # Trace a ray
        #     n_i = self.origin_ij[0] + 0
        #     n_j = self.origin_ij[1] + 0
        #     for n in range(1, max_inc):
        #         self.occupancy[
        #             int(n_i), int(n_j + j_inc)
        #         ] = 0  # the extra assignments make the ray 'thicker'
        #         n_i += i_inc
        #         self.occupancy[int(n_i), int(n_j)] = 0
        #         n_j += j_inc
        #         self.occupancy[int(n_i), int(n_j)] = 0

        # # set occupancy to 1 for ray hits (after the rest to avoid erasing hits)
        # for k in range(len(self.ij_hits)):
        #     ij = self.ij_hits[k]
        #     i = ij[0]
        #     j = ij[1]
        #     if (
        #         i > 0
        #         and j > 0
        #         and i != self.origin_ij[0]
        #         and j != self.origin_ij[1]
        #         and i < self.occupancy.shape[0]
        #         and j < self.occupancy.shape[1]
        #     ):
        #         self.occupancy[i, j] = 1
        #         self.inflate(i,j)



    def xy2ij(self,x,y):
        i = int(round((x - self.x_min) / self.resolution))
        j = int(round((y - self.y_min) / self.resolution))    


        if i < 0:
            i = 0
        elif i >= self.occupancy.shape[0]:
            i = self.occupancy.shape[0] - 1

        if j < 0:
            j = 0
        elif j >= self.occupancy.shape[1]:
            j = self.occupancy.shape[1] - 1

        return i,j

    def ij2xy(self, i, j):
        x = i * self.resolution + self.x_min
        y = j * self.resolution + self.y_min
        return x, y

    # 画线
    @staticmethod
    def bresenham(start, end):
        """
        Implementation of Bresenham's line drawing algorithm
        See en.wikipedia.org/wiki/Bresenham's_line_algorithm
        Bresenham's Line Algorithm
        Produces a np.array from start and end (original from roguebasin.com)
        >>> points1 = bresenham((4, 4), (6, 10))
        >>> print(points1)
        np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
        """
        # setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        # swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
        dx = x2 - x1  # recalculate differentials
        dy = y2 - y1  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y1 < y2 else -1
        # iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = [y, x] if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        if swapped:  # reverse the list if the coordinates were swapped
            points.reverse()
        # points = np.array(points)
        return points

    def inflate(self,i,j):
        # 内圈膨胀障碍物实体，实现机器人的配置空间
        # 外圈膨胀用于添加额外代价，使算法倾向于避开障碍物

        # 内圈
        # for n in range(i-self.inflate_step,i+self.inflate_step+1):
        #     if n < 0 or n > self.occupancy.shape[0]-1:
        #         continue
        #     for m in range(j-self.inflate_step,j+self.inflate_step+1):
        #         if m < 0 or m > self.occupancy.shape[1]-1:
        #             continue
        #         self.occupancy[n][m] = 1
        
        for n in range(-self.inflate_step - self.config_space_step,self.inflate_step + self.config_space_step+1):
            if i+n < 0 or i+n >= self.occupancy.shape[0]:
                continue
            for m in range(-self.inflate_step - self.config_space_step,self.inflate_step + self.config_space_step+1):
                if j+m < 0 or j+m >= self.occupancy.shape[1]:
                    continue
 
                if n >= -self.config_space_step and n<=self.config_space_step and m >= -self.config_space_step and m <= self.config_space_step:
                    self.occupancy[i+n][j+m] = 1
                else:
                    self.occupancy[i+n][j+m] = max(self.occupancy[i+n][j+m],math.exp(-math.sqrt(n**2+m**2)/10))

    def show(self):
        gridshow(self.occupancy)

    def plot(self,other_point_x = None,other_point_y = None,path_x = None,path_y = None):
        from matplotlib import pyplot as plt
        plt.figure()
        plt.gca().set_aspect('equal', adjustable='box')   
        plt.axis("equal")
        plt.grid(True)

        free_x  = []
        obs_x = []
        unkonw_x = []
        free_y  = []
        obs_y = []
        unkonw_y = []
        inflat_x = []
        inflat_y = []

        for i in range(self.occupancy.shape[0]):
                for j in range(self.occupancy.shape[1]):
                        if self.occupancy[i][j] == 0:
                            free_x.append(i)
                            free_y.append(j)
                        elif  self.occupancy[i][j] == 1:
                            obs_x.append(i)
                            obs_y.append(j)
                        elif math.isclose(self.occupancy[i][j], UNKNOW, rel_tol=1e-6):
                            unkonw_x.append(i)
                            unkonw_y.append(j)
                        # else:
                        #     inflat_x.append(i)
                        #     inflat_y.append(j)

        plt.scatter(free_x,free_y,s=2,marker='s',c=(1,1,1))   
        if other_point_x is not None and other_point_y is not None:
            # plt.plot(other_point_x,other_point_y,marker='s',lw=2,c=(1,0,0))         
            plt.scatter(other_point_x,other_point_y,marker='s',s=8,c=(1,0,0,0.5))  
        

        if path_x is not None and path_y is not None:
            plt.scatter(path_x,path_y,marker='s',s=2,c=(0,1,0))  

        plt.scatter(inflat_x,inflat_y,s=2,marker='s',c=(0,0,1))
        # for i in range(len(inflat_x)):
            # plt.scatter(inflat_x[i],inflat_y[i],s=2,marker='s',c=(0,0,self.occupancy[inflat_x[i]][inflat_y[i]]))
        plt.scatter(obs_x,obs_y,s=2,marker='s',c=(0,0,0))                            
        plt.scatter(unkonw_x,unkonw_y,s=2,marker='s',c=(0.5,0.5,0.5))                            

def gridshow(*args, **kwargs):
    """ utility function for showing 2d grids in matplotlib,
    wrapper around pyplot.imshow

    use 'extent' = [-1, 1, -1, 1] to change the axis values """
    from matplotlib import pyplot as plt
    if not 'origin' in kwargs:
        kwargs['origin'] = 'lower'
    if not 'cmap' in kwargs:
        kwargs['cmap'] = plt.cm.Greys
    return plt.imshow(*(arg.T if i == 0 else arg for i, arg in enumerate(args)), **kwargs)
    


class lidar2gridmap:

    def __init__(self, scans = None , resolution = 0.3,inflation = 0.3):# 机器人半径0.3 碰撞机器人半径个栅格则能将机器人当做质点处理
        # xy局部地图，以自车位姿作为坐标系原点和坐标系朝向
        # ij占据地图的索引，左下角为0,0
        drad =  2*np.pi/len(scans)
        self.x_min = 100
        self.x_max = -100
        self.y_min = 100
        self.y_max = -100
        self.xy_hits = []
        self.ij_hits = []
        self.resolution = resolution
        self.inflate_step = math.ceil(inflation/resolution)

        for i in range(len(scans)):
            x = scans[i]*np.cos(i*drad)
            y = scans[i]*np.sin(i*drad)
            self.xy_hits.append((x,y))
            if x < self. x_min:
                self. x_min = x
            if y < self. y_min:
                self. y_min = y  

            if x > self. x_max:
                self. x_max = x
            if y > self. y_max:
                self. y_max = y            
        self. x_min = round(self. x_min - 1 / 2.0)
        self. y_min = round(self. y_min - 1 / 2.0)
        self. x_max = round(self. x_max + 1 / 2.0)
        self. y_max = round(self. y_max + 1 / 2.0)

        self.occupancy =UNKNOW * np.ones(
            (int((round(self.x_max - self.x_min) / self.resolution)), int(round((self.y_max - self.y_min) / self.resolution)))
        )
        print(self.occupancy.shape)
        self.origin_ij = ( int(round(-self.x_min / self.resolution)) , int(round(-self.y_min / self.resolution)) )# 自车坐标系局部地图原点(0,0)所处索引
        


    def show(self):
        gridshow(self.occupancy)

    @staticmethod
    def bresenham(start, end): # 画线算法
        """
        Implementation of Bresenham's line drawing algorithm
        See en.wikipedia.org/wiki/Bresenham's_line_algorithm
        Bresenham's Line Algorithm
        Produces a np.array from start and end (original from roguebasin.com)
        >>> points1 = bresenham((4, 4), (6, 10))
        >>> print(points1)
        np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
        """
        # setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        # swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
        dx = x2 - x1  # recalculate differentials
        dy = y2 - y1  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y1 < y2 else -1
        # iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = [y, x] if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        if swapped:  # reverse the list if the coordinates were swapped
            points.reverse()
        # points = np.array(points)
        return points




class Node2D:
    def __init__(self, i,j,x,y):
        self.i = i
        self.j = j
        self.x = x
        self.y = y


class cube:
    def __init__(self, s_node,e_node):
        # s,e为corridor的膨胀基的xy坐标点
        self.s= s_node
        self.e = e_node

        if  self.e.i >  self.s.i:
            self.pos_di = self.e.i - self.s.i
            self.neg_di = 0
        else:
            self.pos_di = 0
            self.neg_di = self.s.i - self.e.i
    
        if self.e.j >  self.s.j:
            self.pos_dj = self.e.j - self.s.j
            self.neg_dj = 0
        else:
            self.pos_dj = 0
            self.neg_dj = self.s.j - self.e.j
        
        # start的xy坐标到膨胀后cube左下和右上点的dx,dy距离
        self.pos_x = 0
        self.pos_y = 0
        self.neg_x = 0
        self.neg_y = 0
        # AABB box
        self.min_x = 0
        self.min_y = 0
        self.max_x = 0
        self.max_y = 0
        self.min_i = 0
        self.min_j = 0
        self.max_i = 0
        self.max_j = 0         

class safe_travel_corridor:
    def __init__(self, local_map,path):
        self.grid_map = local_map
        # self.path = path # 自车局部坐标下的path, xy坐标
        self.path = path # 自车局部坐标下的path, i,j坐标

        self.cubes = []
        self.generate()

    def generate(self):
        index1 = 0
        index2 = 0
        while True:
            # 找到 path[index1],path[index2]两点间构成的矩形内部无障碍物点，
            # 后续以这个矩形作为"1/4"的corridor box，以此作为膨胀基
            n = index2
            i_min = self.path[index2][0]
            i_max = self.path[index2][0]
            j_min = self.path[index2][1]
            j_max = self.path[index2][1]
            while n < len(self.path):

                #### path xy
                # p1 = self.path[index2]
                # p2 = self.path[n]
                # p1_index = self.grid_map.xy2ij(*p1)
                # p2_index = self.grid_map.xy2ij(*p2)
                # i_min = min(p1_index[0],p2_index[0])
                # i_max = max(p1_index[0],p2_index[0])
                # j_min = min(p1_index[1],p2_index[1])
                # j_max = max(p1_index[1],p2_index[1])
                ######

                p1_index = self.path[index2]
                p2_index = self.path[n]
                prev_neg = (p1_index[1] - p2_index[1])/(p1_index[0] - p2_index[0] + 1e-6) <= 0
                # i_min = min(p1_index[0],p2_index[0])
                # i_max = max(p1_index[0],p2_index[0])
                # j_min = min(p1_index[1],p2_index[1])
                # j_max = max(p1_index[1],p2_index[1])

                i_min = min(i_min,p2_index[0])
                i_max = max(i_max,p2_index[0])
                j_min = min(j_min,p2_index[1])
                j_max = max(j_max,p2_index[1])

                safe = True
                # 找到路径上两点矩形空间内完全无障碍的空间大小
                # 可以是增量的检查，每次n+=1后只会多出一行，一列需要额外检查

                # 检查index2-n的box内部是否有障碍
                for i in range(i_min,i_max+1):
                    for j in range(j_min,j_max+1):
                            # if self.grid_map.occupancy[i][j] == 1:
                            # if self.grid_map.occupancy[i][j] == 1 or self.grid_map.occupancy[i][j] == -1:
                            neg = (p1_index[1] -j)/(p1_index[0] - i + 1e-6) <= 0
                            if self.grid_map.occupancy[i][j] >= 1 or math.isclose(self.grid_map.occupancy[i][j], UNKNOW, rel_tol=1e-6) \
                               : # 斜率变化的tube要挑出来  or (index2!=n and neg !=prev_neg)
                                if  (index2!=n and neg !=prev_neg):
                                    print(prev_neg)
                                    print(index2,n)
                                    print(self.path[index2])
                                    print(self.path[n])
                                    print(i,j)
                                    print(p1_index)
                                print("p:",index2,n)
                                print("col:",i,j)
                                safe = False
                                break
                    if not safe:
                        break
                
                if not safe:
                    break
                n += 1

            index1 = index2
            index2 = n - 1



            # s_index = self.grid_map.xy2ij(self.path[index1][0],self.path[index1][1])
            # s_node = Node2D(s_index[0],s_index[1],self.path[index1][0],self.path[index1][1] )

            # s_xy = self.grid_map.ij2xy(*self.path[index1])
            # s_node = Node2D(self.path[index1][0],self.path[index1][1],s_xy[0],s_xy[1])
            s_xy = self.grid_map.ij2xy(i_min,j_min)
            s_node = Node2D(i_min,j_min,s_xy[0],s_xy[1])

            # e_index = self.grid_map.xy2ij(self.path[index2][0],self.path[index2][1])
            # e_node = Node2D(e_index[0],e_index[1],self.path[index2][0],self.path[index2][1] )

            # e_xy = self.grid_map.ij2xy(*self.path[index2])
            # e_node = Node2D( self.path[index2][0],self.path[index2][1],e_xy[0],e_xy[1] )
            e_xy = self.grid_map.ij2xy(i_max,j_max)
            e_node = Node2D( i_max,j_max,e_xy[0],e_xy[1] )

            if n == 14:
                    print(index1)
                    print("s path ij:"+"("+str(self.path[index1][0])+","+str(self.path[index1][1])+")")
                    print("s path xy:"+"("+str(s_xy[0])+","+str(s_xy[1])+")")
                    print("e path ij:"+"("+str(self.path[index2][0])+","+str(self.path[index2][1])+")")
                    print("e path xy:"+"("+str(e_xy[0])+","+str(e_xy[1])+")")

            temp_cube = cube(s_node,e_node)
            temp_cube = self.expand_cube(temp_cube)
            if n == 14:
                new_ld_ij =  (temp_cube.s.i - temp_cube.neg_di,temp_cube.s.j - temp_cube.neg_dj)
                new_ru_ij =  (temp_cube.s.i + temp_cube.pos_di,temp_cube.s.j + temp_cube.pos_dj)
                print("left down ij:"+"("+str(new_ld_ij[0])+","+str(new_ld_ij[1])+")")
                print("right up ij:"+"("+str(new_ru_ij[0])+","+str(new_ru_ij[1])+")")

            temp_cube = self.update_cube(temp_cube)
            if n == 14:
                print("left down ij:"+"("+str(temp_cube.min_i)+","+str(temp_cube.min_j)+")")
                print("right up ij:"+"("+str(temp_cube.max_i)+","+str(temp_cube.max_j)+")")
                print("left down xy:"+"("+str(temp_cube.min_x)+","+str(temp_cube.min_y)+")")
                print("right up xy:"+"("+str(temp_cube.max_x)+","+str(temp_cube.max_y)+")")

            self.cubes.append(temp_cube)

  

            if index2 >= len(self.path) - 1:
                break # while

            if index1 == index2:
                print("stc fail")
                # print("break",index1,index2)
                # print(self.path[index1])
                return False


        print(len(self.cubes))
        if len(self.cubes) == 1:
            # 将这一个大corridor分成两个
            # print("divide")
            min_length = math.sqrt( (self.cubes[0].e.x - self.cubes[0].s.x)**2 + ( self.cubes[0].e.y - self.cubes[0].s.y )**2 )/2.1
            self.divide_corridor(min_length)
        return True
    def get_cubes_overlap(self):
        pass
    def expand_cube(self,cube):
            max_expand_pix_num = 4
            at_least_succ_once = True
            i_pos_ori = cube.pos_di
            i_neg_ori = cube.neg_di
            j_pos_ori = cube.pos_dj
            j_neg_ori = cube.neg_dj

            # 检查发现碰撞了后续就可以不检查了
            while at_least_succ_once:
                at_least_succ_once = False
                if cube.pos_di - i_pos_ori <= max_expand_pix_num:
                    cube.pos_di += 1
                    if not self.check_cube_safe(cube):
                        cube.pos_di -= 1
                    else:
                        at_least_succ_once = True
                
                if cube.neg_di - i_neg_ori <= max_expand_pix_num:
                    cube.neg_di += 1
                    if not self.check_cube_safe(cube):
                        cube.neg_di -= 1
                    else:
                        at_least_succ_once = True

                if cube.pos_dj - j_pos_ori <= max_expand_pix_num:
                    cube.pos_dj += 1
                    if not self.check_cube_safe(cube):
                        cube.pos_dj -= 1
                    else:
                        at_least_succ_once = True

                if cube.neg_dj - j_neg_ori <= max_expand_pix_num:
                    cube.neg_dj += 1
                    if not self.check_cube_safe(cube):
                        cube.neg_dj -= 1
                    else:
                        at_least_succ_once = True  

            return cube

    def update_cube(self,cube):
        new_ld_ij =  (cube.s.i - cube.neg_di,cube.s.j - cube.neg_dj)
        new_ld_xy = self.grid_map.ij2xy(*new_ld_ij)
        cube.x_neg = cube.s.x - new_ld_xy[0] + 0.5*self.grid_map.resolution
        cube.y_neg = cube.s.y - new_ld_xy[1] + 0.5*self.grid_map.resolution
        cube.min_x = cube.s.x - cube.x_neg
        cube.min_y = cube.s.y - cube.y_neg
        cube.min_i = new_ld_ij[0]
        cube.min_j = new_ld_ij[1]

        new_ru_ij =  (cube.s.i + cube.pos_di,cube.s.j + cube.pos_dj)
        new_ru_xy = self.grid_map.ij2xy(*new_ru_ij)
        cube.x_pos = new_ru_xy[0] - cube.s.x + 0.5*self.grid_map.resolution
        cube.y_pos = new_ru_xy[1] - cube.s.y + 0.5*self.grid_map.resolution
        cube.max_x = cube.s.x + cube.x_pos
        cube.max_y = cube.s.y + cube.y_pos
        cube.max_i = new_ru_ij[0]
        cube.max_j = new_ru_ij[1]

        print(new_ld_xy[0],new_ld_xy[1],new_ru_xy[0],new_ru_xy[1])

        return cube

    # 当只有一个corridor时选择将这个大corridor拆分,沿s->e按min length为步长分
    def divide_corridor(self,min_length):
        new_cubes = []
        for i in range(len(self.cubes)):
            diff = ( self.cubes[i].e.x - self.cubes[i].s.x , self.cubes[i].e.y - self.cubes[i].s.y )
            diff_norm =  math.sqrt(diff[0]**2 + diff[1]**2) 
            if  diff_norm >= 2*min_length:
                divide_num = math.floor(diff_norm/min_length)
                node_index = self.grid_map.xy2ij(self.cubes[i].s.x,self.cubes[i].s.y)
                node = Node2D(node_index[0],node_index[1],self.cubes[i].s.x,self.cubes[i].s.y)
                for j in range(1,divide_num+1):
                    cur_pos = (self.cubes[i].s.x + j/divide_num*diff[0],self.cubes[i].s.y + j/divide_num*diff[1])
                    cur_index = self.grid_map.xy2ij(*cur_pos)

                    cur_node = Node2D(cur_index[0],cur_index[1],cur_pos[0],cur_pos[1])
                    temp_cube = cube(node,cur_node)
                    temp_cube = self.expand_cube(temp_cube)
                    temp_cube = self.update_cube(temp_cube)
                    new_cubes.append(temp_cube)
                    node = cur_node
            else:
                new_cubes.append(self.cubes[i])
        self.cubes = new_cubes

    # 检查cube内是否完全无障碍
    # check可以只检查一行或一列
    def check_cube_safe(self,cube):
        for i in range(cube.s.i-cube.neg_di,cube.s.i+cube.pos_di+1):
            for j in range(cube.s.j-cube.neg_dj,cube.s.j+cube.pos_dj+1):
                # if self.grid_map.occupancy[i][j] == 1:
                # if self.grid_map.occupancy[i][j] ==1 or self.grid_map.occupancy[i][j] == -1:
                if self.grid_map.occupancy[i][j] == 1 or math.isclose(self.grid_map.occupancy[i][j], UNKNOW, rel_tol=1e-6):

                    if math.isclose(self.grid_map.occupancy[i][j], UNKNOW, rel_tol=1e-6):
                        print("unknow "+str(i)+","+str(j))
                    return False
        return True    

import queue

class astar_node:
    def __init__(self, x,y,h):
        self.x = x
        self.y = y
        self.g = -1
        self.f = h
        self.parent = None
        self.is_close = False
    # def __cmp__(self,other):
    #     return self.f > other.f
    def __lt__(self,other):
        # return self.f > other.f
        return self.f < other.f

# 纯A*无法实现走廊要求的路径规划 需要用Potential Field algorithm
avoid_weight = 50.
def Astar( occupancy,start,goal):
    # Quadratic Potential
    dist_map = {} # 吸力map,保存坐标到起点的路径长
    graph = []
    for i in range(occupancy.shape[0]):
        graph.append([])
        for j in range(occupancy.shape[1]):
            graph[-1].append(astar_node(i,j,avoid_weight*occupancy[i][j]+math.sqrt((i-goal[0])**2+(j-goal[1])**2)))
    open = queue.PriorityQueue()
    s = graph[start[0]][start[1]]
    s.g = 0
    open.put(s)
    path = []
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while not open.empty():
        top = open.get()
        if top.is_close:
            continue
        top.is_close = True
        if top.x == goal[0] and top.y == goal[1]:
            now = top
            while now is not None:
                path.append((now.x,now.y))
                now = now.parent
            print("ok")
            return path[::-1]#,dist_map
        for dir in dirs:
            if  ( top.x + dir[0] >= occupancy.shape[0] ) or (top.x + dir[0] < 0) or  ( top.y + dir[1] >= occupancy.shape[1]) or (top.y + dir[1] < 0) \
                or graph[top.x+dir[0]][top.y+dir[1]].is_close or occupancy[top.x+dir[0]][top.y+dir[1]] >= 1 or math.isclose(occupancy[top.x+dir[0]][top.y+dir[1]], UNKNOW, rel_tol=1e-6):
                continue
            now = graph[top.x+dir[0]][top.y+dir[1]]
            edge = 1
            if abs(dir[0]) + abs(dir[1])==2:
                edge = 1.414213562
            if now.g < 0:# 从未被探索
                now.g = top.g + edge
                now.f += now.g
                dist_map[(now.x,now.y)] = now.g
                now.parent = top
            elif top.g+edge < now.g:
                now.f -= now.g
                now.g = top.g +edge
                now.f += now.g
                now.parent = top
                dist_map[(now.x,now.y)] = now.g

            open.put(now)


    return None#,None

# A Light Formulation of the E∗Interpolated Path Replanner
# https://blog.csdn.net/weixin_40884570/article/details/121505255
class astar_poten_node:
    def __init__(self, x,y):
        self.x = x
        self.y = y
        self.cost = -1
        self.potential = 1e10
        self.parent = None
    # def __cmp__(self,other):
    #     return self.f > other.f
    def __lt__(self,other):
        # return self.f > other.f
        return self.cost < other.cost

def Astar_poten( occupancy,start,goal):
    # Quadratic Potential
    graph = []
    for i in range(occupancy.shape[0]):
        graph.append([])
        for j in range(occupancy.shape[1]):
            graph[-1].append(astar_poten_node(i,j))


    open = queue.PriorityQueue()
    s = graph[start[0]][start[1]]
    s.potential = 0
    s.cost = 0
    open.put(s)
    path = []
    dirs = [(-1,0),(0,-1),(0,1),(1,0)]
    natural_cost = 0.2
    while not open.empty():
        top = open.get()

        if top.x == goal[0] and top.y == goal[1]:
            now = top
            while now is not None:
                path.append((now.x,now.y))
                now = now.parent
            return path[::-1]
        for dir in dirs:
            if  ( top.x + dir[0] > occupancy.shape[0] ) or (top.x + dir[0] < 0) or  ( top.y + dir[1] > occupancy.shape[1]) or (top.y + dir[1] < 0) \
                or graph[top.x+dir[0]][top.y+dir[1]].potential < 1e10 or occupancy[top.x+dir[0]][top.y+dir[1]] >= 1 :
                continue
            now = graph[top.x+dir[0]][top.y+dir[1]]
            lp =  graph[top.x+dir[0]-1][top.y+dir[1]].potential
            rp =  graph[top.x+dir[0]+1][top.y+dir[1]].potential
            dp =  graph[top.x+dir[0]][top.y+dir[1]-1].potential
            up =  graph[top.x+dir[0]][top.y+dir[1]+1].potential
            if lp < rp:
                tc = lp
            else:
                tc = rp
            if up < dp:
                ta = up
            else:
                ta = dp
            hf = occupancy[now.x][now.y] + natural_cost
            dc = tc-ta
            if dc < 0:
                dc = -dc
                ta = tc
            if dc >= hf:
                now.potential = ta +hf
            else:
                d = dc/hf
                now.potential = ta +hf*(-0.2301*d**2+0.5307*d+0.7040)
            now.cost = now.potential +( abs(now.x - goal[0]) + abs(now.y - goal[1]))*natural_cost

            open.put(now)


    return None

def calc_potential_field(gx, gy, ox, oy, reso, rr, sx, sy):
    # # 终点 障碍物 分辨率 半径 起点
    # minx = min(min(ox), sx, gx) - AREA_WIDTH / 2.0
    # miny = min(min(oy), sy, gy) - AREA_WIDTH / 2.0
    # maxx = max(max(ox), sx, gx) + AREA_WIDTH / 2.0
    # maxy = max(max(oy), sy, gy) + AREA_WIDTH / 2.0
    # xw = int(round((maxx - minx) / reso))
    # yw = int(round((maxy - miny) / reso))

    # # calc each potential 势场大小
    # pmap = [[0.0 for i in range(yw)] for i in range(xw)]

    # # astar search
    # dist_map, path = Astar_search(gx, gy, sx, sy, ox, oy, 1, reso, minx, miny, maxx, maxy)
    pass
    for ix in range(xw):
        x = ix * reso + minx

        for iy in range(yw):
            y = iy * reso + miny
            # ug = calc_attractive_potential(x, y, gx, gy)
            ug = calc_attractive_potential_expanded(ix, iy, dist_map)
            uo = calc_repulsive_potential(x, y, ox, oy, rr)
            uf = ug + uo
            pmap[ix][iy] = uf

    return pmap, minx, miny, path


KP = 5.0  # attractive potential gain
ETA = 100.0  # repulsive potential gain
def calc_attractive_potential(x, y, gx, gy):
    return 0.5 * KP * np.hypot(x - gx, y - gy)

# 吸力 1/2*gain*dist(坐标，终点)
def calc_attractive_potential_expanded(x_ind, y_ind, dist_map):
    ind = (x_ind, y_ind)
    if ind not in dist_map.keys():
        return 1000
    else:
        return 0.5 * KP * dist_map[(x_ind, y_ind)]

# 斥力 到所有障碍物的最小距离
def calc_repulsive_potential(x, y, ox, oy, rr):
    # search nearest obstacle
    minid = -1
    dmin = float("inf")
    for i, _ in enumerate(ox):
        d = np.hypot(x - ox[i], y - oy[i])
        if dmin >= d:
            dmin = d
            minid = i

    # calc repulsive potential
    dq = np.hypot(x - ox[minid], y - oy[minid])

    if dq <= rr:
        if dq <= 0.1:
            dq = 0.1

        # rr为机器人半径，这里其实相当于把膨胀给算进去了
        return 0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2
    else:
        return 0.0