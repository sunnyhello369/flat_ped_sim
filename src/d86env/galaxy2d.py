import math
import random
from ssl import match_hostname
import numpy as np
import matplotlib.pyplot as plt
import time
# import cv2 as cv

def polar2xy(center_x,center_y,center_yaw,scan):
    # 计算世界坐标系下点云坐标，center_x,center_y,center_yaw 都是世界坐标下信息
    # scan 0自车正方向 逆时针转 .. 1079
    drad = 1./len(scan)*2*math.pi
    x = []
    y = []
    for i in range(len(scan)):
        x.append(center_x + scan[i]*math.cos(center_yaw + i*drad))
        y.append(center_y + scan[i]*math.sin(center_yaw + i*drad))
    return x,y

def NormalizeAngle(d_theta):# -pi-pi
    d_theta_normalize = d_theta
    while d_theta_normalize > math.pi:
        d_theta_normalize = d_theta_normalize - 2 * math.pi
    while d_theta_normalize < -math.pi:
        d_theta_normalize = d_theta_normalize + 2 * math.pi
    return d_theta_normalize


def NormalizeAngleTo2Pi(d_theta):# 0-2pi
    d_theta_normalize = d_theta
    while d_theta_normalize > 2 * math.pi:
        d_theta_normalize = d_theta_normalize - 2 * math.pi
    while d_theta_normalize < 0:
        d_theta_normalize = d_theta_normalize + 2 * math.pi
    return d_theta_normalize


def Graham_Andrew_Scan(points):
    n = len(points)
    if n < 3:
        return points,[i for i in range(n)]
    def cal_cross_product(p1 , p2, p3):# v1 x v2
        v1 = (p2[0]-p1[0],p2[1]-p1[1])
        v2 = (p3[0]-p1[0],p3[1]-p1[1])
        return v1[0] * v2[1] - v1[1] * v2[0]

    sort_index = list(range(n))
    # 按x升序排序，x接近则按y升序排序
    sort_index = sorted(sort_index, key=lambda x:(points[x][0],points[x][1]))
    count = 0
    last_count = 1
    res = []
    indexs = []
    for i in range(2*n):
        if i == n:
            last_count = count
        if i < n:
            index = sort_index[i]
        else:
            index = sort_index[2*n - 1 - i]
        pt = points[index]
        
        while count > last_count and cal_cross_product(res[count - 2],res[count - 1],pt) <= 0:
            res.pop()
            indexs.pop()
            count -= 1
        res.append(pt)
        indexs.append(index)
        count += 1
    count -= 1
    if count < 3:
        return None,None

    return res,indexs


def galaxy(scans_xy,origin_x = 0.,origin_y = 0.,radius = 15.):
    flip_data = []
    safe_radius = radius
    for x,y in scans_xy:
        dx = x - origin_x
        dy = y - origin_y
        norm2 = math.hypot(x,y)
        if norm2 < safe_radius:
            safe_radius = norm2
        if norm2 < 1e-6:
            continue
        # flip = p*(2R-norm2)/norm2
        flip_data.append((dx+2*(radius - norm2)*dx/norm2,dy+2*(radius - norm2)*dy/norm2))
        # flip_data.append((dx*(2*radius - norm2)/norm2,dy*(2*radius - norm2)/norm2))
    
    # plt.scatter([p[0] for p in flip_data],[p[1] for p in flip_data],marker='s',s=8,c=(0,1,0,0.5))  
   
    # convex hull
    _,vertex_indexs = Graham_Andrew_Scan(flip_data) # flip点云的凸包，反映射后不一定为凸 所以下面还有再计算一次凸包

    # vertex = cv.convexHull(np.array(flip_data,dtype=np.float32),clockwise=False)
    # plt.plot([scans_xy[vertex_indexs[i]][0] for i in range(len(vertex_indexs))],[scans_xy[vertex_indexs[i]][1] for i in range(len(vertex_indexs))],c='b')  
    vertex_indexs.pop()


    # 判断点云中心是否为星型多边形的一个顶点
    #     # 判断点云中心是否为星型多边形的一个顶点
    #     isOriginAVertex = False
    #     OriginIndex = -1
    #     vertexs = []
    #     for i in range(len(vertex_indexs)):
    #         if vertex_indexs[i] == len(scans_xy):
    #             isOriginAVertex = True
    #             OriginIndex =   i
    #             vertexs.append((origin_x,origin_y))
    #         else:
    #             vertexs.append(scans_xy[vertex_indexs[i]])    
                
    #     if isOriginAVertex:
    #         last_index = (OriginIndex-1)%len(vertexs)
    #         nxt_index =  (OriginIndex+1)%len(vertexs)
    #         dx = (scans_xy[vertex_indexs[last_index]][0] + origin_x + scans_xy[vertex_indexs[nxt_index]][0])/3. - origin_x
    #         dy = (scans_xy[vertex_indexs[last_index]][1] + origin_y + scans_xy[vertex_indexs[nxt_index]][1])/3. - origin_y

    #         d = math.hypot(dx,dy)
    #         interior_x = 0.99*safe_radius*dx/d+ origin_x
    #         interior_y = 0.99*safe_radius*dy/d+ origin_y        

    vertexs = []
    for i in range(len(vertex_indexs)):
            vertexs.append(scans_xy[vertex_indexs[i]])    

    interior_x = origin_x
    interior_y = origin_y

    _,indexs = Graham_Andrew_Scan(vertexs) # 要求逆时针 计算反映射后点云凸包
    indexs.pop()
    # plt.plot([vertexs[indexs[i]][0] for i in range(len(indexs))],[vertexs[indexs[i]][1] for i in range(len(indexs))],c='g')  
    
    constraints = [] # (a,b,c) a x + b y <= c
    for j in range(len(indexs )):
        jplus1 = (j+1)%len(indexs)
        # 星型多边形的一条边  
        rayV = (vertexs[indexs[jplus1]][0] - vertexs[indexs[j]][0],vertexs[indexs[jplus1]][1] - vertexs[indexs[j]][1])
       # 顺时针转90度
        normalj = (rayV[1],-rayV[0]) # point to outside
        norm = math.hypot(normalj[0],normalj[1])
        normalj = (normalj[0]/norm,normalj[1]/norm)
        ij = indexs[j]

        while ij != indexs[jplus1]:
            # 多边形内部三角形(以多边形边作为底边，以原点和顶点连线作为其他边)腰边在高方向的投影
            c = (vertexs[ij][0] - interior_x)*normalj[0] + (vertexs[ij][1] - interior_y)*normalj[1]
            constraints.append((normalj[0],normalj[1],c))
            ij = (ij+1)%len(vertexs)
    dual_points = []
    for c in constraints:
        dual_points.append((c[0]/c[2],c[1]/c[2]))
    
    dual_verterx =  Graham_Andrew_Scan(dual_points)[0]# 顺时针
    dual_verterx.pop()
    dual_verterx.reverse()
    final_vertex = []

    # 输出顺时针
    for i in range(len(dual_verterx)):
        iplus1 = (i+1)%len(dual_verterx)
        rayi =(dual_verterx[iplus1][0] - dual_verterx[i][0],dual_verterx[iplus1][1] - dual_verterx[i][1])
        c = rayi[1]*dual_verterx[i][0] - rayi[0]*dual_verterx[i][1]
        final_vertex.append((interior_x + rayi[1]/c,interior_y - rayi[0]/c))

    # print(final_vertex)
    # print(len(final_vertex))
    # final_vertex.append(final_vertex[0])
    # plt.plot([final_vertex[i][0] for i in range(len(final_vertex))],[final_vertex[i][1] for i in range(len(final_vertex))],c='k')  
    # plt.show()

    # output_constraints = []
    # for i in range(len(final_vertex)):
    #     iplus1 = (i+1)%len(final_vertex)
    #     rayi = (final_vertex[iplus1][0] - final_vertex[i][0],final_vertex[iplus1][1] - final_vertex[i][1])
    #     c = rayi[1]*final_vertex[i][0] - rayi[0]*final_vertex[i][1]
    #     output_constraints.append((rayi[1],-rayi[0],c)) # a x + b y <= c;
    # print(output_constraints)
    # return output_constraints,final_vertex
    return final_vertex

# def sparse_scan(scans_xy,drad):
#     prev_index = -1
#     next_index = 1
#     points = []
#     thetas = []
#     for i in range(len(scans_xy)-1):
#         prev = scans_xy[prev_index]
#         next = scans_xy[next_index]
#         p = scans_xy[i]
#         dtheta =NormalizeAngle(math.atan2(p[1] - prev[1],p[0] - prev[0]) - math.atan2(next[1] - p[1],next[0] - p[0]))
#         prev_index = i
#         next_index += 1
#         if abs(dtheta) > drad - 1e-6:
#             points.append(p)
#             thetas.append(math.atan2(p[1],p[0]))
#     return points,thetas

def sparse_scan(scans_xy,drad):
    if len(scans_xy) <= 3:
        thetas = []
        for p in scan_xy:
            thetas.append(math.atan2(p[1],p[0]))
        return points,thetas
    prev_index = -1
    next_index = 1
    points = []
    thetas = []
    now = 0
    while next_index < len(scans_xy):
        prev = scans_xy[prev_index]
        next = scans_xy[next_index]
        p = scans_xy[now]
        dtheta =NormalizeAngle(math.atan2(p[1] - prev[1],p[0] - prev[0]) - math.atan2(next[1] - p[1],next[0] - p[0]))
        # print(prev_index," prev ",prev)
        # print(next_index," next ",next)
        # print(now," now ",p)
        # print("dtheta",dtheta)
        # print("=======================================================================")

        if abs(dtheta) > drad - 1e-6:
            points.append(p)
            thetas.append(math.atan2(p[1],p[0]))
            prev_index = now
            now = next_index
        next_index += 1
    

    prev = scans_xy[-2]
    next = scans_xy[0]
    p = scans_xy[-1]
    dtheta =NormalizeAngle(math.atan2(p[1] - prev[1],p[0] - prev[0]) - math.atan2(next[1] - p[1],next[0] - p[0]))
    if abs(dtheta) > drad - 1e-6:
        points.append(p)
        thetas.append(math.atan2(p[1],p[0]))

    return points,thetas

def is_in_convex(point,convex,is_clock_wise = False):
    # convex为逆时针
    for i in range(1,len(convex)):
        cross = convex[i-1][0]*convex[i][1] - convex[i-1][1]*convex[i][0] +  (convex[i][0] - convex[i-1][0])*point[1]+ (convex[i-1][1] - convex[i][1])*point[0]
        if (cross < 0 and not is_clock_wise) or (cross > 0 and is_clock_wise):
            return False
    return True

# def galaxy_xyin_360out(scans_xy,origin_x = 0.,origin_y = 0.,radius = 15.):
#     lidar_num = len(scans_xy)
#     # drad = 2*math.pi/lidar_num
#     drad = 2*math.pi/360

#     points,_ = sparse_scan(scans_xy,drad)
#     # plt.scatter([points[i][0] for i in range(len(points))],[points[i][1] for i in range(len(points))],c='y')  

#     flip_data = []
#     safe_radius = radius
#     for x,y in points:
#         dx = x - origin_x
#         dy = y - origin_y
#         norm2 = math.hypot(x,y)
#         if norm2 < safe_radius:
#             safe_radius = norm2
#         if norm2 < 1e-6:
#             continue
#         # flip = p*(2R-norm2)/norm2
#         flip_data.append((dx+2*(radius - norm2)*dx/norm2,dy+2*(radius - norm2)*dy/norm2))
#         # flip_data.append((dx*(2*radius - norm2)/norm2,dy*(2*radius - norm2)/norm2))
    
#     plt.scatter([p[0] for p in flip_data],[p[1] for p in flip_data],marker='s',s=8,c=(0,1,0,0.5))  

#     # convex hull
#     _,vertex_indexs = Graham_Andrew_Scan(flip_data) # flip点云的凸包，反映射后不一定为凸 所以下面还有再计算一次凸包
#     if vertex_indexs is None or len(vertex_indexs) == 0:
#         return None,None,None 
#     vertex_indexs.pop()
    
#     vertexs = []
#     for i in range(len(vertex_indexs)):
#             vertexs.append(points[vertex_indexs[i]])   
#     # 反映射後的點雲 
#     # plt.plot([vertexs[i][0] for i in range(len(vertexs))],[vertexs[i][1] for i in range(len(vertexs))],c='r')  
#     interior_x = origin_x
#     interior_y = origin_y
#     _,indexs = Graham_Andrew_Scan(vertexs) # 要求逆时针 计算反映射后点云凸包
#     if indexs is None or len(indexs) == 0:
#         return None,None,None 
#     unsafe_convex = [vertexs[indexs[i]] for i in range(len(indexs))]
#     print(unsafe_convex)
#     print(is_in_convex((0.,0.),unsafe_convex))
#     plt.plot([vertexs[indexs[i]][0] for i in range(len(indexs))],[vertexs[indexs[i]][1] for i in range(len(indexs))],c='g')  

#     indexs.pop()


    
#     constraints = [] # (a,b,c) a x + b y <= c
#     for j in range(len(indexs )):
#         jplus1 = (j+1)%len(indexs)
#         # 星型多边形的一条边  
#         rayV = (vertexs[indexs[jplus1]][0] - vertexs[indexs[j]][0],vertexs[indexs[jplus1]][1] - vertexs[indexs[j]][1])
#        # 顺时针转90度
#         normalj = (rayV[1],-rayV[0]) # point to outside
#         norm = math.hypot(normalj[0],normalj[1])
#         normalj = (normalj[0]/norm,normalj[1]/norm)
#         ij = indexs[j]

#         while ij != indexs[jplus1]:
#             # 多边形内部三角形(以多边形边作为底边，以原点和顶点连线作为其他边)腰边在高方向的投影
#             c = (vertexs[ij][0] - interior_x)*normalj[0] + (vertexs[ij][1] - interior_y)*normalj[1] + 1e-6
#             constraints.append((normalj[0],normalj[1],c))
#             ij = (ij+1)%len(vertexs)
#     dual_points = []

#     for c in constraints:
#         dual_points.append((c[0]/c[2],c[1]/c[2]))
#     # plt.plot([dual_points[i][0] for i in range(len(dual_points))],[dual_points[i][1] for i in range(len(dual_points))],c='b')  

#     dual_verterx =  Graham_Andrew_Scan(dual_points)[0]# 顺时针
#     if dual_verterx is None or len(dual_verterx) == 0:
#         return None,None,None
#     dual_verterx.pop()
#     dual_verterx.reverse()
#     final_vertex = []
#     plt.plot([dual_verterx[i][0] for i in range(len(dual_verterx))],[dual_verterx[i][1] for i in range(len(dual_verterx))],c=(0,1,0,0.2))  

#     # 输出顺时针
#     for i in range(len(dual_verterx)):
#         iplus1 = (i+1)%len(dual_verterx)
#         rayi =(dual_verterx[iplus1][0] - dual_verterx[i][0],dual_verterx[iplus1][1] - dual_verterx[i][1])
#         c = rayi[1]*dual_verterx[i][0] - rayi[0]*dual_verterx[i][1]
#         final_vertex.append((interior_x + rayi[1]/c,interior_y - rayi[0]/c))
#     final_vertex.append(final_vertex[0])

#     # for i in range(len(final_vertex)):
#     #     print("======================================")
#     #     print(final_vertex[i])

#     # plt.plot([final_vertex[i][0] for i in range(len(final_vertex))],[final_vertex[i][1] for i in range(len(final_vertex))],c='k')  

#     # 转为与scans数量相同的极坐标形式
#     convex = []
#     thetas = []
#     points = []
#     convex,thetas = sparse_scan(final_vertex,drad)
#     print(convex)
#     # 分配角度范围
#     thetas.append(thetas[0])
#     convex.append(convex[0])
#     print(is_in_convex((0.,0.),convex))
#     # plt.plot([convex[i][0] for i in range(len(convex))],[convex[i][1] for i in range(len(convex))],c='k')  

#     nums = []
#     for i in range(len(thetas)-2):
#         nums.append(int(lidar_num*NormalizeAngleTo2Pi(thetas[i] - thetas[i+1])/math.pi/2.)) # 算上起点顶点的个数
#     nums.append(lidar_num - sum(nums))


#     polar_convex = []
#     polar_convex_theta = []
#     for i in range(len(convex)-1):
#         p = convex[i]
#         dist = math.hypot(p[0]-convex[i+1][0],p[1]-convex[i+1][1])
#         t = math.atan2(convex[i+1][1] - p[1],convex[i+1][0] - p[0])
#         dir = (math.cos(t),math.sin(t))
#         for j in range(nums[i]):
#             x = p[0] + j*dir[0]/nums[i]*dist
#             y = p[1] + j*dir[1]/nums[i]*dist
#             polar_convex.append(math.hypot(x,y))
#             polar_convex_theta.append(math.atan2(y,x))
#             points.append((x,y))
#     points.append(points[0])
    

#     plt.plot([convex[i][0] for i in range(len(convex))],[convex[i][1] for i in range(len(convex))],c='k')  
#     # print(len(final_vertex))
#     # print(len(polar_convex))
#     # print(lidar_num)
#     # print("polar_convex",polar_convex)
#     # plt.scatter([points[i][0] for i in range(len(points))],[points[i][1] for i in range(len(points))],c='b')  
#     plt.show()


#     # final_vertex重复了第一个点
#     return convex,polar_convex,polar_convex_theta


def galaxy_xyin_360out(scans_xy,origin_x = 0.,origin_y = 0.,radius = 15.):
    lidar_num = len(scans_xy)
    # drad = 2*math.pi/lidar_num
    drad = 2*math.pi/60

    # points,_ = sparse_scan(scans_xy,drad)
    # plt.scatter([points[i][0] for i in range(len(points))],[points[i][1] for i in range(len(points))],c='y')  

    flip_data = []
    safe_radius = radius
    for x,y in scans_xy:
        dx = x - origin_x
        dy = y - origin_y
        norm2 = math.hypot(x,y)
        if norm2 < safe_radius:
            safe_radius = norm2
        if norm2 < 1e-6:
            continue
        # flip = p*(2R-norm2)/norm2
        flip_data.append((dx+2*(radius - norm2)*dx/norm2,dy+2*(radius - norm2)*dy/norm2))
        # flip_data.append((dx*(2*radius - norm2)/norm2,dy*(2*radius - norm2)/norm2))
    
    # plt.scatter([p[0] for p in flip_data],[p[1] for p in flip_data],marker='s',s=8,c=(0,1,0,0.5))  

    # convex hull
    _,vertex_indexs = Graham_Andrew_Scan(flip_data) # flip点云的凸包，反映射后不一定为凸 所以下面还有再计算一次凸包
    if vertex_indexs is None or len(vertex_indexs) == 0:
        return None,None,None 
    vertex_indexs.pop()
    
    vertexs = []
    for i in range(len(vertex_indexs)):
            vertexs.append(scans_xy[vertex_indexs[i]])   
    # 反映射後的點雲 
    # plt.plot([vertexs[i][0] for i in range(len(vertexs))],[vertexs[i][1] for i in range(len(vertexs))],c='r')  
    interior_x = origin_x
    interior_y = origin_y
    _,indexs = Graham_Andrew_Scan(vertexs) # 要求逆时针 计算反映射后点云凸包
    if indexs is None or len(indexs) == 0:
        return None,None,None 
    unsafe_convex = [vertexs[indexs[i]] for i in range(len(indexs))]
    print(unsafe_convex)
    print(is_in_convex((0.,0.),unsafe_convex))
    plt.plot([vertexs[indexs[i]][0] for i in range(len(indexs))],[vertexs[indexs[i]][1] for i in range(len(indexs))],c='g')  

    indexs.pop()


    
    constraints = [] # (a,b,c) a x + b y <= c
    for j in range(len(indexs )):
        jplus1 = (j+1)%len(indexs)
        # 星型多边形的一条边  
        rayV = (vertexs[indexs[jplus1]][0] - vertexs[indexs[j]][0],vertexs[indexs[jplus1]][1] - vertexs[indexs[j]][1])
       # 顺时针转90度
        normalj = (rayV[1],-rayV[0]) # point to outside
        norm = math.hypot(normalj[0],normalj[1])
        normalj = (normalj[0]/norm,normalj[1]/norm)
        ij = indexs[j]

        while ij != indexs[jplus1]:
            # 多边形内部三角形(以多边形边作为底边，以原点和顶点连线作为其他边)腰边在高方向的投影
            c = (vertexs[ij][0] - interior_x)*normalj[0] + (vertexs[ij][1] - interior_y)*normalj[1] + 1e-6
            constraints.append((normalj[0],normalj[1],c))
            ij = (ij+1)%len(vertexs)
    dual_points = []

    for c in constraints:
        dual_points.append((c[0]/c[2],c[1]/c[2]))
    # plt.plot([dual_points[i][0] for i in range(len(dual_points))],[dual_points[i][1] for i in range(len(dual_points))],c='b')  

    dual_verterx =  Graham_Andrew_Scan(dual_points)[0]# 顺时针
    if dual_verterx is None or len(dual_verterx) == 0:
        return None,None,None
    dual_verterx.pop()
    dual_verterx.reverse()
    final_vertex = []
    plt.plot([dual_verterx[i][0] for i in range(len(dual_verterx))],[dual_verterx[i][1] for i in range(len(dual_verterx))],c=(0,1,0,0.2))  

    # 输出顺时针
    for i in range(len(dual_verterx)):
        iplus1 = (i+1)%len(dual_verterx)
        rayi =(dual_verterx[iplus1][0] - dual_verterx[i][0],dual_verterx[iplus1][1] - dual_verterx[i][1])
        c = rayi[1]*dual_verterx[i][0] - rayi[0]*dual_verterx[i][1]
        final_vertex.append((interior_x + rayi[1]/c,interior_y - rayi[0]/c))
    final_vertex.append(final_vertex[0])

    # for i in range(len(final_vertex)):
    #     print("======================================")
    #     print(final_vertex[i])

    # plt.plot([final_vertex[i][0] for i in range(len(final_vertex))],[final_vertex[i][1] for i in range(len(final_vertex))],c='k')  

    # 转为与scans数量相同的极坐标形式
    convex = []
    thetas = []
    points = []
    convex,thetas = sparse_scan(final_vertex,drad)
    print(convex)
    # 分配角度范围
    thetas.append(thetas[0])
    convex.append(convex[0])
    print(is_in_convex((0.,0.),convex,is_clock_wise=True))
    # plt.plot([convex[i][0] for i in range(len(convex))],[convex[i][1] for i in range(len(convex))],c='k')  

    nums = []
    for i in range(len(thetas)-2):
        nums.append(int(lidar_num*NormalizeAngleTo2Pi(thetas[i] - thetas[i+1])/math.pi/2.)) # 算上起点顶点的个数
    nums.append(lidar_num - sum(nums))


    polar_convex = []
    polar_convex_theta = []
    for i in range(len(convex)-1):
        p = convex[i]
        dist = math.hypot(p[0]-convex[i+1][0],p[1]-convex[i+1][1])
        t = math.atan2(convex[i+1][1] - p[1],convex[i+1][0] - p[0])
        dir = (math.cos(t),math.sin(t))
        for j in range(nums[i]):
            x = p[0] + j*dir[0]/nums[i]*dist
            y = p[1] + j*dir[1]/nums[i]*dist
            polar_convex.append(math.hypot(x,y))
            polar_convex_theta.append(math.atan2(y,x))
            points.append((x,y))
    points.append(points[0])
    

    plt.plot([convex[i][0] for i in range(len(convex))],[convex[i][1] for i in range(len(convex))],c='k')  
    # print(len(final_vertex))
    # print(len(polar_convex))
    # print(lidar_num)
    # print("polar_convex",polar_convex)
    # plt.scatter([points[i][0] for i in range(len(points))],[points[i][1] for i in range(len(points))],c='b')  
    plt.show()


    # final_vertex重复了第一个点
    return convex,polar_convex,polar_convex_theta

# if __name__ == "__main__":
#     # closest = 6.
#     closest = 2.
#     farest = 15.
#     scans =closest+ (farest - closest)*np.random.random_sample(360)
#     map_id = 386
#     path = "/home/chen/desire_10086/flat_ped_sim/src/d86env/000_scans_robotstates_actions_rewards_dones.npz"
#     scans = np.load(path)["scans"][map_id]
#     scan_x,scan_y = polar2xy(0,0,0,scans)
    
#     plt.scatter(scan_x,scan_y,marker='s',s=8,c=(1,0,0,0.5))  
#     start_time = time.time()
#     # galaxy([p for p in zip(scan_x,scan_y)],radius=farest)
#     galaxy_xyin_360out([p for p in zip(scan_x,scan_y)],radius=15.)
#     print(time.time() - start_time)
#     # p = [(-2,3),(-2,-3),(4,-3),(4,4),(3,0),(2,4),(1,0),(-2,3)]
#     # res,index = Graham_Andrew_Scan(p)
#     # print(res)
#     # print(index)
#     # plt.plot([p[index[i]][0] for i in range(len(index))],[p[index[i]][1] for i in range(len(index))],c='b')  
#     # plt.show()

if __name__ == "__main__":
    # closest = 6.
    closest = 2.
    farest = 10.
    scans =closest+ (farest - closest)*np.random.random_sample(360)
    map_id = 386
    path = "/home/chen/desire_10086/flat_ped_sim/src/d86env/000_scans_robotstates_actions_rewards_dones.npz"
    # scans =closest+ (farest - closest)*np.random.random_sample(360)
    # scans = [9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.63804436, 7.87623024, 6.76110315, 5.8458457, 5.16124964, 4.59345722, 4.19314623, 3.79380298, 3.53243303, 3.17341781, 2.96868706, 2.80790186, 2.65416694, 2.46317244, 2.25311494, 2.27611756, 2.1206615, 1.98518157, 1.93628669, 1.81558418, 1.66682196, 1.60744071, 1.56126988, 1.44782114, 1.52316594, 1.32293677, 1.44657862, 1.2677381, 1.28337669, 1.19975805, 1.25680709, 1.05069089, 1.16019213, 0.99271411, 1.0329845, 1.00826454, 0.98525506, 0.94817513, 0.94660348, 0.93930513, 0.90368003, 0.96576184, 0.77542025, 0.86634642, 0.93650705, 0.80761701, 0.80587524, 0.79938334, 0.77636021, 0.63866425, 0.71309656, 0.68224114, 0.7080788, 0.69332188, 0.64734161, 0.66368675, 0.75180823, 0.49239826, 0.56276113, 0.70155841, 0.65932512, 0.67616183, 0.53454202, 0.51510191, 0.62957513, 0.55495673, 0.53343552, 0.58386636, 0.63557655, 0.45842546, 0.51272619, 0.57681197, 0.56294686, 0.65066034, 0.60879695, 0.53746933, 0.48905295, 0.5942418, 0.54737198, 0.53438145, 0.52053726, 0.49389797, 0.63138193, 0.52809358, 0.53491455, 0.4847666, 0.43077987, 0.57911724, 0.55901998, 0.52828497, 0.58002204, 0.51250815, 0.64133066, 0.53246683, 0.57810783, 0.5391444, 0.53930116, 0.59713238, 0.54809541, 0.4336049, 0.49671477, 0.56878555, 0.56237239, 0.48436177, 0.60546857, 0.56818467, 0.65190709, 0.57348937, 0.60018677, 0.54245985, 0.69224387, 0.61791784, 0.64520824, 0.67383665, 0.76321799, 0.71530145, 0.66610318, 0.6906572, 0.74324673, 0.73575872, 0.76906234, 0.72819334, 0.79702991, 0.69206375, 0.76693386, 0.79403394, 0.79509193, 0.76415271, 0.75184482, 0.8412295, 0.84295839, 0.88988882, 0.97607833, 0.98860711, 0.94831389, 0.96845025, 1.00643647, 1.08665788, 1.12401021, 1.06604886, 1.16185272, 1.10108781, 1.26339781, 1.35735452, 1.31287122, 1.4155848, 1.40892172, 1.47754836, 1.5442009, 1.55657506, 1.76532054, 1.65690231, 1.89653754, 1.98275805, 1.99131632, 2.13848424, 2.25833106, 2.46758294, 2.595613, 2.75658441, 2.90817475, 3.15257931, 3.45988345, 3.74787879, 4.02259827, 4.51974249, 5.09808826, 5.71959162, 6.62393522, 7.64387751, 9.30093575, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019, 9.67500019]

    # scans = np.load(path)["scans"][map_id]
    # scans = [9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.14552784,7.57864857,6.42770767,5.56967592
    #         ,4.93907166,4.38144541,3.92515492,3.59699035,3.37449479,3.04417729
    #         ,2.92817998,2.63813567,2.44409919,2.32711029,2.2071085,2.09245086
    #         ,1.8802166,1.77337408,1.7674458,1.62646854,1.60443008,1.51119626
    #         ,1.4116981,1.4609493,1.28971279,1.34248769,1.20264256,1.1382184
    #         ,1.20392096,1.11763191,1.11398685,1.02448189,1.01279509,1.04634035
    #         ,0.9177106,1.04994595,0.9564907,0.79555601,0.84509748,0.87285846
    #         ,0.88618618,0.8501603,0.76500231,0.66945839,0.71048456,0.7651096
    #         ,0.75944561,0.71693772,0.71889955,0.72033685,0.65761536,0.6677494
    #         ,0.6536575,0.57533485,0.65047473,0.50695688,0.60450095,0.49016464
    #         ,0.59335792,0.5587579,0.61133748,0.58483303,0.47292054,0.59989607
    #         ,0.41078806,0.57523745,0.53667712,0.51731575,0.58259159,0.49668884
    #         ,0.46227115,0.56107914,0.44574636,0.53949004,0.52838576,0.57957089
    #         ,0.56575173,0.56917495,0.48529804,0.4590193,0.4699167,0.39870161
    #         ,0.50311905,0.50726849,0.47550869,0.45685142,0.49835247,0.50055629
    #         ,0.45203131,0.52324271,0.55403334,0.48497885,0.45866883,0.56349611
    #         ,0.56533718,0.50932711,0.5146156,0.50890809,0.54230183,0.50669569
    #         ,0.49780434,0.56255889,0.61077982,0.53596723,0.51157999,0.6011216
    #         ,0.57943374,0.53590631,0.55063421,0.5077998,0.60410577,0.64298373
    #         ,0.57434839,0.56158441,0.58000326,0.55557919,0.68863839,0.65267801
    #         ,0.56194621,0.75275677,0.66206014,0.68869287,0.65273511,0.63123006
    #         ,0.77473587,0.70193225,0.71863323,0.76819807,0.76691109,0.778202
    #         ,0.84099668,0.91729194,0.87641042,0.95241183,0.86803395,0.98991102
    #         ,0.99594933,1.04509425,1.03846812,1.08753276,1.08509469,1.09219944
    #         ,1.12991166,1.19715822,1.18642986,1.36748981,1.28335106,1.44944108
    #         ,1.39501131,1.55733955,1.48224735,1.72446418,1.80284095,1.78401542
    #         ,1.90542579,1.9608562,2.06798077,2.35521603,2.50226474,2.61199927
    #         ,2.83010769,3.05442452,3.35035944,3.54357123,3.90576768,4.27627993
    #         ,4.80207729,5.52261734,6.24468231,7.32563543,8.86038494,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019
    #         ,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019,9.67500019]


    scan_x,scan_y = polar2xy(0,0,0,scans)
    # print("x:")
    # print(scan_x)
    # print("y:")
    # print(scan_y)
    scan_xy = [(9.675,0),(9.67353,0.168852),(9.66911,0.337653),(9.66174,0.50635),(9.65143,0.674894),(9.63818,0.843232),(9.622,1.01131),(9.60288,1.17909),(9.58084,1.3465),(9.55588,1.5135),(9.52802,1.68005),(9.49724,1.84608),(9.46358,2.01155),(9.42703,2.1764),(9.38761,2.34059),(9.34533,2.50407),(9.30021,2.66679),(9.25225,2.8287),(9.20147,2.98974),(9.14789,3.14987),(9.09153,3.30904),(9.03239,3.46721),(8.9705,3.62432),(8.90588,3.78032),(8.83855,3.93518),(8.76853,4.08883),(8.69583,4.24124),(8.62049,4.39236),(8.54252,4.54214),(8.46195,4.69053),(8.3788,4.8375),(8.29309,4.98299),(8.20487,5.12697),(8.11414,5.26938),(8.02094,5.41019),(7.9253,5.54935),(7.82724,5.68682),(7.7268,5.82256),(7.624,5.95652),(7.51889,6.08868),(7.41148,6.21897),(7.30182,6.34737),(7.18993,6.47384),(7.07585,6.59833),(6.95961,6.72082),(6.84126,6.84126),(6.72082,6.95961),(6.59833,7.07585),(6.47384,7.18993),(6.34737,7.30182),(6.21897,7.41148),(6.08868,7.51889),(5.95653,7.624),(5.82256,7.7268),(5.68682,7.82724),(5.54935,7.9253),(5.41019,8.02094),(5.26938,8.11414),(5.12697,8.20487),(4.98299,8.29309),(4.8375,8.3788),(4.69053,8.46195),(4.54214,8.54252),(4.39236,8.62049),(4.24124,8.69583),(4.08883,8.76853),(3.93518,8.83855),(3.78032,8.90588),(3.62432,8.9705),(3.46721,9.03239),(3.30905,9.09153),(3.14987,9.14789),(2.98974,9.20147),(2.8287,9.25225),(2.66679,9.30021),(2.50407,9.34533),(2.34059,9.38761),(2.1764,9.42703),(2.01155,9.46358),(1.84608,9.49724),(1.68005,9.52802),(1.5135,9.55588),(1.3465,9.58084),(1.17909,9.60288),(1.01131,9.622),(0.843232,9.63818),(0.674894,9.65143),(0.506351,9.66174),(0.337653,9.66911),(0.168852,9.67353),(-4.22908e-07,9.675),(-0.168852,9.67353),(-0.337652,9.66911),(-0.50635,9.66174),(-0.674894,9.65143),(-0.843232,9.63818),(-1.01131,9.622),(-1.17909,9.60288),(-1.3465,9.58084),(-1.5135,9.55588),(-1.68005,9.52802),(-1.84608,9.49724),(-2.00338,9.42519),(-2.13797,9.26055),(-2.2516,9.03068),(-2.39565,8.94069),(-2.49941,8.71647),(-2.65635,8.68852),(-2.74152,8.43753),(-2.859,8.30315),(-2.99451,8.22736),(-3.09712,8.06828),(-3.21886,7.96695),(-3.31575,7.81142),(-3.46836,7.79006),(-3.52125,7.55134),(-3.61373,7.40924),(-3.74082,7.34177),(-3.80542,7.15695),(-3.91411,7.06124),(-4.08405,7.07378),(-4.14763,6.90281),(-4.21528,6.74585),(-4.3152,6.64482),(-4.44519,6.59027),(-4.57357,6.53173),(-4.60125,6.33308),(-4.71721,6.25995),(-4.83033,6.18254),(-4.92393,6.08054),(-4.98269,5.93813),(-5.06472,5.8263),(-5.15203,5.72191),(-5.23569,5.61459),(-5.30291,5.49133),(-5.35344,5.35344),(-5.52957,5.33984),(-5.62163,5.24225),(-5.71991,5.15023),(-5.8766,5.10845),(-5.88204,4.93562),(-5.96101,4.82714),(-6.05847,4.73339),(-6.22693,4.69233),(-6.20198,4.506),(-6.32602,4.42953),(-6.41244,4.32524),(-6.51387,4.23015),(-6.61806,4.13542),(-6.68159,4.0147),(-6.84381,3.95127),(-6.89968,3.82456),(-6.92229,3.68065),(-7.12142,3.62855),(-7.14484,3.48477),(-7.29767,3.40296),(-7.39767,3.29366),(-7.44155,3.15875),(-7.64985,3.09074),(-7.71399,2.96113),(-7.70898,2.80584),(-7.94039,2.73409),(-7.88869,2.56319),(-8.15681,2.49379),(-8.28057,2.37441),(-8.30196,2.22451),(-8.53167,2.12718),(-8.53536,1.97054),(-8.70478,1.85026),(-8.84754,1.71979),(-8.90702,1.57055),(-9.12807,1.44574),(-9.18988,1.29155),(-9.34843,1.14784),(-9.44822,0.993048),(-9.60688,0.840495),(-9.65143,0.674894),(-9.66174,0.506351),(-9.66911,0.337652),(-9.67353,0.168853),(-9.675,-8.45815e-07),(-9.67353,-0.168852),(-9.66911,-0.337652),(-9.66174,-0.506351),(-9.65143,-0.674893),(-9.63818,-0.843233),(-9.622,-1.01131),(-9.60288,-1.17908),(-9.58084,-1.3465),(-9.55589,-1.5135),(-9.52802,-1.68005),(-9.49724,-1.84608),(-9.46358,-2.01155),(-9.42703,-2.1764),(-9.38761,-2.34059),(-9.34533,-2.50407),(-9.30021,-2.66679),(-9.25225,-2.8287),(-9.20147,-2.98974),(-9.14789,-3.14987),(-9.09153,-3.30905),(-9.03239,-3.46721),(-8.9705,-3.62432),(-8.90588,-3.78032),(-8.83855,-3.93518),(-8.76853,-4.08883),(-8.69583,-4.24124),(-8.62049,-4.39236),(-8.54252,-4.54214),(-8.46195,-4.69053),(-8.3788,-4.8375),(-8.29309,-4.98299),(-8.20487,-5.12697),(-8.11414,-5.26938),(-8.02094,-5.41019),(-7.9253,-5.54935),(-7.82724,-5.68682),(-7.7268,-5.82256),(-7.624,-5.95652),(-7.51889,-6.08868),(-7.41148,-6.21897),(-7.30181,-6.34737),(-7.18993,-6.47384),(-7.07585,-6.59833),(-6.95961,-6.72082),(-6.84126,-6.84126),(-6.72082,-6.95961),(-6.59833,-7.07585),(-6.47384,-7.18993),(-6.34737,-7.30182),(-6.21897,-7.41148),(-6.08867,-7.51889),(-5.95653,-7.624),(-5.82256,-7.7268),(-5.68682,-7.82724),(-5.54935,-7.9253),(-5.41019,-8.02094),(-5.26938,-8.11414),(-5.12697,-8.20486),(-4.983,-8.29309),(-4.8375,-8.3788),(-4.69053,-8.46195),(-4.54214,-8.54252),(-4.39236,-8.62049),(-4.24124,-8.69583),(-4.08883,-8.76853),(-3.93518,-8.83855),(-3.78032,-8.90588),(-3.62432,-8.9705),(-3.46721,-9.03239),(-3.30904,-9.09153),(-3.14987,-9.14789),(-2.98974,-9.20147),(-2.8287,-9.25225),(-2.66679,-9.30021),(-2.50407,-9.34533),(-2.34059,-9.38761),(-2.1764,-9.42703),(-2.01155,-9.46358),(-1.84608,-9.49724),(-1.68005,-9.52802),(-1.5135,-9.55588),(-1.3465,-9.58084),(-1.17909,-9.60288),(-1.01131,-9.622),(-0.843232,-9.63818),(-0.674895,-9.65143),(-0.506352,-9.66174),(-0.337655,-9.66911),(-0.168851,-9.67353),(1.15373e-07,-9.675),(0.168851,-9.67353),(0.337651,-9.66911),(0.506352,-9.66174),(0.674895,-9.65143),(0.843232,-9.63818),(1.01131,-9.622),(1.17908,-9.60288),(1.3465,-9.58084),(1.5135,-9.55588),(1.68005,-9.52802),(1.84608,-9.49724),(2.01154,-9.46358),(2.1764,-9.42703),(2.34059,-9.38761),(2.50407,-9.34533),(2.66679,-9.30021),(2.82869,-9.25225),(2.98974,-9.20147),(3.14987,-9.14789),(3.30904,-9.09153),(3.46721,-9.03239),(3.62432,-8.9705),(3.78032,-8.90588),(3.93518,-8.83855),(4.08883,-8.76853),(4.24124,-8.69583),(4.39236,-8.62049),(4.54214,-8.54252),(4.69053,-8.46195),(4.8375,-8.3788),(4.98299,-8.29309),(5.12697,-8.20486),(5.26938,-8.11414),(5.41019,-8.02094),(5.54935,-7.9253),(5.68682,-7.82724),(5.82256,-7.7268),(5.95653,-7.624),(6.08867,-7.51889),(6.21897,-7.41148),(6.34737,-7.30182),(6.47384,-7.18993),(6.59833,-7.07585),(6.72082,-6.95961),(6.84126,-6.84126),(6.95961,-6.72082),(7.07585,-6.59833),(7.18993,-6.47384),(7.30181,-6.34737),(7.41148,-6.21897),(7.51889,-6.08868),(7.624,-5.95652),(7.7268,-5.82256),(7.82724,-5.68682),(7.9253,-5.54935),(8.02094,-5.41019),(8.11414,-5.26938),(8.20487,-5.12697),(8.29309,-4.98299),(8.37879,-4.8375),(8.46195,-4.69053),(8.54252,-4.54214),(8.62049,-4.39236),(8.69583,-4.24124),(8.76853,-4.08883),(8.83855,-3.93518),(8.90588,-3.78032),(8.9705,-3.62432),(9.03239,-3.46721),(9.09153,-3.30905),(9.14789,-3.14987),(9.20147,-2.98974),(9.25225,-2.8287),(9.30021,-2.66679),(9.34533,-2.50408),(9.38761,-2.34059),(9.42703,-2.1764),(9.46358,-2.01155),(9.49724,-1.84608),(9.52801,-1.68005),(9.55589,-1.5135),(9.58084,-1.3465),(9.60288,-1.17909),(9.622,-1.01131),(9.63818,-0.843235),(9.65143,-0.674893),(9.66174,-0.50635),(9.66911,-0.337654),(9.67353,-0.168854)]

    plt.scatter([p[0] for p in scan_xy],[p[1] for p in scan_xy],marker='s',s=8,c=(1,0,0,0.5))  
    # plt.scatter(scan_x,scan_y,marker='s',s=8,c=(1,0,0,0.5))  

    start_time = time.time()
    # points,_ = Graham_Andrew_Scan(scan_xy)
    
    # galaxy_xyin_360out([p for p in zip(scan_x,scan_y)],radius=10.)
    galaxy_xyin_360out(scan_xy,radius=10.)
    print(time.time() - start_time)
    # p = [(-2,3),(-2,-3),(4,-3),(4,4),(3,0),(2,4),(1,0),(-2,3)]
    # res,index = Graham_Andrew_Scan(p)
    # print(res)
    # print(index)
    # plt.plot([p[index[i]][0] for i in range(len(index))],[p[index[i]][1] for i in range(len(index))],c='b')  
    # plt.show()

# if __name__ == "__main__":
#     # closest = 6.
#     closest = 2.
#     farest = 15.
#     scans =closest+ (farest - closest)*np.random.random_sample(360)
#     map_id = 386
#     path = "/home/chen/desire_10086/flat_ped_sim/src/d86env/000_scans_robotstates_actions_rewards_dones.npz"
#     # path = "/home/tou/navrep/datasets/V/navreptrain"
#     scans = np.load(path)["scans"]
#     fig =plt.figure("galaxy2D")
    
#     for scan in scans:
#         scan_x,scan_y = polar2xy(0,0,0,scan)
#         plt.plot(0,0, 'go')
#         plt.scatter(scan_x,scan_y,marker='s',s=8,c=(1,0,0,0.5)) 
#         t = time.time()
#         galaxy_xyin_360out([p for p in zip(scan_x,scan_y)],radius=15.)
#         print(time.time() - t)
#         plt.pause(0.001)
#         fig.clf()
#         # galaxy([p for p in zip(scan_x,scan_y)],radius=farest)
#         # galaxy_xyin_360out([p for p in zip(scan_x,scan_y)],radius=15.)
#         # plt.show()
#     exit(0)