import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Circle
from matplotlib.markers import MarkerStyle
import math
from rl_data_plot_json import rl_DataPlotter
from teb_data_plot_json import teb_DataPlotter


def calc_cross_point(a1,b1,c1,a2,b2,c2):
    D = a1*b2 - a2*b1
    print(D)
    if abs(D) < 1e-6:# 平行
        return None
    return ((b1*c2 - b2*c1)/D,(a2*c1 - a1*c2)/D)


# 导入math模块，用于计算平方根和绝对值
import math

def point_to_segment_distance(point, segment):
    # 将参数解构为坐标值
    x0, y0 = point # 点的坐标
    x1, y1 = segment[0] # 线段的第一个端点坐标
    x2, y2 = segment[1] # 线段的第二个端点坐标

    # 计算线段的长度
    segment_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) **2)

    # 如果线段长度为零，说明线段退化为一个点，直接返回点到点的距离
    if segment_length < 1e-6:
        return False,math.sqrt((x0 - x1) ** 2 + (y0 - y1) **2)

    # 计算点到线段所在直线的距离，使用公式 |Ax0 + By0 + C| / sqrt(A^2 + B^2)
    # 其中 A = y2 - y1, B = x1 - x2, C = x2 * y1 - x1 * y2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    distance_to_line = abs(A * x0 + B * y0 + C) / math.sqrt(A ** 2 + B ** 2)

    # 计算点在直线上的投影点的坐标，使用公式 
    # (B^2 * x0 - A * B * y0 - A * C) / (A^2 + B^2), 
    # (A^2 * y0 - A * B * x0 - B * C) / (A^2 + B^2)
    projection_x = (B ** 2 * x0 - A * B * y0 - A * C) / (A ** 2 + B ** 2)
    projection_y = (A ** 2 * y0 - A * B * x0 - B * C) / (A ** 2 + B ** 2)

    # 判断投影点是否在线段上，使用向量的数量积判断
    # 如果 (x1 - projection_x) * (x2 - projection_x) <= 0 并且 (y1 - projection_y) * (y2 - projection_y) <= 0，
    # 则投影点在线段上，否则不在线段上
    if (x1 - projection_x) * (x2 - projection_x) <= 0 and (y1 - projection_y) * (y2 - projection_y) <= 0:
        # 投影点在线段上，直接返回点到直线的距离
        return True,distance_to_line
    else:
    # 投影点不在线段上，返回点到线段两个端点中较近的一个的距离
        return False,min(math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2), math.sqrt((x0 - x2) ** 2 + (y0 - y2) ** 2))

obs_num = 30
xlim = (-5.5, 6.5)
ylim = (-5.5, 6.5)
save_path = "D:/RL/_1repertory_log/230616_data/"
font = fm.FontProperties(fname="C:/Windows/Fonts/simsun.ttc") # times.ttf 
# d10
# eps_list = [308,350,538,632,760,776,890] # 155i 201i 254 308i 692i 746i 896
# eps_list = [155, 201, 254, 308, 692, 746,760,890, 896]
# d20
# eps_list = [39,157,474,644,685] # 39 238 157 282i 474 685 960
# eps_list = [157]
# d30
eps_list = [378,413,515,547,771] # 75,167,377,413,515!,547!,771!
# eps_list = [i for i in range(1000)]


def plot(teb_info,teb_plotter,rl_info,rl_plotter):
    global xlim
    global ylim
    global save_path
    global font
    global eps_list
    teb_x_trajs,teb_y_trajs,teb_max_steps,teb_obstacle_trajectorys,start_positions,goal_positions = teb_info[0],teb_info[1],teb_info[2],teb_info[3],teb_info[4],teb_info[5]
    rl_x_trajs,rl_y_trajs,rl_max_steps,rl_obstacle_trajectorys,start_positions,goal_positions,polygon_points_by_steps,ctrl_network_points_by_steps = rl_info[0],rl_info[1],rl_info[2],rl_info[3],rl_info[4],rl_info[5],rl_info[6],rl_info[7]

    text_offset = 0.15
    
    for eps in range(0,len(teb_obstacle_trajectorys)):
        teb_max_step = teb_max_steps[eps]
        rl_max_step = rl_max_steps[eps]
        time_max_step = 0
        if teb_max_step > rl_max_step:
            plot_traj = teb_obstacle_trajectorys[eps]
            time_max_step = teb_max_step
        else:
            plot_traj = rl_obstacle_trajectorys[eps]
            time_max_step = rl_max_step

        close_num = 0
        proj_on_line = False
        proj_on_line_num = 0

        for i in range(len(plot_traj)):
            if len(plot_traj[i]) == 0:
                continue
            s_proj_on_line,sd = point_to_segment_distance((plot_traj[i][0][0],plot_traj[i][0][1]),(start_positions[eps],goal_positions[eps]))
            if s_proj_on_line:
                proj_on_line_num += 1
            f_proj_on_line,fd = point_to_segment_distance((plot_traj[i][-1][0],plot_traj[i][-1][1]),(start_positions[eps],goal_positions[eps]))
            if f_proj_on_line:
                proj_on_line_num += 1 
            # if s_proj_on_line and f_proj_on_line and sd - fd > 0.5:
            #     close_num += 1
            if sd - fd > 0 and sd < 5 and fd < 5:
                close_num += 1
            # if s_proj_on_line and f_proj_on_line and sd - fd > 0:
            #     close_num += 1
 
        if close_num < 15 :# or proj_on_line_num < 10
            continue

        res = rl_plotter.stats.get_eps_stats(eps_list[eps]+1)
        if res is not None:
            rl_success,time, distance, velocity = res
            print(f'RL Eps {eps_list[eps]} - success: {rl_success}, time: {time}, distance: {distance}, velocity: {velocity}')
        else:
            continue

        res = teb_plotter.stats.get_eps_stats(eps_list[eps]+1)
        if res is not None:
            teb_success,time, distance, velocity = res
            print(f'TEB Eps {eps_list[eps]} - success: {teb_success}, time: {time}, distance: {distance}, velocity: {velocity}')
        else:
            continue

        if not (rl_success and not teb_success):
            continue
        
        teb_x_trajs[eps][0] = rl_x_trajs[eps][0]
        teb_y_trajs[eps][0] = rl_y_trajs[eps][0]

        fig, ax = plt.subplots(1,2)
        ax[0].set_xlim(xlim)
        ax[0].set_ylim(ylim)
        ax[0].set_aspect('equal') # Set 1:1 aspect ratio
        ax[0].set_xlabel("X")
        ax[0].set_ylabel("Y")
        # ax[0].grid()
        ax[0].set_title(f"TEB",fontsize = 40)


        ax[1].set_xlim(xlim)
        ax[1].set_ylim(ylim)
        ax[1].set_aspect('equal') # Set 1:1 aspect ratio
        ax[1].set_xlabel("X")
        ax[1].set_ylabel("Y")
        # ax[1].grid()
        # ax[1].set_title(f"RL {eps}")
        ax[1].set_title("本方法", fontproperties=font,fontsize=40)

        # ax[0].plot(start_positions[eps][0],start_positions[eps][1], 'rs', markersize=10)
        # ax[0].text(start_positions[eps][0]+text_offset,start_positions[eps][1]+text_offset, "START", fontsize=12, color='red')
        # ax[0].plot(goal_positions[eps][0],goal_positions[eps][1], 'bs', markersize=10)
        # ax[0].scatter(goal_positions[eps][0],goal_positions[eps][1],c=(0,1,0,1), marker='*',s = 100)
        # ax[0].text(goal_positions[eps][0]+text_offset,goal_positions[eps][1]+text_offset, "GOAL", fontsize=12, color='blue')
        # circle = Circle((goal_positions[eps][0],goal_positions[eps][1]), 1, color='r', alpha=0.1)  # Add circle for robot radius
        # ax[0].add_patch(circle)


        if teb_success:
            dist = (goal_positions[eps][1]-teb_y_trajs[eps][-1])**2+(goal_positions[eps][0]-teb_x_trajs[eps][-1])**2
            dist = math.sqrt(dist)
            theta = math.atan2(goal_positions[eps][1]-teb_y_trajs[eps][-1],goal_positions[eps][0]-teb_x_trajs[eps][-1])

            # x = teb_x_trajs[eps][-1] + 1*dist*math.cos(theta)
            # y = teb_y_trajs[eps][-1] + 1*dist*math.sin(theta)
            teb_x_trajs[eps].append(goal_positions[eps][0])
            teb_y_trajs[eps].append(goal_positions[eps][1])

        for i in range(len(plot_traj)):
            a = 0.05
            alpha_step  = (0.5-0.05)/len(plot_traj[i])
            f = plot_traj[i][0]
            l = plot_traj[i][-1]
            traj_dist = math.sqrt((f[0] - l[0])**2+(f[1] - l[1])**2)
            draw_flag1 = False
            draw_flag2 = False
            draw_flag3 = False
            for j in range(len(plot_traj[i])):
                p = plot_traj[i][j]
                x,y,step,r = p[0],p[1],p[2],p[3]
                ax[0].plot(x, y, 'ro', markersize=2)
                ax[1].plot(x, y, 'ro', markersize=2)

                circle = Circle((x, y), r, color='r', alpha=a)
                ax[0].add_patch(circle)
                circle = Circle((x, y), r, color='r', alpha=a)
                ax[1].add_patch(circle)
                # circle = Circle((x, y), 0.3, color='r', alpha=0.1)
                # ax[0].add_patch(circle)
                if j == 0:
                    x_text_offset = text_offset
                    y_text_offset = text_offset
                else:
                    theta = math.atan2(y-plot_traj[i][j-1][1],x-plot_traj[i][j-1][0])
                    x_text_offset = 1.414*text_offset*math.cos(theta)
                    y_text_offset = 1.414*text_offset*math.sin(theta)

                # if traj_dist > 4.:
                #     if step == plot_traj[i][-1][2]:
                #         ax[0].text(x+text_offset, y+text_offset, f"{time_max_step*0.1:.1f}", fontsize=20, color='black') 
                #         ax[1].text(x+text_offset, y+text_offset, f"{time_max_step*0.1:.1f}", fontsize=20, color='black') 

                #     if step >= int(3.*plot_traj[i][-1][2]/4.) and not draw_flag1:
                #         draw_flag1 = True
                #         ax[0].text(x+text_offset, y+text_offset, f"{3.*time_max_step*0.1/4.:.1f}", fontsize=20, color='black') 
                #         ax[1].text(x+text_offset, y+text_offset, f"{3.*time_max_step*0.1/4:.1f}", fontsize=20, color='black') 

                #     if step >= int(plot_traj[i][-1][2]/2.) and not draw_flag2:
                #         draw_flag2 = True
                #         ax[0].text(x+text_offset, y+text_offset, f"{time_max_step*0.1/2.:.1f}", fontsize=20, color='black') 
                #         ax[1].text(x+text_offset, y+text_offset, f"{time_max_step*0.1/2.:.1f}", fontsize=20, color='black') 

                #     if step >= int(plot_traj[i][-1][2]/4.) and not draw_flag3:
                #         draw_flag3 = True
                #         ax[0].text(x+text_offset, y+text_offset, f"{time_max_step*0.1/4.:.1f}", fontsize=20, color='black') 
                #         ax[1].text(x+text_offset, y+text_offset, f"{time_max_step*0.1/4.:.1f}", fontsize=20, color='black') 

                # elif traj_dist > 3.:
                #     if step == plot_traj[i][-1][2]:
                #         ax[0].text(x+text_offset, y+text_offset, f"{time_max_step*0.1:.1f}", fontsize=20, color='black') 
                #         ax[1].text(x+text_offset, y+text_offset, f"{time_max_step*0.1:.1f}", fontsize=20, color='black') 
                #     if step >= int(2*plot_traj[i][-1][2]/3.) and not draw_flag2:
                #         draw_flag2 = True
                #         ax[0].text(x+text_offset, y+text_offset, f"{2*time_max_step*0.1/3.:.1f}", fontsize=20, color='black') 
                #         ax[1].text(x+text_offset, y+text_offset, f"{2*time_max_step*0.1/3.:.1f}", fontsize=20, color='black') 
                #     if step >= int(plot_traj[i][-1][2]/3.) and not draw_flag3:
                #         draw_flag3 = True
                #         ax[0].text(x+text_offset, y+text_offset, f"{time_max_step*0.1/3.:.1f}", fontsize=20, color='black') 
                #         ax[1].text(x+text_offset, y+text_offset, f"{time_max_step*0.1/3.:.1f}", fontsize=20, color='black') 

                # elif traj_dist > 2.:
                #     if step == plot_traj[i][-1][2]:
                #         ax[0].text(x+text_offset, y+text_offset, f"{time_max_step*0.1:.1f}", fontsize=20, color='black') 
                #         ax[1].text(x+text_offset, y+text_offset, f"{time_max_step*0.1:.1f}", fontsize=20, color='black') 
                #     if step >= int(plot_traj[i][-1][2]/2.) and not draw_flag1:
                #         draw_flag1 = True
                #         ax[0].text(x+text_offset, y+text_offset, f"{time_max_step*0.1/2.:.1f}", fontsize=20, color='black') 
                #         ax[1].text(x+text_offset, y+text_offset, f"{time_max_step*0.1/2.:.1f}", fontsize=20, color='black') 
                # else:
                #     if step == plot_traj[i][-1][2]:
                #         ax[0].text(x+text_offset, y+text_offset, f"{time_max_step*0.1:.1f}", fontsize=20, color='black') 
                #         ax[1].text(x+text_offset, y+text_offset, f"{time_max_step*0.1:.1f}", fontsize=20, color='black') 

                if traj_dist > 2.5:
                    if step == plot_traj[i][-1][2]:
                        ax[0].text(x+x_text_offset, y+y_text_offset, f"{step*0.1:.1f}", fontsize=20, color='black') 
                        ax[1].text(x+x_text_offset, y+y_text_offset, f"{step*0.1:.1f}", fontsize=20, color='black') 
                    if step >= int(plot_traj[i][-1][2]/2.) and not draw_flag1:
                        draw_flag1 = True
                        ax[0].text(x+x_text_offset, y+y_text_offset, f"{step*0.1:.1f}", fontsize=20, color='black') 
                        ax[1].text(x+x_text_offset, y+y_text_offset, f"{step*0.1:.1f}", fontsize=20, color='black') 
                else:
                    if step == plot_traj[i][-1][2]:
                        ax[0].text(x+x_text_offset, y+y_text_offset, f"{step*0.1:.1f}", fontsize=20, color='black') 
                        ax[1].text(x+x_text_offset, y+y_text_offset, f"{step*0.1:.1f}", fontsize=20, color='black') 

                if j != 0:
                    dis_sq = (x-plot_traj[i][j-1][0])**2 + (y-plot_traj[i][j-1][1])**2
                    if dis_sq > (0.4)**2:
                        theta = math.atan2(y-plot_traj[i][j-1][1],x-plot_traj[i][j-1][0])
                        res = 0.2
                        dd = math.sqrt(dis_sq)/res
                        for k in range(1,math.ceil(dd)+1):
                            px = plot_traj[i][j-1][0] + k*res*math.cos(theta)
                            py = plot_traj[i][j-1][1] + k*res*math.sin(theta)
                            circle = Circle((px, py), r, color='r', alpha=a)
                            ax[0].add_patch(circle)
                            ax[0].plot(px, py, 'ro', markersize=3)
                            circle = Circle((px, py), r, color='r', alpha=a)
                            ax[1].add_patch(circle)
                            ax[1].plot(px, py, 'ro', markersize=3)

                a += alpha_step

            if i == len(plot_traj)-1:
                dyl = ax[0].plot([x for x, _, _ ,_ in plot_traj[i]], [y for _, y, _ ,_ in plot_traj[i]], 'r-', linewidth=1,label='动态障碍物')
                dyl = ax[1].plot([x for x, _, _ ,_ in plot_traj[i]], [y for _, y, _ ,_ in plot_traj[i]], 'r-', linewidth=1)
            else:
                ax[0].plot([x for x, _, _ ,_ in plot_traj[i]], [y for _, y, _ ,_ in plot_traj[i]], 'r-', linewidth=1)
                ax[1].plot([x for x, _, _ ,_ in plot_traj[i]], [y for _, y, _ ,_ in plot_traj[i]], 'r-', linewidth=1)

        a = 0.05
        alpha_step  = (0.85-0.05)/len(teb_x_trajs[eps])
        num1 = 0
        for step,p in enumerate(zip(teb_x_trajs[eps],teb_y_trajs[eps])):
            x,y = p[0],p[1]
            circle = Circle((x, y), 0.225, color='b', alpha=a)  # Add circle for robot radius
            ax[0].add_patch(circle)
            ax[0].plot(x, y, 'bo', markersize=3)

            if step == 0:
                x_text_offset = text_offset
                y_text_offset = text_offset
            else:
                theta = math.atan2(y-teb_y_trajs[eps][step-1],x-teb_x_trajs[eps][step-1])
                x_text_offset = 1.414*text_offset*math.cos(theta)
                y_text_offset = 1.414*text_offset*math.sin(theta)

            if len(teb_x_trajs[eps]) >= 20:
                if num1 == len(teb_x_trajs[eps]) - 1:
                    ax[0].text(x+x_text_offset, y+y_text_offset, f"{teb_max_step*0.1:.1f}", fontsize=20, color='green') 
                if num1 == int(3.*(len(teb_x_trajs[eps]) - 1)/4.):
                    ax[0].text(x+x_text_offset, y+y_text_offset, f"{3.*teb_max_step*0.1/4.:.1f}", fontsize=20, color='green') 
                if num1 == int((len(teb_x_trajs[eps]) - 1)/2.):
                    ax[0].text(x+x_text_offset, y+y_text_offset, f"{teb_max_step*0.1/2.:.1f}", fontsize=20, color='green') 
                if num1 == int((len(teb_x_trajs[eps]) - 1)/4.):
                    ax[0].text(x+x_text_offset, y+y_text_offset, f"{teb_max_step*0.1/4.:.1f}", fontsize=20, color='green') 
            elif len(teb_x_trajs[eps]) >= 15:
                if num1 == len(teb_x_trajs[eps]) - 1:
                    ax[0].text(x+x_text_offset, y+y_text_offset, f"{teb_max_step*0.1:.1f}", fontsize=20, color='green') 
                if num1 == int((2*len(teb_x_trajs[eps]) - 1)/3.):
                    ax[0].text(x+x_text_offset, y+y_text_offset, f"{2*teb_max_step*0.1/3.:.1f}", fontsize=20, color='green') 
                if num1 == int((len(teb_x_trajs[eps]) - 1)/3.):
                    ax[0].text(x+x_text_offset, y+y_text_offset, f"{teb_max_step*0.1/3.:.1f}", fontsize=20, color='green') 
            elif len(teb_x_trajs[eps]) >= 10:
                if num1 == len(teb_x_trajs[eps]) - 1:
                    ax[0].text(x+x_text_offset, y+y_text_offset, f"{teb_max_step*0.1:.1f}", fontsize=20, color='green') 

                if num1 == int((len(teb_x_trajs[eps]) - 1)/2.):
                    ax[0].text(x+x_text_offset, y+y_text_offset, f"{teb_max_step*0.1/2.:.1f}", fontsize=20, color='green') 
            else:
                if num1 == len(teb_x_trajs[eps]) - 1:
                    ax[0].text(x+x_text_offset, y+y_text_offset, f"{teb_max_step*0.1:.1f}", fontsize=20, color='green') 

            # 插补
            if step != 0:
                dis_sq = (x-teb_x_trajs[eps][step-1])**2+(y-teb_y_trajs[eps][step-1])**2
                if dis_sq > (0.4)**2:
                    theta = math.atan2(y-teb_y_trajs[eps][step-1],x-teb_x_trajs[eps][step-1])
                    res = 0.2
                    dd = math.sqrt(dis_sq)/res

                    for i in range(1,math.floor(dd)+1):
                        x = teb_x_trajs[eps][step-1] + i*res*math.cos(theta)
                        y = teb_y_trajs[eps][step-1] + i*res*math.sin(theta)
                        circle = Circle((x, y), 0.225, color='b', alpha=a)  # Add circle for robot radius
                        ax[0].add_patch(circle)
                        ax[0].plot(x, y, 'bo', markersize=3)
            
            a += alpha_step
            num1 += 1
        rbl = ax[0].plot(teb_x_trajs[eps], teb_y_trajs[eps], 'b-', linewidth=1,label='机器人')  # Connect the trajectory points
        gol = ax[0].scatter(goal_positions[eps][0],goal_positions[eps][1],color=(0,1,0,1), marker='*',s = 300,label='终点',zorder=2)
        # gol = ax[0].scatter(rl_x_trajs[eps][-1],rl_y_trajs[eps][-1],color=(0,1,0,1), marker='*',s = 300,label='终点',zorder=2)

        if not teb_success:
            col = ax[0].scatter(teb_x_trajs[eps][-1],teb_y_trajs[eps][-1],color=(0,0,0,1), marker='X',s = 200,label='碰撞',zorder=2)
        else:
            circle = Circle((goal_positions[eps][0],goal_positions[eps][1]), 0.225, color='b', alpha=a)
            ax[0].add_patch(circle)
            ax[0].plot(goal_positions[eps][0],goal_positions[eps][1], 'bo', markersize=3)


        # ax[1].plot(start_positions[eps][0],start_positions[eps][1], 'rs', markersize=10)
        # ax[1].text(start_positions[eps][0]+text_offset,start_positions[eps][1]+text_offset, "START", fontsize=12, color='red')
        # ax[1].plot(goal_positions[eps][0],goal_positions[eps][1], 'bs', markersize=10)
        # ax[1].scatter(goal_positions[eps][0],goal_positions[eps][1],c=(0,1,0,1), marker='*',s = 100)
        # ax[1].text(goal_positions[eps][0]+text_offset,goal_positions[eps][1]+text_offset, "GOAL", fontsize=12, color='blue')
        # circle = Circle((goal_positions[eps][0],goal_positions[eps][1]), 1, color='r', alpha=0.1)  # Add circle for robot radius
        # ax[1].add_patch(circle)


        if rl_success:
            rl_x_trajs[eps].append(goal_positions[eps][0])
            rl_y_trajs[eps].append(goal_positions[eps][1])
        # Now we can plot the polygon points and control network points for each step
        # num1 = 0
        # num2 = 0
        # for step in sorted(polygon_points_by_steps[eps].keys()):
            
        #     for polygon_points in polygon_points_by_steps[eps][step]:

        #         if num1 == 0 and num2 == 0:
        #             cnl = ax[1].plot(*polygon_points.T, 'g-', linewidth=1,label='安全凸空间')
        #         else:
        #             ax[1].plot(*polygon_points.T, 'g-', linewidth=1)  # Draw the polygon
        #         ax[1].plot(*polygon_points.T, 'go', markersize=3)  # Mark the points
        #         num2 += 1
        #     num1 += 1

        # for i in range(len(plot_traj)):
        #     a = 0.05
        #     alpha_step  = (0.5-0.05)/len(plot_traj[i])
        #     for j in range(len(plot_traj[i])):
        #         p = plot_traj[i][j]
        #         x,y,step,r = p[0],p[1],p[2],p[3]
        #         ax[1].plot(x, y, 'ro', markersize=2)
        #         circle = Circle((x, y), r, color='r', alpha=a)
        #         ax[1].add_patch(circle)
        #         # circle = Circle((x, y), 0.3, color='r', alpha=0.1)
        #         # ax[0].add_patch(circle)
        #         if step == plot_traj[i][-1][2]:
        #             ax[1].text(x+text_offset, y+text_offset, f"{step*0.1:.1f}", fontsize=20, color='black')

        #         if j != 0:
        #             dis_sq = (x-plot_traj[i][j-1][0])**2 +(y-plot_traj[i][j-1][1])**2
        #             if dis_sq > (0.4)**2:
        #                 theta = math.atan2(y-plot_traj[i][j-1][1],x-plot_traj[i][j-1][0])
        #                 res = 0.2
        #                 dd = math.sqrt(dis_sq)/res
        #                 for k in range(1,math.ceil(dd)+1):
        #                     px = plot_traj[i][j-1][0] + k*res*math.cos(theta)
        #                     py = plot_traj[i][j-1][1] + k*res*math.sin(theta)
        #                     circle = Circle((px, py), r, color='r', alpha=a)
        #                     ax[1].add_patch(circle)
        #                     ax[1].plot(px, py, 'ro', markersize=3)


        #         a += alpha_step
        #     ax[1].plot([x for x, _, _ ,_ in plot_traj[i]], [y for _, y, _,_ in plot_traj[i]], 'r-', linewidth=1)


        
        a = 0.05
        alpha_step  = (0.85-0.05)/len(rl_x_trajs[eps])
        num1 = 0
        for step,p in enumerate(zip(rl_x_trajs[eps],rl_y_trajs[eps])):
            x,y = p[0],p[1]
            circle = Circle((x, y), 0.225, color='b', alpha=a)  # Add circle for robot radius
            ax[1].add_patch(circle)
            ax[1].plot(x, y, 'bo', markersize=3)
            if step == 0:
                x_text_offset = text_offset
                y_text_offset = text_offset
            else:
                theta = math.atan2(y-rl_y_trajs[eps][step-1],x-rl_x_trajs[eps][step-1])
                x_text_offset = 1.414*text_offset*math.cos(theta)
                y_text_offset = 1.414*text_offset*math.sin(theta)

            if len(rl_x_trajs[eps]) >= 20:
                if num1 == len(rl_x_trajs[eps]) - 1:
                    ax[1].text(x+x_text_offset, y+y_text_offset, f"{rl_max_step*0.1:.1f}", fontsize=20, color='green') 
                if num1 == int(3.*(len(rl_x_trajs[eps]) - 1)/4.):
                    ax[1].text(x+x_text_offset, y+y_text_offset, f"{3.*rl_max_step*0.1/4.:.1f}", fontsize=20, color='green') 
                if num1 == int((len(rl_x_trajs[eps]) - 1)/2.):
                    ax[1].text(x+x_text_offset, y+y_text_offset, f"{rl_max_step*0.1/2.:.1f}", fontsize=20, color='green') 
                if num1 == int((len(rl_x_trajs[eps]) - 1)/4.):
                    ax[1].text(x+x_text_offset, y+y_text_offset, f"{rl_max_step*0.1/4.:.1f}", fontsize=20, color='green') 
            elif len(rl_x_trajs[eps]) >= 15:
                if num1 == len(rl_x_trajs[eps]) - 1:
                    ax[1].text(x+x_text_offset, y+y_text_offset, f"{rl_max_step*0.1:.1f}", fontsize=20, color='green') 
                if num1 == int((2*len(rl_x_trajs[eps]) - 1)/3.):
                    ax[1].text(x+x_text_offset, y+y_text_offset, f"{2*rl_max_step*0.1/3.:.1f}", fontsize=20, color='green') 
                if num1 == int((len(rl_x_trajs[eps]) - 1)/3.):
                    ax[1].text(x+x_text_offset, y+y_text_offset, f"{rl_max_step*0.1/3.:.1f}", fontsize=20, color='green') 
            elif len(rl_x_trajs[eps]) >= 10:
                if num1 == len(rl_x_trajs[eps]) - 1:
                    ax[1].text(x+x_text_offset, y+y_text_offset, f"{rl_max_step*0.1:.1f}", fontsize=20, color='green') 

                if num1 == int((len(rl_x_trajs[eps]) - 1)/2.):
                    ax[1].text(x+x_text_offset, y+y_text_offset, f"{rl_max_step*0.1/2.:.1f}", fontsize=20, color='green') 
            else:
                if num1 == len(rl_x_trajs[eps]) - 1:
                    ax[1].text(x+x_text_offset, y+y_text_offset, f"{rl_max_step*0.1:.1f}", fontsize=20, color='green') 

            if step != 0:
                dis_sq = (x-rl_x_trajs[eps][step-1])**2+(y-rl_y_trajs[eps][step-1])**2
                if dis_sq > (0.4)**2:
                    theta = math.atan2(y-rl_y_trajs[eps][step-1],x-rl_x_trajs[eps][step-1])
                    res = 0.2
                    dd = math.sqrt(dis_sq)/res - 1
                    
                    for i in range(1,math.floor(dd)+1):
                        x = rl_x_trajs[eps][step-1] + i*res*math.cos(theta)
                        y = rl_y_trajs[eps][step-1] + i*res*math.sin(theta)
                        circle = Circle((x, y), 0.225, color='b', alpha=a)  # Add circle for robot radius
                        ax[1].add_patch(circle)
                        ax[1].plot(x, y, 'bo', markersize=3)

            a += alpha_step
            num1 += 1
        ax[1].plot(rl_x_trajs[eps], rl_y_trajs[eps], 'b-', linewidth=1)  # Connect the trajectory points
        ax[1].scatter(goal_positions[eps][0],goal_positions[eps][1],color=(0,1,0,1), marker='*',s = 300,zorder=2)
        # ax[1].scatter(rl_x_trajs[eps][-1],rl_y_trajs[eps][-1],color=(0,1,0,1), marker='*',s = 300,zorder=2)

        # circle = Circle((goal_positions[eps][0],goal_positions[eps][1]), 0.225, color='b', alpha=a)
        # ax[1].add_patch(circle)
        # ax[1].plot(goal_positions[eps][0],goal_positions[eps][1], 'bo', markersize=3)

        # num1 = 0
        # num2 = 0
        # for step in sorted(ctrl_network_points_by_steps[eps].keys()):
        #     for ctrl_network_points in ctrl_network_points_by_steps[eps][step]:
        #         ax[1].plot(*ctrl_network_points.T, 'y-', linewidth=1)  # Draw the points connected
        #         if num1 == 0 and num2 == 0:
        #             refl = ax[1].plot(*ctrl_network_points.T, 'yo', markersize=3,label='参考位置点')
        #         else:
        #             ax[1].plot(*ctrl_network_points.T, 'yo', markersize=3)  # Mark the points
        #         num2 += 1
        #     num1 += 1

        # 超出 终点部分
        def on_key_press(event):
            if event.key == 'd':
                if not os.path.exists(save_path+"plots"):
                    os.makedirs(save_path+"plots")
                plt.savefig(f"{save_path}plots/plot_eps_{data_eps}.png")
                print(f'Image saved to {save_path}plots/plot_eps_{data_eps}.png')
                plt.close(fig)
            else :
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key_press)
        plt.rcParams['legend.title_fontsize'] = 20
        # fig.legend(handles=[dyl[0], rbl[0],gol,cnl[0],refl[0]], labels=['动态障碍物', '机器人','终点','安全凸空间','参考位置点'], loc='upper right', prop=font)
        plt.show()


def plot_convex(teb_info,teb_plotter,rl_info,rl_plotter):
    global xlim
    global ylim
    global save_path
    global font
    global eps_list
    teb_x_trajs,teb_y_trajs,teb_max_steps,teb_obstacle_trajectorys,start_positions,goal_positions = teb_info[0],teb_info[1],teb_info[2],teb_info[3],teb_info[4],teb_info[5]
    rl_x_trajs,rl_y_trajs,rl_max_steps,rl_obstacle_trajectorys,start_positions,goal_positions,polygon_points_by_steps,ctrl_network_points_by_steps = rl_info[0],rl_info[1],rl_info[2],rl_info[3],rl_info[4],rl_info[5],rl_info[6],rl_info[7]

    text_offset = 0.15
    
    for eps in range(0,len(teb_obstacle_trajectorys)):
        teb_max_step = teb_max_steps[eps]
        rl_max_step = rl_max_steps[eps]

        if teb_max_step > rl_max_step:
            plot_traj = teb_obstacle_trajectorys[eps]
        else:
            plot_traj = rl_obstacle_trajectorys[eps]

        close_num = 0
        proj_on_line = False
        proj_on_line_num = 0

        # for i in range(len(plot_traj)):
        #     if len(plot_traj[i]) == 0:
        #         continue
        #     s_proj_on_line,sd = point_to_segment_distance((plot_traj[i][0][0],plot_traj[i][0][1]),(start_positions[eps],goal_positions[eps]))
        #     if s_proj_on_line:
        #         proj_on_line_num += 1
        #     f_proj_on_line,fd = point_to_segment_distance((plot_traj[i][-1][0],plot_traj[i][-1][1]),(start_positions[eps],goal_positions[eps]))
        #     if f_proj_on_line:
        #         proj_on_line_num += 1 
        #     # s_proj_on_line and f_proj_on_line and sd - fd > 0.5
        #     if sd - fd > 0 and sd < 5 and fd < 5:
        #         close_num += 1
 
        # if close_num < 15 :# or proj_on_line_num < 10
        #     continue

        res = rl_plotter.stats.get_eps_stats(eps_list[eps]+1)
        if res is not None:
            rl_success,time, distance, velocity = res
            print(f'RL Eps {eps_list[eps]} - success: {rl_success}, time: {time}, distance: {distance}, velocity: {velocity}')
        else:
            continue

        res = teb_plotter.stats.get_eps_stats(eps_list[eps]+1)
        if res is not None:
            teb_success,time, distance, velocity = res
            print(f'TEB Eps {eps_list[eps]} - success: {teb_success}, time: {time}, distance: {distance}, velocity: {velocity}')
        else:
            continue

        if not rl_success and teb_success:
            continue
        
        fig, ax = plt.subplots()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal') # Set 1:1 aspect ratio
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        # ax[0].grid()
        # ax.set_title(f"TEB")


        if rl_success:
            rl_x_trajs[eps].append(goal_positions[eps][0])
            rl_y_trajs[eps].append(goal_positions[eps][1])


        # Now we can plot the polygon points and control network points for each step
        num1 = 0
        num2 = 0
        a = 0.2
        alpha_step  = (1.-0.2)/len(ctrl_network_points_by_steps[eps].keys())
        for step in sorted(polygon_points_by_steps[eps].keys()):
            
            for polygon_points in polygon_points_by_steps[eps][step]:

                if num1 == 0 and num2 == 0:
                    cnl = ax.plot(*polygon_points.T, 'k-', linewidth=2.5,label='安全凸空间') # ,alpha = a
                else:
                    ax.plot(*polygon_points.T, 'k-', linewidth=2.5)  # Draw the polygon
                ax.plot(*polygon_points.T, 'ko', markersize=3)  # Mark the points
                num2 += 1
            a += alpha_step
            num1 += 1

        for i in range(len(plot_traj)):
            a = 0.05
            alpha_step  = (0.85-0.05)/len(plot_traj[i])
            for j in range(len(plot_traj[i])):
                p = plot_traj[i][j]
                x,y,step,r = p[0],p[1],p[2],p[3]
                ax.plot(x, y, 'ro', markersize=2)
                circle = Circle((x, y), r, color='r', alpha=a)
                ax.add_patch(circle)
                # circle = Circle((x, y), 0.3, color='r', alpha=0.1)
                # ax[0].add_patch(circle)

                if j != 0:
                    dis_sq = (x-plot_traj[i][j-1][0])**2 + (y-plot_traj[i][j-1][1])**2
                    if dis_sq > (0.4)**2:
                        theta = math.atan2(y-plot_traj[i][j-1][1],x-plot_traj[i][j-1][0])
                        res = 0.2
                        dd = math.sqrt(dis_sq)/res
                        for k in range(1,math.ceil(dd)+1):
                            px = plot_traj[i][j-1][0] + k*res*math.cos(theta)
                            py = plot_traj[i][j-1][1] + k*res*math.sin(theta)
                            circle = Circle((px, py), r, color='r', alpha=a)
                            ax.add_patch(circle)
                            ax.plot(px, py, 'ro', markersize=3)

                if step == plot_traj[i][-1][2]:
                    ax.text(x+text_offset, y+text_offset, f"{step*0.1:.1f}", fontsize=20, color='black')
                a += alpha_step
            ax.plot([x for x, _, _ ,_ in plot_traj[i]], [y for _, y, _,_ in plot_traj[i]], 'r-', linewidth=1)

        a = 0.05
        alpha_step  = (0.85-0.05)/len(rl_x_trajs[eps])
        num1 = 0
        for step,p in enumerate(zip(rl_x_trajs[eps],rl_y_trajs[eps])):
            x,y = p[0],p[1]
            circle = Circle((x, y), 0.225, color='b', alpha=a)  # Add circle for robot radius
            ax.add_patch(circle)
            ax.plot(x, y, 'bo', markersize=3)
            if num1 == len(rl_x_trajs[eps]) - 1:
                ax.text(x+text_offset, y+text_offset, f"{rl_max_step*0.1:.1f}", fontsize=20, color='green') 
            if step != 0:
                dis_sq = (x-rl_x_trajs[eps][step-1])**2+(y-rl_y_trajs[eps][step-1])**2
                if dis_sq > (0.4)**2:
                    theta = math.atan2(y-rl_y_trajs[eps][step-1],x-rl_x_trajs[eps][step-1])
                    res = 0.2
                    dd = math.sqrt(dis_sq)/res - 1
                    
                    for i in range(1,math.floor(dd)+1):
                        x = rl_x_trajs[eps][step-1] + i*res*math.cos(theta)
                        y = rl_y_trajs[eps][step-1] + i*res*math.sin(theta)
                        circle = Circle((x, y), 0.225, color='b', alpha=a)  # Add circle for robot radius
                        ax.add_patch(circle)
                        ax.plot(x, y, 'bo', markersize=3)
            #             if ((goal_positions[eps][0]-x)+(goal_positions[eps][1]-y))**2 < 0.3**2:
            #                 ax[1].text(x+text_offset, y+text_offset, f"{int(step)}", fontsize=20, color='green') 
            #                 break
            # if ((goal_positions[eps][0]-x)+(goal_positions[eps][1]-y))**2 < 0.3**2:
            #     ax[1].text(x+text_offset, y+text_offset, f"{int(step)}", fontsize=20, color='green') 
            #     break


            a += alpha_step
            num1 += 1
        ax.plot(rl_x_trajs[eps], rl_y_trajs[eps], 'b-', linewidth=1)  # Connect the trajectory points
        ax.scatter(goal_positions[eps][0],goal_positions[eps][1],color=(0,1,0,1), marker='*',s = 300,zorder=2)

        num1 = 0
        num2 = 0
        a = 0.2
        alpha_step  = (1.-0.2)/len(ctrl_network_points_by_steps[eps].keys())
        for step in sorted(ctrl_network_points_by_steps[eps].keys()):
            
            for ctrl_network_points in ctrl_network_points_by_steps[eps][step]:
                ax.plot(*ctrl_network_points.T, 'go', markersize=3)
                if num1 == 0 and num2 == 0:
                    refl = ax.plot(*ctrl_network_points.T, 'g-', linewidth=3,label='参考位置点')
                else:
                    ax.plot(*ctrl_network_points.T, 'g-', linewidth=3)
                num2 += 1
            a += alpha_step
            num1 += 1

        def on_key_press(event):
            if event.key == 'd':
                if not os.path.exists(save_path+"plots"):
                    os.makedirs(save_path+"plots")
                plt.savefig(f"{save_path}plots/plot_eps_{data_eps}.png")
                print(f'Image saved to {save_path}plots/plot_eps_{data_eps}.png')
                plt.close(fig)
            else :
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key_press)
        # plt.rcParams['legend.title_fontsize'] = 20
        # plt.legend(handles=[cnl[0],refl[0]], labels=['安全凸空间','参考位置点'], prop=font ) # , loc='upper right' ,fontsize=80
        plt.show()




def main():
    rl_plotter = rl_DataPlotter(obs_num,xlim = xlim,ylim = ylim,eps_list = eps_list)
    teb_plotter = teb_DataPlotter(obs_num,xlim = xlim,ylim = ylim,eps_list = eps_list)
    plotters = [rl_plotter,teb_plotter]
    rl_infos = [rl_plotter.run(),teb_plotter.run()]

    plot_rl(rl_infos,plotters)


if __name__ == "__main__":
    main()
