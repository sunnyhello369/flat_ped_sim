import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Circle
from matplotlib.markers import MarkerStyle
import math
from rl2_data_plot_json import rl_DataPlotter
from copy import deepcopy


def calc_cross_point(a1, b1, c1, a2, b2, c2):
    D = a1 * b2 - a2 * b1
    print(D)
    if abs(D) < 1e-6:  # 平行
        return None
    return ((b1 * c2 - b2 * c1) / D, (a2 * c1 - a1 * c2) / D)


# 导入math模块，用于计算平方根和绝对值
import math


def point_to_segment_distance(point, segment):
    # 将参数解构为坐标值
    x0, y0 = point  # 点的坐标
    x1, y1 = segment[0]  # 线段的第一个端点坐标
    x2, y2 = segment[1]  # 线段的第二个端点坐标

    # 计算线段的长度
    segment_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # 如果线段长度为零，说明线段退化为一个点，直接返回点到点的距离
    if segment_length < 1e-6:
        return False, math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

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
        return True, distance_to_line
    else:
        # 投影点不在线段上，返回点到线段两个端点中较近的一个的距离
        return False, min(math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2), math.sqrt((x0 - x2) ** 2 + (y0 - y2) ** 2))


xlim = (-5.5, 26.5)
ylim = (-5.5, 19.5)
save_path = "D:/RL/_1repertory_log/230616_data/"
font = fm.FontProperties(fname="C:/Windows/Fonts/simsun.ttc")  # times.ttf
# d10
# eps_list = [308,350,538,632,760,776,890] # 155i 201i 254 308i 692i 746i 896
# eps_list = [155, 201, 254, 308, 692, 746,760,890, 896]
# d20
# eps_list = [39,157,474,644,685] # 39 238 157 282i 474 685 960
# eps_list = [157]
# d30
# eps_list = [378,413,515,547,771] # 75,167,377,413,515!,547!,771!
# eps_list = [i for i in range(1000)]

scence = "s10"
# scence = "s10d5"
# scence = "s10d10"
path = ["flat_ped_sim_0502_sparse_polar2p_limit/" + scence, "flat_ped_sim_0430_e2e/" + scence,
        "flat_ped_sim_0424_convex_vel/" + scence, "flat_ped_sim_0506polar1p/" + scence,
        "flat_ped_sim_0510_scan2p/" + scence]
# s10
# eps_list = [123,155,161,211,238,249,262,265,316,322,341,365,383] # 249,316,341通道
# eps_list = [249,262,316,322,365] # 249,316,341通道
# eps_list = [i for i in range(1,1000,10)]
eps_list = [123]
# s10d5
# eps_list = [70,87,215,253,263] # 88! 279
# eps_list = [i for i in range(1,1000,10)]
# eps_list = [231]
# s10d10
# eps_list = [5,51,64,66,94,116,184,202,203,213,234,270,330,371] # 67通道 51i 126 178 203 214 234 276i 330 400i
# eps_list = [5,51,66,116,184,202,203,213,234,270,330,371] # 270 ok 330 ok
# eps_list = [i for i in range(1,1000,9)]
# eps_list = [51, 202, 234, 270, 330]
# eps_list = [330]


def plot_rl(rl_infos, rl_plotters):
    global xlim
    global ylim
    global save_path
    global font
    global eps_list

    text_offset = 0.15
    check = False
    copy_rl_infos = deepcopy(rl_infos)
    copy_rl_plotters = deepcopy(rl_plotters)

    for eps in range(0, len(copy_rl_infos[0][0])):  # rl_infos[0][0] 本方法rl_x_trajs
        stats = []
        res = copy_rl_plotters[0].stats.get_eps_stats(eps_list[eps] + 1)
        if res is not None:
            rl_success, time, distance, velocity = res
            print(
                f'RL Eps {eps_list[eps] + 1} - success: {rl_success}, time: {time}, distance: {distance}, velocity: {velocity}')
        else:
            continue

        # if not rl_success : 
        #     continue

        success_num = 1
        stats.append((rl_success, time, distance, velocity))
        for i in range(1, len(copy_rl_plotters)):
            res = copy_rl_plotters[i].stats.get_eps_stats(eps_list[eps] + 1)
            if res is not None:
                rl_success, time, distance, velocity = res
                print(
                    f'RL {i} Eps {eps_list[eps] + 1} - success: {rl_success}, time: {time}, distance: {distance}, velocity: {velocity}')
            stats.append((rl_success, time, distance, velocity))
            if rl_success:
                success_num += 1


        # if success_num > 3:
        #     continue
        # if check and stats[0][0]:
        #     check = not check
        #     continue
        # check = not check

        time_max_step = -1
        for rl_info in copy_rl_infos:
            rl_x_trajs, rl_y_trajs, rl_max_steps, rl_obstacle_trajectorys, start_positions, goal_positions, polygon_points_by_steps, ctrl_network_points_by_steps, static_obss, traj_steps = rl_info
            rl_max_step = rl_max_steps[eps]
            if rl_max_step > time_max_step:
                time_max_step = rl_max_step
                plot_traj = rl_obstacle_trajectorys[eps]

        # size1 = 22
        # size2 = 12
        # size3 = 15
        # fig, axs = plt.subplots(2, 3,constrained_layout=False, gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1, 1]}) # , sharex=True, sharey=True
        # # fig, axs = plt.subplots(2, 3, gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1, 1]}) # , sharex=True, sharey=True
        # axs[0,0].set_xlim(xlim)
        # axs[0,0].set_ylim(ylim)
        # axs[0,0].set_aspect('equal') # Set 1:1 aspect ratio
        # axs[0,0].set_xlabel("X")
        # axs[0,0].set_ylabel("Y")
        # axs[0,0].set_title("设计1", fontproperties=font,fontsize=30)
        # axs[0,1].set_xlim(xlim)
        # axs[0,1].set_ylim(ylim)
        # axs[0,1].set_aspect('equal') # Set 1:1 aspect ratio
        # axs[0,1].set_xlabel("X")
        # axs[0,1].set_ylabel("Y")
        # axs[0,1].set_title("设计2", fontproperties=font,fontsize=30)
        # axs[0,2].set_xlim(xlim)
        # axs[0,2].set_ylim(ylim)
        # axs[0,2].set_aspect('equal') # Set 1:1 aspect ratio
        # axs[0,2].set_xlabel("X")
        # axs[0,2].set_ylabel("Y")
        # axs[0,2].set_title("设计3", fontproperties=font,fontsize=30)
        # axs[1,0].set_xlim(xlim)
        # axs[1,0].set_ylim(ylim)
        # axs[1,0].set_aspect('equal') # Set 1:1 aspect ratio
        # axs[1,0].set_xlabel("X")
        # axs[1,0].set_ylabel("Y")
        # axs[1,0].set_title("设计4", fontproperties=font,fontsize=30)
        # axs[1,2].set_xlim(xlim)
        # axs[1,2].set_ylim(ylim)
        # axs[1,2].set_aspect('equal') # Set 1:1 aspect ratio
        # axs[1,2].set_xlabel("X")
        # axs[1,2].set_ylabel("Y")
        # axs[1,2].set_title("本方法", fontproperties=font,fontsize=30)
        # fig.delaxes(axs[1, 1])
        # axs_index = [(0,0),(0,1),(0,2),(1,0),(1,2)]
        # pm = {0:(1,2),1:(0,0),2:(0,1),3:(0,2),4:(1,0)}

        # fig, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1]}) # , sharex=True, sharey=True
        # axs[0].set_xlim(xlim)
        # axs[0].set_ylim(ylim)
        # axs[0].set_aspect('equal') # Set 1:1 aspect ratio
        # axs[0].set_xlabel("X")
        # axs[0].set_ylabel("Y")
        # axs[0].set_title("设计1", fontproperties=font,fontsize=30)
        # axs[1].set_xlim(xlim)
        # axs[1].set_ylim(ylim)
        # axs[1].set_aspect('equal') # Set 1:1 aspect ratio
        # axs[1].set_xlabel("X")
        # axs[1].set_ylabel("Y")
        # axs[1].set_title("设计2", fontproperties=font,fontsize=30)
        # axs[2].set_xlim(xlim)
        # axs[2].set_ylim(ylim)
        # axs[2].set_aspect('equal') # Set 1:1 aspect ratio
        # axs[2].set_xlabel("X")
        # axs[2].set_ylabel("Y")
        # axs[2].set_title("设计3", fontproperties=font,fontsize=30)
        # axs_index = [0,1,2]
        # pm = {0:0,1:1,2:2}

        # size1 = 18
        # size2 = 12
        # size3 = 15
        # rl_infos = copy_rl_infos[1:5]
        # rl_plotters = copy_rl_plotters[1:5]
        # stats = stats[1:5]
        # fig, axs = plt.subplots(2, 2,figsize=(12.8, 10)) # , sharex=True, sharey=True , gridspec_kw={'height_ratios': [1, 1.1], 'width_ratios': [1, 1]}
        # axs[0,0].set_xlim(xlim)
        # axs[0,0].set_ylim(ylim)
        # axs[0,0].set_aspect('equal') # Set 1:1 aspect ratio
        # axs[0,0].set_xlabel("X[m]")
        # axs[0,0].set_ylabel("Y[m]")
        # axs[0,0].set_title("设计1", fontproperties=font,fontsize=size1)
        # axs[0,1].set_xlim(xlim)
        # axs[0,1].set_ylim(ylim)
        # axs[0,1].set_aspect('equal') # Set 1:1 aspect ratio
        # axs[0,1].set_xlabel("X[m]")
        # axs[0,1].set_ylabel("Y[m]")
        # axs[0,1].set_title("设计2", fontproperties=font,fontsize=size1)
        # axs[1,0].set_xlim(xlim)
        # axs[1,0].set_ylim(ylim)
        # axs[1,0].set_aspect('equal') # Set 1:1 aspect ratio
        # axs[1,0].set_xlabel("X[m]")
        # axs[1,0].set_ylabel("Y[m]")
        # axs[1,0].set_title("设计3", fontproperties=font,fontsize=size1)
        # axs[1,1].set_xlim(xlim)
        # axs[1,1].set_ylim(ylim)
        # axs[1,1].set_aspect('equal') # Set 1:1 aspect ratio
        # axs[1,1].set_xlabel("X[m]")
        # axs[1,1].set_ylabel("Y[m]")
        # axs[1,1].set_title("设计4", fontproperties=font,fontsize=size1)
        # axs_index = [(0,0),(0,1),(1,0),(1,1)]
        # pm = {0:(0,0),1:(0,1),2:(1,0),3:(1,1)}

        size1 = 40
        size2 = 20
        size3 = 26
        rl_infos = [copy_rl_infos[0]]
        rl_plotters = [copy_rl_plotters[0]]
        stats = [stats[0]]
        fig, ax_ = plt.subplots(1, 1, figsize=(12.8, 10))  # , sharex=True, sharey=True
        ax_.set_xlim(xlim)
        ax_.set_ylim(ylim)
        ax_.set_aspect('equal')  # Set 1:1 aspect ratio
        ax_.set_xlabel('X[m]', fontsize=19)
        ax_.set_ylabel('Y[m]', fontsize=19)
        ax_.tick_params(axis='x', labelsize=19)
        ax_.tick_params(axis='y', labelsize=19)
        ax_.set_title("本方法", fontproperties=font, fontsize=size1)
        axs = [ax_]
        axs_index = [0]
        pm = {0: 0}

        for polygon_points in static_obss[eps]:
            for ax in axs_index:
                # axs[ax].fill([p[0] for p in polygon_points], [p[1] for p in polygon_points], color="cornflowerblue")
                axs[ax].fill([p[0] for p in polygon_points], [p[1] for p in polygon_points], color=(0,1,0,0.5))
                axs[ax].plot(*polygon_points.T, 'k-', linewidth=1.5)  # Draw the polygon
                axs[ax].plot(*polygon_points.T, 'ko', markersize=1.5)  # Mark the points

        # Now we can plot the polygon points and control network points for each step

        for i in range(len(plot_traj)):
            a = 0.05
            if len(plot_traj[i]) != 0:
                alpha_step = (0.85 - 0.05) / len(plot_traj[i])
                f = plot_traj[i][0]
                l = plot_traj[i][-1]
                traj_dist = math.sqrt((f[0] - l[0]) ** 2 + (f[1] - l[1]) ** 2)
                draw_flag1 = False
                draw_flag2 = False
                draw_flag3 = False
            for j in range(len(plot_traj[i])):
                p = plot_traj[i][j]
                x, y, step, r = p[0], p[1], p[2], p[3]

                for ax in axs_index:
                    axs[ax].plot(x, y, 'ro', markersize=2)
                    # circle = Circle((x, y), r, color='r', alpha=a)
                    circle = Circle((x, y), r, color=(1,0,1), alpha=a)
                    axs[ax].add_patch(circle)
                if j == 0:
                    x_text_offset = text_offset
                    y_text_offset = text_offset
                else:
                    theta = math.atan2(y - plot_traj[i][j - 1][1], x - plot_traj[i][j - 1][0])
                    x_text_offset = 1.414 * text_offset * math.cos(theta)
                    y_text_offset = 1.414 * text_offset * math.sin(theta)

                if traj_dist > 2.5:
                    if step == plot_traj[i][-1][2]:
                        for ax in axs_index:
                            axs[ax].text(x + x_text_offset, y + y_text_offset, f"{step * 0.1:.1f}", fontsize=size2,
                                         color='black')

                    if step >= int(plot_traj[i][-1][2] / 2.) and not draw_flag1:
                        draw_flag1 = True
                        for ax in axs_index:
                            axs[ax].text(x + x_text_offset, y + y_text_offset, f"{step * 0.1:.1f}", fontsize=size2,
                                         color='black')
                else:
                    if step == plot_traj[i][-1][2]:
                        for ax in axs_index:
                            axs[ax].text(x + x_text_offset, y + y_text_offset, f"{step * 0.1:.1f}", fontsize=size2,
                                         color='black')

                if j != 0:
                    dis_sq = (x - plot_traj[i][j - 1][0]) ** 2 + (y - plot_traj[i][j - 1][1]) ** 2
                    if dis_sq > (0.4) ** 2:
                        theta = math.atan2(y - plot_traj[i][j - 1][1], x - plot_traj[i][j - 1][0])
                        res = 0.2
                        dd = math.sqrt(dis_sq) / res
                        for k in range(1, math.ceil(dd) + 1):
                            px = plot_traj[i][j - 1][0] + k * res * math.cos(theta)
                            py = plot_traj[i][j - 1][1] + k * res * math.sin(theta)
                            for ax in axs_index:
                                axs[ax].plot(x, y, 'ro', markersize=2)
                                # circle = Circle((x, y), r, color='r', alpha=a)
                                circle = Circle((x, y), r, color=(1,0,1), alpha=a)
                                axs[ax].add_patch(circle)
                a += alpha_step
            for ax in axs_index:
                axs[ax].plot([x for x, _, _, _ in plot_traj[i]], [y for _, y, _, _ in plot_traj[i]], 'r-', linewidth=1)

        for i, stat in enumerate(stats):
            rl_success, time, distance, velocity = stat
            if len(rl_infos[i][0][eps]) > 0:
                rl_infos[i][0][eps][0] = rl_infos[0][4][eps][0]
                rl_infos[i][1][eps][0] = rl_infos[0][4][eps][1]
            if rl_success:
                # goal_positions rl_infos[i][5]
                if len(rl_infos[i][0][eps]) > 0:
                    rl_infos[i][0][eps].append(rl_infos[i][5][eps][0])
                    rl_infos[i][1][eps].append(rl_infos[i][5][eps][1])
                    rl_infos[i][-1][eps].append(rl_infos[i][-1][eps][-1] + 1)
                # pass
            else:
                if len(rl_infos[i][0][eps]) > 2:
                    lx, ly = rl_infos[i][0][eps][-2], rl_infos[i][1][eps][-2]
                    x, y = rl_infos[i][0][eps][-1], rl_infos[i][1][eps][-1]
                    theta = math.atan2(y - ly, x - lx)
                    cx = x + velocity * 0.1 * math.cos(theta)
                    cy = y + velocity * 0.1 * math.sin(theta)
                    rl_infos[i][0][eps].append(cx)
                    rl_infos[i][1][eps].append(cy)
                    rl_infos[i][-1][eps].append(rl_infos[i][-1][eps][-1] + 1)

        for i, rl_info in enumerate(rl_infos):

            rl_x_trajs, rl_y_trajs, rl_max_steps, rl_obstacle_trajectorys, start_positions, goal_positions, polygon_points_by_steps, ctrl_network_points_by_steps, static_obss, traj_steps = rl_info
            a = 0.05
            if len(rl_x_trajs[eps]) != 0:
                alpha_step = (0.85 - 0.05) / len(rl_x_trajs[eps])
            num1 = 0

            axs[pm[i]].plot(rl_x_trajs[eps], rl_y_trajs[eps], 'b-', linewidth=1)  # Connect the trajectory points
            for step, p in enumerate(zip(rl_x_trajs[eps], rl_y_trajs[eps])):
                # print(stats[i][1],len(rl_x_trajs[eps])*0.1,rl_max_steps[eps]*0.1,len(traj_steps[eps]))

                x, y = p[0], p[1]
                circle = Circle((x, y), 0.225, color='b', alpha=a)  # Add circle for robot radius
                axs[pm[i]].add_patch(circle)
                axs[pm[i]].plot(x, y, 'bo', markersize=3)

                if step == 0:
                    x_text_offset = text_offset
                    y_text_offset = text_offset
                else:
                    theta = math.atan2(y - rl_y_trajs[eps][step - 1], x - rl_x_trajs[eps][step - 1])
                    x_text_offset = 1.414 * text_offset * math.cos(theta)
                    y_text_offset = 1.414 * text_offset * math.sin(theta)

                if len(rl_x_trajs[eps]) >= 20:
                    if num1 == len(rl_x_trajs[eps]) - 1:
                        axs[pm[i]].text(x + x_text_offset, y + y_text_offset, f"{traj_steps[eps][step] * 0.1:.1f}",
                                        fontsize=size3, color='#06C2AC', fontweight='bold')
                    if num1 == int(3. * (len(rl_x_trajs[eps]) - 1) / 4.):
                        axs[pm[i]].text(x + x_text_offset, y + y_text_offset, f"{traj_steps[eps][step] * 0.1:.1f}",
                                        fontsize=size3, color='#06C2AC', fontweight='bold')
                    if num1 == int((len(rl_x_trajs[eps]) - 1) / 2.):
                        axs[pm[i]].text(x + x_text_offset, y + y_text_offset, f"{traj_steps[eps][step] * 0.1:.1f}",
                                        fontsize=size3, color='#06C2AC', fontweight='bold')
                    if num1 == int((len(rl_x_trajs[eps]) - 1) / 4.):
                        axs[pm[i]].text(x + x_text_offset, y + y_text_offset, f"{traj_steps[eps][step] * 0.1:.1f}",
                                        fontsize=size3, color='#06C2AC', fontweight='bold')
                elif len(rl_x_trajs[eps]) >= 15:
                    if num1 == len(rl_x_trajs[eps]) - 1:
                        axs[pm[i]].text(x + x_text_offset, y + y_text_offset, f"{traj_steps[eps][step] * 0.1:.1f}",
                                        fontsize=size3, color='#06C2AC', fontweight='bold')
                    if num1 == int((2 * len(rl_x_trajs[eps]) - 1) / 3.):
                        axs[pm[i]].text(x + x_text_offset, y + y_text_offset, f"{traj_steps[eps][step] * 0.1:.1f}",
                                        fontsize=size3, color='#06C2AC', fontweight='bold')
                    if num1 == int((len(rl_x_trajs[eps]) - 1) / 3.):
                        axs[pm[i]].text(x + x_text_offset, y + y_text_offset, f"{traj_steps[eps][step] * 0.1:.1f}",
                                        fontsize=size3, color='#06C2AC', fontweight='bold')
                elif len(rl_x_trajs[eps]) >= 10:
                    if num1 == len(rl_x_trajs[eps]) - 1:
                        axs[pm[i]].text(x + x_text_offset, y + y_text_offset, f"{traj_steps[eps][step] * 0.1:.1f}",
                                        fontsize=size3, color='#06C2AC', fontweight='bold')

                    if num1 == int((len(rl_x_trajs[eps]) - 1) / 2.):
                        axs[pm[i]].text(x + x_text_offset, y + y_text_offset, f"{traj_steps[eps][step] * 0.1:.1f}",
                                        fontsize=size3, color='#06C2AC', fontweight='bold')
                else:
                    if num1 == len(rl_x_trajs[eps]) - 1:
                        axs[pm[i]].text(x + x_text_offset, y + y_text_offset, f"{traj_steps[eps][step] * 0.1:.1f}",
                                        fontsize=size3, color='#06C2AC', fontweight='bold')

                if step != 0:
                    dis_sq = (x - rl_x_trajs[eps][step - 1]) ** 2 + (y - rl_y_trajs[eps][step - 1]) ** 2
                    if dis_sq > (0.4) ** 2:
                        theta = math.atan2(y - rl_y_trajs[eps][step - 1], x - rl_x_trajs[eps][step - 1])
                        res = 0.2
                        dd = math.sqrt(dis_sq) / res - 1

                        for j in range(1, math.floor(dd) + 1):
                            x = rl_x_trajs[eps][step - 1] + j * res * math.cos(theta)
                            y = rl_y_trajs[eps][step - 1] + j * res * math.sin(theta)
                            circle = Circle((x, y), 0.225, color='b', alpha=a)  # Add circle for robot radius
                            axs[pm[i]].add_patch(circle)
                            axs[pm[i]].plot(x, y, 'bo', markersize=3)
                #             if ((goal_positions[eps][0]-x)+(goal_positions[eps][1]-y))**2 < 0.3**2:
                #                 ax[1].text(x+text_offset, y+text_offset, f"{int(step)}", fontsize=20, color='green') 
                #                 break
                # if ((goal_positions[eps][0]-x)+(goal_positions[eps][1]-y))**2 < 0.3**2:
                #     ax[1].text(x+text_offset, y+text_offset, f"{int(step)}", fontsize=20, color='green') 
                #     break

                a += alpha_step
                num1 += 1
            axs[pm[i]].scatter(goal_positions[eps][0], goal_positions[eps][1], color=(1, 0, 0, 1), marker='*', s=600,
                               zorder=2)

        # num1 = 0
        # num2 = 0
        # a = 0.2
        # rl_x_trajs,rl_y_trajs,rl_max_steps,rl_obstacle_trajectorys,start_positions,goal_positions,polygon_points_by_steps,ctrl_network_points_by_steps,static_obss,traj_steps = rl_infos[0]
        # alpha_step  = (1.-0.2)/len(ctrl_network_points_by_steps[eps].keys())
        # for step in sorted(polygon_points_by_steps[eps].keys()):

        #     for polygon_points in polygon_points_by_steps[eps][step]:

        #         if num1 == 0 and num2 == 0:
        #             cnl = axs[0].plot(*polygon_points.T, 'k-', linewidth=2.5,label='安全凸空间') # ,alpha = a
        #         else:
        #             axs[0].plot(*polygon_points.T, 'k-', linewidth=2.5)  # Draw the polygon
        #         axs[0].plot(*polygon_points.T, 'ko', markersize=3)  # Mark the points
        #         num2 += 1
        #     a += alpha_step
        #     num1 += 1

        # num1 = 0
        # num2 = 0
        # a = 0.2
        # rl_x_trajs,rl_y_trajs,rl_max_steps,rl_obstacle_trajectorys,start_positions,goal_positions,polygon_points_by_steps,ctrl_network_points_by_steps,static_obss,traj_steps = rl_infos[0]
        # alpha_step  = (1.-0.2)/len(ctrl_network_points_by_steps[eps].keys())
        # for step in sorted(ctrl_network_points_by_steps[eps].keys()):

        #     for ctrl_network_points in ctrl_network_points_by_steps[eps][step]:
        #         axs[0].plot(*ctrl_network_points.T, 'go', markersize=1)
        #         if num1 == 0 and num2 == 0:
        #             refl = axs[0].plot(*ctrl_network_points.T, 'g-', linewidth=0.5,label='参考路径',zorder=2)
        #         else:
        #             axs[0].plot(*ctrl_network_points.T, 'g-', linewidth=0.5,zorder=2)
        #         num2 += 1
        #     a += alpha_step
        #     num1 += 1

        def on_key_press(event):
            if event.key == 'd':
                if not os.path.exists(save_path + "plots"):
                    os.makedirs(save_path + "plots")
                plt.savefig(f"{save_path}plots/plot_eps_{eps_list[eps] + 1}.png")
                print(f'Image saved to {save_path}plots/plot_eps_{eps_list[eps] + 1}.png')
                plt.close(fig)
            else:
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key_press)
        # box = axs[1, 0].get_position() # 获取当前子图的位置
        # box.x0 = box.x0 + 0.125 # 将左下角的x坐标向右移动0.125
        # box.x1 = box.x1 + 0.125 # 将右上角的x坐标向右移动0.125

        # axs[1, 0].set_position(box) # 设置新的子图位置
        # box = axs[1, 2].get_position() # 获取当前子图的位置
        # box.x0 = box.x0 - 0.125 # 将左下角的x坐标向右移动0.125
        # box.x1 = box.x1 - 0.125 # 将右上角的x坐标向右移动0.125
        # axs[1, 2].set_position(box) # 设置新的子图位置

        # 调整子图之间的间距
        # plt.subplots_adjust(wspace=0.1, hspace=0.1)
        # plt.rcParams['legend.title_fontsize'] = 20
        # plt.legend(handles=[cnl[0],refl[0]], labels=['安全凸空间','参考路径'], prop=font ) # , loc='upper right' ,fontsize=80
        plt.tight_layout()
        plt.show()


def main():
    rl_plotter = rl_DataPlotter(path[0], xlim=xlim, ylim=ylim, eps_list=eps_list)
    rl1_plotter = rl_DataPlotter(path[1], xlim=xlim, ylim=ylim, eps_list=eps_list)
    rl2_plotter = rl_DataPlotter(path[2], xlim=xlim, ylim=ylim, eps_list=eps_list)
    rl3_plotter = rl_DataPlotter(path[3], xlim=xlim, ylim=ylim, eps_list=eps_list)
    rl4_plotter = rl_DataPlotter(path[4], xlim=xlim, ylim=ylim, eps_list=eps_list)
    plotters = [rl_plotter, rl1_plotter, rl2_plotter, rl3_plotter, rl4_plotter]
    rl_infos = [rl_plotter.run(), rl1_plotter.run(), rl2_plotter.run(), rl3_plotter.run(), rl4_plotter.run()]

    # rl1_plotter = rl_DataPlotter(path[1],xlim = xlim,ylim = ylim,eps_list = eps_list)
    # rl2_plotter = rl_DataPlotter(path[2],xlim = xlim,ylim = ylim,eps_list = eps_list)
    # rl3_plotter = rl_DataPlotter(path[3],xlim = xlim,ylim = ylim,eps_list = eps_list)
    # plotters = [rl1_plotter,rl2_plotter,rl3_plotter]
    # rl_infos = [rl1_plotter.run(),rl2_plotter.run(),rl3_plotter.run()]

    # rl_plotter = rl_DataPlotter(path[0],xlim = xlim,ylim = ylim,eps_list = eps_list)
    # rl4_plotter = rl_DataPlotter(path[4],xlim = xlim,ylim = ylim,eps_list = eps_list)
    # plotters = [rl_plotter,rl4_plotter]
    # rl_infos = [rl_plotter.run(),rl4_plotter.run()]

    # rl1_plotter = rl_DataPlotter(path[1],xlim = xlim,ylim = ylim,eps_list = eps_list)
    # rl2_plotter = rl_DataPlotter(path[2],xlim = xlim,ylim = ylim,eps_list = eps_list)
    # plotters = [rl1_plotter,rl2_plotter]
    # rl_infos = [rl1_plotter.run(),rl2_plotter.run()]

    # rl3_plotter = rl_DataPlotter(path[3],xlim = xlim,ylim = ylim,eps_list = eps_list)
    # rl4_plotter = rl_DataPlotter(path[4],xlim = xlim,ylim = ylim,eps_list = eps_list)
    # plotters = [rl3_plotter,rl4_plotter]
    # rl_infos = [r3_plotter.run(),rl4_plotter.run()]

    # rl_plotter = rl_DataPlotter(path[0],xlim = xlim,ylim = ylim,eps_list = eps_list)
    # plotters = [rl_plotter]
    # rl_infos = [rl_plotter.run()]

    plot_rl(rl_infos, plotters)


if __name__ == "__main__":
    main()
