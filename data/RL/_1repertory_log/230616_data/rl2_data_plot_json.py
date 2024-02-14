import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import math
import scipy.spatial as spatial
import numpy as np
from rl2_data_analayze_json import RobotStatistics
from copy import deepcopy

# 使点集按照逆时针或顺时针顺序排列,从而方便后续绘制闭合的多边形。
def counter_lock_sort(points):
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

class rl_DataPlotter:
    def __init__(self, path = "flat_ped_sim_0502_sparse_polar2p_limit/s10d10",xlim = (-5.5, 6.5),ylim = (-5.5, 6.5),eps_list = [i for i in range(1000)],show_figures=True):
        self.show_figures = show_figures
        self.xlim = xlim
        self.ylim = ylim
        self.eps_list = eps_list
        self.min_distance = 0.3  # Set your min distance here
        self.max_distance = 3.0  # Set your max distance here
        self.min_distance_obs = 0.2  # Set your min distance here
        self.max_distance_obs = 3.0  # Set your max distance here

        self.if_e2e = False
        self.stats = RobotStatistics(path)
        self.path = "D:/RL/_1repertory_log/230616_data/"
        # Load data
        # with open("/home/dmz/flat_ped_sim/src/data_collected_plot.json") as f:
        with open(self.path+path+"_scence_plot.json") as f:
            self.data = json.load(f)
        scence = path.split("/")[-1].split("_")[0]
        with open("D:/RL/_1repertory_log/230616_data/"+scence+"_scence.json") as f:
            self.scene_data = json.load(f)
            self.scene_data = self.scene_data["scenarios"]

        self.odom_data = self.data["robot_state"]
        self.dynamic_obstacle_data = self.data["dynamic_obstacle"]
        self.ctrl_network_points_data = self.data["ctrl_network_points"]
        self.convex_polygon_data = self.data["convex_polygon"]

    def check_distance(self, point1, point2):
        distance = np.linalg.norm(np.array(point1) - np.array(point2))
        return self.min_distance <= distance <= self.max_distance
    def check_distance_obs(self, point1, point2):
        distance = np.linalg.norm(np.array(point1) - np.array(point2))
        return self.min_distance_obs <= distance <= self.max_distance_obs

    def data_process(self,data_eps, odom_data_eps, dynamic_obstacle_data_eps,ctrl_network_points_data_eps,convex_polygon_data_eps):
        odom_data_eps.sort(key=lambda x: (x[1]))
        x_traj, y_traj = [], []
        start_pos, end_pos = None, None
        index = 0
        last_point = None
        robot_traj = []
        traj_step = []
        max_step = 0

        static_obs = []
        static_obstacle_data = self.scene_data[data_eps-1]["static_obstacles"]
        for i in range(len(static_obstacle_data)):
            vertex = counter_lock_sort(static_obstacle_data["static_obs_"+str(i)]["vertices"])
            vertex.append(vertex[0])
            # static_obstacle_data["static_obs_"+str(i)]["vertices"].append(static_obstacle_data["static_obs_"+str(i)]["vertices"][0])
            static_obs.append(np.array(vertex))


        for eps, step, position, linear_velocity, linear_acceleration, start_position, goal_position in odom_data_eps[::1]:  # Sparse the data
            x, y = position

            # 和上一个点last_point距离是否在范围内check_distance,避免点太密集
            if last_point is None or self.check_distance(last_point, position):
            # if True:
                if np.isnan(x) or np.isnan(y) or step <= 1:
                    continue
                if not self.xlim[0] <= x <= self.xlim[1] or not self.ylim[0] <= y <= self.ylim[1]:
                    continue

                x_traj.append(x)
                y_traj.append(y)
                traj_step.append(step)
                robot_traj.append([x,y])
                last_point = position
                max_step = max(max_step,step)

        # if len(x_traj) > 0:
        #     print("data_eps:",data_eps,eps)
        #     print("traj 0:",(x_traj[0],y_traj[0]))
        #     print("traj -1:",(x_traj[-1],y_traj[-1]))


        # Sort dynamic obstacles by obstacle_id and step
        dynamic_obstacle_data_eps.sort(key=lambda x: (x[1], x[0]))
        # Plot dynamic obstacles
        index = 0
        current_obstacle_id = None
        obstacle_trajectory = [[]]
        last_point = None
        start = None

        for step, obstacle_id, x, y, eps in dynamic_obstacle_data_eps[::1]:  # Sparse the data
            r = self.scene_data[eps-1]["dynamic_obstacles"]["dynamic_obs_"+str(obstacle_id)]["obstacle_radius"]
            v = self.scene_data[eps-1]["dynamic_obstacles"]["dynamic_obs_"+str(obstacle_id)]["linear_velocity"]
            start = self.scene_data[eps-1]["dynamic_obstacles"]["dynamic_obs_"+str(obstacle_id)]["start_pos"]
            end = self.scene_data[eps-1]["dynamic_obstacles"]["dynamic_obs_"+str(obstacle_id)]["waypoints"][-1]

            if current_obstacle_id is None:
                current_obstacle_id = obstacle_id

            if step > max_step:
                continue

            if np.isnan(x) or np.isnan(y):
                continue
            if not self.xlim[0] <= x <= self.xlim[1] or not self.ylim[0] <= y <= self.ylim[1]:
                continue
            if current_obstacle_id != obstacle_id or last_point is None or self.check_distance_obs(last_point, (x, y)):
                if current_obstacle_id != obstacle_id:
                    if len(obstacle_trajectory[-1]) <= 1:
                        obstacle_trajectory[-1] = [(start[0],start[1],0,r)]
                        dist = math.sqrt(((start[0] - end[0])**2+(start[1] - end[1])**2))
                        theta = math.atan2(end[1] - start[1],end[0] - start[0])
                        steps = min(math.ceil(dist/v*10),30)
                        for step in range(1,steps):
                            px = obstacle_trajectory[-1][step-1][0] + v*0.1*math.cos(theta)
                            py = obstacle_trajectory[-1][step-1][1] + v*0.1*math.sin(theta)
                            obstacle_trajectory[-1].append((px, py, step, r))
                    # Plot the trajectory of the obstacle
                    current_obstacle_id = obstacle_id
                    obstacle_trajectory.append([])
                obstacle_trajectory[-1].append((x, y, step, r))
                last_point = (x, y)

        if start is not None and len(obstacle_trajectory[-1]) <= 1:
            obstacle_trajectory[-1] = [(start[0],start[1],0,r)]
            dist = math.sqrt(((start[0] - end[0])**2+(start[1] - end[1])**2))
            theta = math.atan2(end[1] - start[1],end[0] - start[0])
            steps = min(math.ceil(dist/v*10),30)
            for step in range(1,steps):
                px = obstacle_trajectory[-1][step-1][0] + v*0.1*math.cos(theta)
                py = obstacle_trajectory[-1][step-1][1] + v*0.1*math.sin(theta)
                obstacle_trajectory[-1].append((px, py, step, r))
                dist = math.sqrt(((start[0] -px)**2+(start[1] - py)**2))

        ctrl_network_points_data_eps.sort(key=lambda x: (x[0], x[1]))
        convex_polygon_data_eps.sort(key=lambda x: (x[0], x[1]))
        # Create a dictionary to group polygon points and control network points by step
        polygon_points_by_step = {}
        ctrl_network_points_by_step = {}
        steps = []
        # eps_list = [0,10,18,25,30]
        # eps_list = [6,18,30] # 18 ok 30 ok
        index_set = set([1,40,49,52]) # ,101,112
        for eps, step, polygon_points in convex_polygon_data_eps[::1]:
        # for i in eps_list:
            # eps, step, polygon_points = convex_polygon_data_eps[i]
            # if step not in index_set:
            #     continue
            if step not in polygon_points_by_step:
                polygon_points_by_step[step] = []
            steps.append(step)
            polygon_points_by_step[step].append(np.array(polygon_points))
        # print(steps)
        # print(" ")
        # polygon_points_by_step[38] = []
        # polygon_points_by_step[38].append(np.array(convex_polygon_data_eps[-4][2]))

        if len(robot_traj) != 0:
            path = np.array(robot_traj)
            kdtree = spatial.cKDTree(path)
        steps = []
        index_set = set([1,5,7,9])
        for eps, step, ctrl_network_points in ctrl_network_points_data_eps[::1]:
            # if step  in index_set:
            #     continue
            # if len(robot_traj) != 0:
            #     p = ctrl_network_points[0]
            #     dist, index = kdtree.query([p[0], p[1]])
            #     # 0.45
            #     if dist > 0.55:
            #         continue
            #     p = ctrl_network_points[1]
            #     dist, index = kdtree.query([p[0], p[1]])
            #     if dist > 0.55:
            #         continue

            if step not in ctrl_network_points_by_step:
                ctrl_network_points_by_step[step] = []
            ctrl_network_points_by_step[step].append(np.array(ctrl_network_points))
            steps.append(step)
        # print(steps)
        # p1 = deepcopy(ctrl_network_points_by_step[steps[-3]][-1][1])
        # p1[0] += 0.15
        # p1[1] += 0.4
        # p2 = deepcopy(ctrl_network_points_by_step[steps[-2]][-1][1])
        # p2[0] += 0.3
        # p2[1] += 0.5
        # p3 = deepcopy(ctrl_network_points_by_step[steps[-1]][-1][1])
        # p3[0] += -0.2
        # p3[1] += 0.6

        # ctrl_network_points_by_step[steps[-3]].append(np.array([p1,[end_pos[0],end_pos[1]]]))
        # ctrl_network_points_by_step[steps[-2]].append(np.array([p2,[end_pos[0],end_pos[1]]]))
        # ctrl_network_points_by_step[steps[-1]].append(np.array([p3,[end_pos[0],end_pos[1]]]))


        start_pos = [self.scene_data[data_eps-1]["robot"]["start_pos"][0],self.scene_data[data_eps-1]["robot"]["start_pos"][1]]
        end_pos = [self.scene_data[data_eps-1]["robot"]["goal_pos"][0],self.scene_data[data_eps-1]["robot"]["goal_pos"][1]]
        # print("start pos by scence:",start_pos)
        # print("end pos by scence:",end_pos)
        return x_traj,y_traj,max_step,obstacle_trajectory,start_pos,end_pos,polygon_points_by_step,ctrl_network_points_by_step,static_obs,traj_step

    def plot_data(self, data_eps, odom_data_eps, dynamic_obstacle_data_eps,ctrl_network_points_data_eps,convex_polygon_data_eps):
        # Create a figure and axis
        fig, ax = plt.subplots()
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect('equal') # Set 1:1 aspect ratio
        text_offset = 0.15

        # Plot robot trajectory
        odom_data_eps.sort(key=lambda x: (x[0], x[1]))
        x_traj, y_traj = [], []
        start_pos, end_pos = None, None
        index = 0
        last_point = None
        for eps, step, position, linear_velocity, linear_acceleration, start_position, goal_position in odom_data_eps[::1]:  # Sparse the data
            x, y = position
            if last_point is None or self.check_distance(last_point, position):
                if np.isnan(x) or np.isnan(y) or step == 0:
                    continue
                if not self.xlim[0] <= x <= self.xlim[1] or not self.ylim[0] <= y <= self.ylim[1]:
                    continue
                if index <= 2:
                    start_pos = start_position
                    end_pos = goal_position
                x_traj.append(x)
                y_traj.append(y)
                circle = Circle((x, y), 0.225, color='b', alpha=0.1)  # Add circle for robot radius
                ax.add_patch(circle)
                ax.plot(x, y, 'bo', markersize=3)
                ax.text(x+text_offset, y+text_offset, f"{int(step)}", fontsize=8, color='blue') 
                last_point = position
            ax.plot(x_traj, y_traj, 'b-', linewidth=1)  # Connect the trajectory points
            index += 1

        if start_pos and end_pos:
            ax.plot(start_pos[0],start_pos[1], 'gs', markersize=10)
            ax.text(start_pos[0]+text_offset,start_pos[1]+text_offset, "START", fontsize=12, color='green')
            ax.plot(end_pos[0],end_pos[1], 'rs', markersize=10)
            ax.text(end_pos[0]+text_offset,end_pos[1]+text_offset, "GOAL", fontsize=12, color='red')
            circle = Circle((end_pos[0],end_pos[1]), 1, color='r', alpha=0.1)  # Add circle for robot radius
            ax.add_patch(circle)

        # Plot dynamic obstacles
        # Sort dynamic obstacles by obstacle_id and step
        dynamic_obstacle_data_eps.sort(key=lambda x: (x[1], x[0]))
        # Plot dynamic obstacles
        index = 0
        current_obstacle_id = None
        obstacle_trajectory = []
        last_point = None
        for step, obstacle_id, x, y, eps in dynamic_obstacle_data_eps[::1]:  # Sparse the data
            if current_obstacle_id is None:
                current_obstacle_id = obstacle_id
            if np.isnan(x) or np.isnan(y):
                continue
            if not self.xlim[0] <= x <= self.xlim[1] or not self.ylim[0] <= y <= self.ylim[1]:
                continue
            if current_obstacle_id != obstacle_id or last_point is None or self.check_distance_obs(last_point, (x, y)):
                if current_obstacle_id != obstacle_id:
                    # Plot the trajectory of the obstacle
                    if len(obstacle_trajectory) > 1:
                        ax.plot([x for x, _, _ in obstacle_trajectory], [y for _, y, _ in obstacle_trajectory], 'r-', linewidth=1)
                    current_obstacle_id = obstacle_id
                    obstacle_trajectory = []
                ax.plot(x, y, 'ro', markersize=2)
                ax.text(x+text_offset, y+text_offset, f"{int(step)}", fontsize=8, color='red')
                # ax.text(x-text_offset, y-text_offset, f"{int(current_obstacle_id)}", fontsize=8, color='black')
                obstacle_trajectory.append((x, y, step))
                circle = Circle((x, y), 0.3, color='r', alpha=0.1)
                ax.add_patch(circle)
                last_point = (x, y)
        # if len(obstacle_trajectory) > 1:
        #     ax.plot([x for x, _, _ in obstacle_trajectory], [y for _, y, _ in obstacle_trajectory], 'r-', linewidth=1)

        if not self.if_e2e:
            ctrl_network_points_data_eps.sort(key=lambda x: (x[0], x[1]))
            convex_polygon_data_eps.sort(key=lambda x: (x[0], x[1]))

            # Create a dictionary to group polygon points and control network points by step
            polygon_points_by_step = {}
            ctrl_network_points_by_step = {}
            for eps, step, polygon_points in convex_polygon_data_eps[::30]:
                if step not in polygon_points_by_step:
                    polygon_points_by_step[step] = []
                polygon_points_by_step[step].append(np.array(polygon_points))

            for eps, step, ctrl_network_points in ctrl_network_points_data_eps[::30]:
                if step not in ctrl_network_points_by_step:
                    ctrl_network_points_by_step[step] = []
                ctrl_network_points_by_step[step].append(np.array(ctrl_network_points))

            # Now we can plot the polygon points and control network points for each step
            for step in sorted(polygon_points_by_step.keys()):
                for polygon_points in polygon_points_by_step[step]:
                    ax.plot(*polygon_points.T, 'g-', linewidth=1)  # Draw the polygon
                    ax.plot(*polygon_points.T, 'go', markersize=3)  # Mark the points

            for step in sorted(ctrl_network_points_by_step.keys()):
                for ctrl_network_points in ctrl_network_points_by_step[step]:
                    ax.plot(*ctrl_network_points.T, 'm-', linewidth=1)  # Draw the points connected
                    ax.plot(*ctrl_network_points.T, 'mo', markersize=3)  # Mark the points

        # Show the plot
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Robot Trajectory and Obstacles for eps {data_eps}")
        plt.grid()

        success,time, distance, velocity = self.stats.get_eps_stats(data_eps)
        print(f'Eps {data_eps} - success: {success}, time: {time}, distance: {distance}, velocity: {velocity}')


        # # Save the figure
        # if not os.path.exists("plots"):
        #     os.makedirs("plots")
        # plt.savefig(f"plots/plot_eps_{eps}.png")
        
        # if self.show_figures:
        #     plt.show()
        
        # plt.close(fig)

        def on_key_press(event):
            if event.key == 'd':
                plots_path = self.path+"plots_30d"
                if not os.path.exists(plots_path):
                    os.makedirs(plots_path)
                plt.savefig(f"{plots_path}/plot_eps_{data_eps}.png")
                print(f'Image saved to {plots_path}/plot_eps_{data_eps}.png')
                plt.close(fig)
            else :
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key_press)

        if self.show_figures:
            plt.show()

    def run(self):
        eps_list = list(set([eps for eps, _, _, _, _, _ ,_ in self.odom_data]))
        x_trajs,y_trajs,max_steps,obstacle_trajectorys,start_positions,goal_positions,polygon_points_by_steps,ctrl_network_points_by_steps,static_obss,traj_steps = [],[],[],[],[],[],[],[],[],[]

        # for i in range(10):# 
        for i in self.eps_list:
            eps = i+1
            odom_data_eps = [entry for entry in self.odom_data if entry[0] == eps]
            dynamic_obstacle_data_eps = [entry for entry in self.dynamic_obstacle_data if entry[-1] == eps]
            ctrl_network_points_data_eps = [entry for entry in self.ctrl_network_points_data if entry[0] == eps]
            convex_polygon_data_eps = [entry for entry in self.convex_polygon_data if entry[0] == eps]
            # self.plot_data(eps, odom_data_eps, dynamic_obstacle_data_eps,ctrl_network_points_data_eps,convex_polygon_data_eps)
            x_traj,y_traj,max_step,obstacle_trajectory,start_position,goal_position,polygon_points_by_step,ctrl_network_points_by_step,static_obs,traj_step = self.data_process(eps, odom_data_eps, dynamic_obstacle_data_eps,ctrl_network_points_data_eps,convex_polygon_data_eps)
            traj_steps.append(traj_step)
            static_obss.append(static_obs)
            x_trajs.append(x_traj)
            y_trajs.append(y_traj)
            max_steps.append(max_step)
            obstacle_trajectorys.append(obstacle_trajectory)
            start_positions.append(start_position)
            goal_positions.append(goal_position)
            polygon_points_by_steps.append(polygon_points_by_step)
            ctrl_network_points_by_steps.append(ctrl_network_points_by_step)
        return x_trajs,y_trajs,max_steps,obstacle_trajectorys,start_positions,goal_positions,polygon_points_by_steps,ctrl_network_points_by_steps,static_obss,traj_steps

def main():
    plotter = DataPlotter(show_figures=True)
    plotter.run()

if __name__ == "__main__":
    main()
