import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from data_analayze_json import RobotStatistics

class DataPlotter:
    def __init__(self, show_figures=True):
        self.show_figures = show_figures
        self.xlim = (-5.5, 6.5)
        self.ylim = (-5.5, 6.5)
        self.min_distance = 0.3  # Set your min distance here
        self.max_distance = 3.0  # Set your max distance here
        self.min_distance_obs = 0.2  # Set your min distance here
        self.max_distance_obs = 3.0  # Set your max distance here

        self.if_e2e = False

        self.stats = RobotStatistics()
        self.path = "/home/dmz/flat_ped_sim/src/d86env/result_model_senario_230612/"
        
        # Load data
        # with open("/home/dmz/flat_ped_sim/src/data_collected_plot.json") as f:
        with open(self.path+"data_collected_plot_30d.json") as f:
            self.data = json.load(f)

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
        for i in range(0,len(eps_list),10):
            eps = eps_list[i]
            odom_data_eps = [entry for entry in self.odom_data if entry[0] == eps]
            dynamic_obstacle_data_eps = [entry for entry in self.dynamic_obstacle_data if entry[-1] == eps]
            ctrl_network_points_data_eps = [entry for entry in self.ctrl_network_points_data if entry[0] == eps]
            convex_polygon_data_eps = [entry for entry in self.convex_polygon_data if entry[0] == eps]
            self.plot_data(eps, odom_data_eps, dynamic_obstacle_data_eps,ctrl_network_points_data_eps,convex_polygon_data_eps)

def main():
    plotter = DataPlotter(show_figures=True)
    plotter.run()

if __name__ == "__main__":
    main()
