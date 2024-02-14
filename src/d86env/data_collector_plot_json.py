#!/usr/bin/env python
import rospy
import numpy as np
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PolygonStamped, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3

import json

import json
import os
import sys
def get_parent_directory(target_string):
    current_file = os.path.abspath(__file__)
    path_parts = current_file.split(os.path.sep)

    if target_string in path_parts:
        target_index = path_parts.index(target_string)
        target_path = os.path.sep.join(path_parts[:target_index+1]) + os.path.sep
        return target_path
    else:
        return None
d86path_json_path = get_parent_directory("flat_ped_sim") + "src/data_collected_plot.json"

class DataCollector:
    def __init__(self):
        self.num_dynamic_obstacle = 30
        # src/flatland/arena_plugins/src/tween2.cpp
        # src/flatland/arena_plugins/include/arena_plugins/tween2.h
        self.if_scenario = False

        self.odom_data = []
        self.mpc_path_data = []
        self.mpc_path_v_data = []
        self.mpc_path_a_data = []
        self.mpc_path_j_data = []
        self.ctrl_network_points_data = []
        self.convex_polygon_data = []
        self.step_reward_data = []
        self.current_step = None
        self.last_step = None

        self.dynamic_obstacle_data = {}
        # self.dynamic_obstacle_data_individual = {}
        
        self.last_step_odom = 500
        self.last_step_mpc_p = 500
        self.last_step_mpc_v = 500
        self.last_step_mpc_a = 500
        self.last_step_mpc_j = 500
        self.last_step_ctrl = 500
        self.last_step_convex = 500
        self.last_step_step = 500

        self.eps = 0
        self.odom_eps = 0
        self.mpc_p_eps = 0
        self.mpc_v_eps = 0
        self.mpc_a_eps = 0
        self.mpc_j_eps = 0
        self.ctrl_eps = 0
        self.convex_eps = 0
        self.step_eps = 0

        self.dynamic_obstacle_scenario_subscribers = []
        self.dynamic_obstacle_random_subscribers = []

        if self.if_scenario:
            # scenario
            for i in range(self.num_dynamic_obstacle):
                topic = "/sim_1/flatland_server/debug/model/obstacle_dynamic_with_traj_{:02d}".format(i)
                callback = self.create_dynamic_obstacle_scenario_callback(i)
                self.dynamic_obstacle_scenario_subscribers.append(rospy.Subscriber(topic, MarkerArray, callback))
        else:
            # random
            for i in range(self.num_dynamic_obstacle):
                topic = "/sim_1/object_with_traj_dynamic_obs_{:d}".format(i)
                callback = self.create_dynamic_obstacle_random_callback(i)
                self.dynamic_obstacle_random_subscribers.append(rospy.Subscriber(topic, Odometry, callback))

    def create_dynamic_obstacle_scenario_callback(self, obstacle_id):
        def callback(msg):
            if self.current_step is not None:
                x = msg.markers[0].pose.position.x
                y = msg.markers[0].pose.position.y
                key = (self.current_step, obstacle_id, self.eps)
                if key not in self.dynamic_obstacle_data:
                    self.dynamic_obstacle_data[key] = (x, y)
        return callback

    def create_dynamic_obstacle_random_callback(self, obstacle_id):
        def callback(msg):
            if self.current_step is not None:
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                key = (self.current_step, obstacle_id, self.eps)
                if key not in self.dynamic_obstacle_data:
                    self.dynamic_obstacle_data[key] = (x, y)
        return callback


    def odom_callback(self, msg):
        position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        linear_velocity = msg.twist.twist.linear
        linear_acceleration = msg.twist.twist.angular
        time = msg.header.stamp
        start_x = msg.pose.covariance[0]
        start_y = msg.pose.covariance[1]
        goal_x = msg.pose.covariance[2]
        goal_y = msg.pose.covariance[3]

        step = int(msg.pose.pose.position.z)
        if step < self.last_step_odom or self.odom_eps < self.eps:
            self.odom_eps += 1
            if self.odom_eps > self.eps:
                self.eps = self.odom_eps
            # self.odom_data.append([])
        self.last_step_odom = step
        self.odom_data.append((self.eps,step, position, linear_velocity, linear_acceleration,(start_x,start_y),(goal_x,goal_y)))

    def step_reward_callback(self, msg):
        step = int(msg.point.x)
        point = msg.point
        reward = point.y
        success = int(point.z)

        if step < self.last_step_step or self.step_eps < self.eps:
            self.step_reward_data.append([])
            self.step_eps += 1
            if self.step_eps > self.eps:
                self.eps = self.step_eps
        self.last_step_step = step

        self.step_reward_data[-1].append((self.eps,step, reward,success))
        self.current_step = step

    def ctrl_network_points_callback(self, msg):
        if self.current_step is not None:
            step = int(msg.id)
            points = [(p.x, p.y) for p in msg.points]

            if step < self.last_step_ctrl or self.ctrl_eps < self.eps:
                # self.ctrl_network_points_data.append([])
                self.ctrl_eps += 1
                if self.ctrl_eps > self.eps:
                    self.eps = self.ctrl_eps

            self.last_step_ctrl = step

            self.ctrl_network_points_data.append((self.eps,self.current_step,points))

    def convex_polygon_callback(self, msg):
        if self.current_step is not None:
            step = int(msg.polygon.points[0].z)
            polygon_points = [(p.x, p.y) for p in msg.polygon.points]

            if step < self.last_step_convex or self.convex_eps < self.eps:
                # self.convex_polygon_data.append([])
                self.convex_eps += 1
                if self.convex_eps > self.eps:
                    self.eps = self.convex_eps

            self.last_step_convex = step
            self.convex_polygon_data.append((self.eps,self.current_step,polygon_points))


    def save_data(self):
        print("Odom data entries:", len(self.odom_data))
        print("MPC path data entries:", len(self.mpc_path_data))
        print("MPC path v data entries:", len(self.mpc_path_v_data))
        print("MPC path a data entries:", len(self.mpc_path_a_data))
        print("MPC path j data entries:", len(self.mpc_path_j_data))
        print("Control network points data entries:", len(self.ctrl_network_points_data))
        print("Convex polygon data entries:", len(self.convex_polygon_data))
        print("Dynamic reward data entries:", len(self.dynamic_obstacle_data))
        print("Step reward data entries:", len(self.step_reward_data))

        data = {
            "robot_state": self.odom_data,
            # "mpc_path": self.mpc_path_data,
            "ctrl_network_points": self.ctrl_network_points_data,
            "convex_polygon": self.convex_polygon_data,
            "step_reward": self.step_reward_data,
            "dynamic_obstacle": [(k[0], k[1], v[0], v[1], k[2]) for k, v in self.dynamic_obstacle_data.items()]
        }
        def default_serializer(o):
            if isinstance(o, Vector3):
                # replace with actual attributes of Vector3
                return {'x': o.x, 'y': o.y, 'z': o.z}
            elif hasattr(o, '__dict__'):
                return o.__dict__
            else:
                raise TypeError(f"Object of type {type(o)} is not JSON serializable")
        
        with open(d86path_json_path, 'w') as f:
            json.dump(data, f, default=default_serializer, sort_keys=True, indent=4)


def main():
    rospy.init_node("data_collector")
    collector = DataCollector()

    rospy.Subscriber("/sim_1/action_odom", Odometry, collector.odom_callback)
    # rospy.Subscriber("/sim_1/mpc_path", Path, collector.mpc_path_callback)
    # rospy.Subscriber("/sim_1/mpc_path_v", Path, collector.mpc_path_v_callback)
    # rospy.Subscriber("/sim_1/mpc_path_a", Path, collector.mpc_path_a_callback)
    # rospy.Subscriber("/sim_1/mpc_path_j", Path, collector.mpc_path_j_callback)
    rospy.Subscriber("/sim_1/ctrl_network_points", Marker, collector.ctrl_network_points_callback)
    rospy.Subscriber("/sim_1/convex_polygon", PolygonStamped, collector.convex_polygon_callback)
    rospy.Subscriber("/sim_1/step_reward", PointStamped, collector.step_reward_callback)
    timer = rospy.Timer(rospy.Duration(0.1), lambda _: None)  # Dummy timer to keep rospy.spin() from blocking

    rospy.spin()

    collector.save_data()
    print("Data saved to "+d86path_json_path)

if __name__ == "__main__":
    main()