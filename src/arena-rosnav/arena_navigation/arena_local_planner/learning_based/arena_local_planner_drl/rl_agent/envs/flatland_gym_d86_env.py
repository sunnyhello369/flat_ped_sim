#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import imp
import os
from operator import is_
from random import randint
import gym
from gym import spaces
from gym.spaces import space
from typing import Union
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import ActionNoise
import yaml

from rl_agent.utils.observation_collector_d86 import ObservationCollector_d86
from rl_agent.utils.action_collector_d86 import ActionCollector_d86

# from rl_agent.utils.CSVWriter import CSVWriter # TODO: @Elias: uncomment when csv-writer exists
from rl_agent.utils.reward_d86 import RewardCalculator_d86
from rl_agent.utils.debug import timeit
import numpy as np
import rospy
from geometry_msgs.msg import Twist,Pose2D
from std_msgs.msg import String
from flatland_msgs.srv import StepWorld, StepWorldRequest
from std_msgs.msg import Bool
import time
from datetime import datetime as dt
import math

from rl_agent.utils.debug import timeit
from task_generator.tasks import *


class FlatlandEnv_d86(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(
        self,
        ns: str,
        reward_fnc: str,
        is_action_space_discrete,
        exp_dist: float = 0.1,
        goal_radius: float = 0.1,
        max_steps_per_episode=100,
        train_mode: bool = True,
        debug: bool = False,
        task_mode: str = "staged",
        PATHS: dict = dict(),
        extended_eval: bool = False,
        *args,
        **kwargs,
    ):
        """Default env
        Flatland yaml node check the entries in the yaml file, therefore other robot related parameters cound only be saved in an other file.
        TODO : write an uniform yaml paser node to handel with multiple yaml files.


        Args:
            task (ABSTask): [description]
            reward_fnc (str): [description]
            train_mode (bool): bool to differ between train and eval env during training
            is_action_space_discrete (bool): [description]
            safe_dist (float, optional): [description]. Defaults to None.
            goal_radius (float, optional): [description]. Defaults to 0.1.
            extended_eval (bool): more episode info provided, no reset when crashing
        """
        super(FlatlandEnv_d86, self).__init__()

        self.ns = ns #单个仿真环境
        try:
            # given every environment enough time to initialize, if we dont put sleep,
            # the training script may crash.
            ns_int = int(ns.split("_")[1])
            time.sleep(ns_int * 2)
        except Exception:
            rospy.logwarn(f"Can't not determinate the number of the environment, training script may crash!")

        # process specific namespace in ros system
        self.ns_prefix = "" if (ns == "" or ns is None) else "/" + ns + "/"
        self.step_size = rospy.get_param("step_size")
        # if not debug:
        if train_mode:
            rospy.init_node(f"train_env_{self.ns}", disable_signals=False)
        else:
            rospy.init_node(f"eval_env_{self.ns}", disable_signals=False)

        self._extended_eval = extended_eval
        self._is_train_mode = train_mode
        self._is_action_space_discrete = is_action_space_discrete
        # 机器人信息读取 两个文件 
        #  arena_local_planner_drl/configs/default_settings.yaml  训练用机器人动力学信息
        #  arena-rosnav/simulator_setup/robot/xxx.model.yaml 机器人传感器默认参数
        self.setup_by_configuration(PATHS["robot_setting"], PATHS["robot_as"])

        # csv writer # TODO: @Elias: uncomment when csv-writer exists
        # self.csv_writer=CSVWriter()
        # rospy.loginfo("======================================================")
        # rospy.loginfo("CSVWriter initialized.")
        # rospy.loginfo("======================================================")

        # reward calculator
        safe_dist = self._robot_radius +exp_dist

        self.reward_calculator = RewardCalculator_d86(
            holonomic=self._holonomic,
            robot_radius=self._robot_radius,
            safe_dist=safe_dist,
            goal_radius=goal_radius,
            rule=reward_fnc,
            extended_eval=self._extended_eval,
        )


        # service clients
        # if self._is_train_mode:
        #     self._service_name_step = f"{self.ns_prefix}step_world"
        #     self._sim_step_client = rospy.ServiceProxy(self._service_name_step, StepWorld)

        # instantiate task manager
        self.task = get_predefined_task(ns, mode=task_mode, start_stage=kwargs["curr_stage"], PATHS=PATHS)
        self._steps_curr_episode = 0
        self._episode = 0
        self._max_steps_per_episode = max_steps_per_episode
        self._last_action = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        self.last_convex_plan_result = None
        # observation collector
        self.observation_collector = ObservationCollector_d86(
            self.ns,
            self._laser_num_beams,
            self._laser_max_range,
            safe_dist = safe_dist,
            exp_dist = exp_dist,
            laser_origin_in_base = self.laser_in_base,
            plan_dynamic_limit = self.plan_dynamic_limit,
            external_time_sync=False,
        )
        self.observation_space = self.observation_collector.get_observation_space()
        self.last_obs_dict = None
        
        # planner
        self.action_collector = ActionCollector_d86(ns=self.ns,
        plan_dynamic_limit = self.plan_dynamic_limit,
        lidar_num = self._laser_num_beams,
        output_points_num = 5)
        self.action_space = self.action_collector.get_action_space()

        # print("obs space:")
        # print(self.observation_space)
        # print("action space:")
        # print(self.action_space)
        
        # for extended eval
        self._action_frequency = 1. / rospy.get_param("/robot_action_rate")
        self._last_robot_pose = None
        self._distance_travelled = 0
        self._safe_dist_counter = 0
        self._collisions = 0
        self._in_crash = False

        self._done_reasons = {
            "0": "Exc. Max Steps",
            "1": "Crash",
            "2": "Goal Reached",
        }
        self._done_hist = 3 * [0]

    # 读入/home/tou/desire_10086/flat_ped_sim/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/configs
    # 中的配置参数
    def setup_by_configuration(self, robot_yaml_path: str, settings_yaml_path: str):
        """get the configuration from the yaml file, including robot radius, discrete action space and continuous action space.

        Args:
            robot_yaml_path (str): [description]
        """
        self._robot_radius = rospy.get_param("radius")
        with open(robot_yaml_path, "r") as fd:
            robot_data = yaml.safe_load(fd)

            # get laser related information
            for plugin in robot_data["plugins"]:
                if plugin["type"] == "Laser" and plugin["name"] == "static_laser":
                    laser_angle_min = plugin["angle"]["min"]
                    laser_angle_max = plugin["angle"]["max"]
                    laser_angle_increment = plugin["angle"]["increment"]
                    self._laser_num_beams = int(round((laser_angle_max - laser_angle_min) / laser_angle_increment))
                    self._laser_max_range = plugin["range"]
                    self.laser_in_base = plugin["origin"] # x,y,yaw in basefootprint


        with open(settings_yaml_path, "r") as fd:
            setting_data = yaml.safe_load(fd)

            self._holonomic = setting_data["robot"]["holonomic"]

            self.plan_dynamic_limit = (setting_data["robot"]["dynamic_limits"]["v"][1],setting_data["robot"]["dynamic_limits"]["a"][1],setting_data["robot"]["dynamic_limits"]["j"][1])

                     
    # step周期为规划周期定为10hz  控制周期定为20hz
    def step(self, action: np.ndarray):
        """
        done_reasons:   0   -   exceeded max steps
                        1   -   collision with obstacle
                        2   -   goal reached
        """
        # action[5*2]  which laser1,which laser2,...,dist1,dist2,...
        # 规划
        # 由laser_convex得到action对应的x,y点 action经过了tanh
        action_points = [] # base下
        drad = 1./self._laser_num_beams*2*np.pi
        
        # scan_xy第一个点对应角度
        # first_polar_convex_theta = np.arctan2(self.last_obs_dict["laser_convex"][0][1],self.last_obs_dict["laser_convex"][0][0])
        if self.last_obs_dict["laser_convex"][0] is not None:
            for i in range(self.action_collector.output_points_num):
                index = int(self._laser_num_beams/2.*(1. + action[i])) % self._laser_num_beams
                # if index == self._laser_num_beams:
                #     index = 0
                dist = (1.+ action[self.action_collector.output_points_num+i])/2.*self.last_obs_dict["laser_convex"][1][index]
                theta = self.last_obs_dict["laser_convex"][2][index]
                # print(dist,action[self.action_collector.output_points_num+i],self.last_obs_dict["laser_convex"][1][int(action[i])])
                action_points.append((dist*np.cos(theta),dist*np.sin(theta)))
        # 控制轨迹执行 控制周期50ms 直到下个规划周期
        convex_plan_result = self.action_collector.plan_step(action_points,self.last_obs_dict["laser_convex"][0],self.last_obs_dict["robot_world_pose"])       
        
        # print(f"Linear: {action[0]}, Angular: {action[1]}")
        self._steps_curr_episode += 1

        # wait for new observations
        # merged_obs已经被normalize过
        
        if self.last_convex_plan_result is not None and self.last_convex_plan_result[0]:
            close_decision_action = [p for p in zip(self.last_convex_plan_result[2][0],self.last_convex_plan_result[2][1])]
        else:
            close_decision_action = None
        merged_obs, obs_dict = self.observation_collector.get_observations(last_action_point_in_robot=close_decision_action)

        # calculate reward

        reward, reward_info = self.reward_calculator.get_reward(
            obs_dict["laser_scan"],
            obs_dict["goal_in_robot_frame"],
            obs_dict["goal_in_robot_frame_ltrajp"],
            action = self._last_action,
            global_plan = obs_dict["global_plan"],
            robot_pose = obs_dict["robot_world_pose"],
            robot_vel = obs_dict["robot_world_vel"],
            plan_result = self.last_convex_plan_result,
        )
        self._last_action = action_points
        self.last_obs_dict  = obs_dict
        self.last_convex_plan_result = convex_plan_result
        # print(f"cum_reward: {reward}")
        done = reward_info["is_done"]

        # extended eval info
        if self._extended_eval:
            self._update_eval_statistics(obs_dict, reward_info)

        # info
        info = {}

        if done:
            info["done_reason"] = reward_info["done_reason"]
            info["is_success"] = reward_info["is_success"]

        if self._steps_curr_episode > self._max_steps_per_episode:
            done = True
            info["done_reason"] = 0
            info["is_success"] = 0

        # for logging
        if self._extended_eval and done:
            info["collisions"] = self._collisions
            info["distance_travelled"] = round(self._distance_travelled, 2)
            info["time_safe_dist"] = self._safe_dist_counter * self._action_frequency
            info["time"] = self._steps_curr_episode * self._action_frequency

        if done:
            if sum(self._done_hist) == 1000 and self.ns_prefix != "/eval_sim/":
                print(
                    f"[ns: {self.ns_prefix}] Last 1000 Episodes: "
                    f"{self._done_hist[0]}x - {self._done_reasons[str(0)]}, "
                    f"{self._done_hist[1]}x - {self._done_reasons[str(1)]}, "
                    f"{self._done_hist[2]}x - {self._done_reasons[str(2)]}, "
                )
                self._done_hist = [0] * 3
            self._done_hist[int(info["done_reason"])] += 1
        
        # obs_{i+1},r_i,done_i,info_i
        return merged_obs, reward, done, info

    def reset(self):
        # set task
        # regenerate start position end goal position of the robot and change the obstacles accordingly
        self._episode += 1

        # self.action_collector.agent_action_pub.publish(Twist())
        # if self._is_train_mode:
        #     self._sim_step_client()
        self.observation_collector._globalplan = None
        # print(self.ns + "re1")
        self.action_collector.call_msg_takeSimStep(Twist(),self.step_size) #  + 0.00001
        # print(self.ns + "re2")
        # else:
        #     try:
        #         rospy.wait_for_message(f"{self.ns_prefix}next_cycle", Bool)
        #     except Exception:
        #         pass

        while True:
            start_pos,goal_pos = self.task.reset()
            # print(self.ns + "re3")
            self.observation_collector.reset(start_pos,goal_pos)

            # print(self.ns + "re7")

            obs, self.last_obs_dict  = self.observation_collector.get_observations(None)
            # print(self.ns + "re8")
            if self.last_obs_dict["laser_scan"].min() > 0.5:
                break

        # print(self.ns + "re4")
        self.action_collector.reset(start_pos,goal_pos)
        # print(self.ns + "re5")
        self.reward_calculator.reset()
        # print(self.ns + "re6")
        self._steps_curr_episode = 0
        self._last_action = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        self.last_convex_plan_result = None
        # extended eval info
        if self._extended_eval:
            self._last_robot_pose = None
            self._distance_travelled = 0
            self._safe_dist_counter = 0
            self._collisions = 0

        return obs  # reward, done, info can't be included



    def close(self):
        pass

    def _update_eval_statistics(self, obs_dict: dict, reward_info: dict):
        """
        Updates the metrics for extended eval mode

        param obs_dict (dict): observation dictionary from ObservationCollector.get_observations(),
            necessary entries: 'robot_pose'
        param reward_info (dict): dictionary containing information returned from RewardCalculator.get_reward(),
            necessary entries: 'crash', 'safe_dist'
        """
        # distance travelled
        if self._last_robot_pose is not None:
            self._distance_travelled += FlatlandEnv_d86.get_distance(self._last_robot_pose, obs_dict["robot_world_pose"])

        # collision detector
        if "crash" in reward_info:
            if reward_info["crash"] and not self._in_crash:
                self._collisions += 1
                # when crash occures, robot strikes obst for a few consecutive timesteps
                # we want to count it as only one collision
                self._in_crash = True
        else:
            self._in_crash = False

        # safe dist detector
        if "safe_dist" in reward_info and reward_info["safe_dist"]:
            self._safe_dist_counter += 1

        self._last_robot_pose = obs_dict["robot_world_pose"]

    @staticmethod
    def get_distance(pose_1: Pose2D, pose_2: Pose2D):
        return math.hypot(pose_2.x - pose_1.x, pose_2.y - pose_1.y)



class FlatlandEnv_d86_e2e(FlatlandEnv_d86):
    """Custom Environment that follows gym interface"""

    def __init__(
        self,
        ns: str,
        reward_fnc: str,
        is_action_space_discrete,
        exp_dist: float = 0.1,
        goal_radius: float = 0.1,
        max_steps_per_episode=100,
        train_mode: bool = True,
        debug: bool = False,
        task_mode: str = "staged",
        PATHS: dict = dict(),
        extended_eval: bool = False,
        *args,
        **kwargs,
    ):
        """Default env
        Flatland yaml node check the entries in the yaml file, therefore other robot related parameters cound only be saved in an other file.
        TODO : write an uniform yaml paser node to handel with multiple yaml files.


        Args:
            task (ABSTask): [description]
            reward_fnc (str): [description]
            train_mode (bool): bool to differ between train and eval env during training
            is_action_space_discrete (bool): [description]
            safe_dist (float, optional): [description]. Defaults to None.
            goal_radius (float, optional): [description]. Defaults to 0.1.
            extended_eval (bool): more episode info provided, no reset when crashing
        """
        super(FlatlandEnv_d86_e2e, self).__init__(
            ns,
            reward_fnc,
            is_action_space_discrete,
            exp_dist,
            goal_radius,
            max_steps_per_episode,
            train_mode,
            debug,
            task_mode,
            PATHS,
            extended_eval,
            *args,
            **kwargs,
        )

        action_space = []

        action_space.extend([
            spaces.Box(
                    low=-1.,
                    high=1.,
                    shape=(2,),
                    dtype=np.float32,
                )
        ])
        low = []
        high = []
        for space in action_space:
            low.extend(space.low.tolist())
            high.extend(space.high.tolist())

        self.action_space = spaces.Box(np.array(low).flatten(), np.array(high).flatten())

    # step周期为规划周期定为10hz  控制周期定为20hz
    def step(self, action: np.ndarray):
        # print("*****************************************************")
        # print(self.ns)
        action_msg = Twist()
        # ax,ay = action[0]*self.plan_dynamic_limit[1],action[1]*self.plan_dynamic_limit[1]
        ax,ay = action[0]*self.plan_dynamic_limit[0],action[1]*self.plan_dynamic_limit[0]
        vx,vy = self.last_obs_dict["robot_world_vel"].x,self.last_obs_dict["robot_world_vel"].y
        # if abs(vx) > self.plan_dynamic_limit[0] or abs(vy) > self.plan_dynamic_limit[0]:
        #     print(vx,vy)
        # sim_after_vx = vx + ax* self.action_collector._action_period
        # sim_after_vy = vy + ay* self.action_collector._action_period
        
        # if abs(sim_after_vx) >= self.plan_dynamic_limit[0]:
        #     if sim_after_vx > 0 :# and ax > 0
        #         ax = (self.plan_dynamic_limit[0] - vx) / self.action_collector._action_period
        #     if sim_after_vx < 0 :# and ax < 0
        #         ax = (-self.plan_dynamic_limit[0] - vx) / self.action_collector._action_period
        
        # if abs(sim_after_vy) >= self.plan_dynamic_limit[0]:
        #     if sim_after_vy > 0:
        #         ay = (self.plan_dynamic_limit[0] - vy) / self.action_collector._action_period
        #     if sim_after_vy < 0:
        #         ay = (-self.plan_dynamic_limit[0] - vy) / self.action_collector._action_period
        # if abs(ax) > self.plan_dynamic_limit[1]:
        #     if ax > 0:
        #         ax = self.plan_dynamic_limit[1]
        #     else:
        #         ax = -self.plan_dynamic_limit[1]
        # if abs(ay) > self.plan_dynamic_limit[1]:
        #     if ay > 0:
        #         ay = self.plan_dynamic_limit[1]
        #     else:
        #         ay = -self.plan_dynamic_limit[1]

        action_msg.linear.x = ax
        action_msg.linear.y = ay
        # print(self.ns + "step 0")
        self.action_collector.call_msg_takeSimStep(action_msg,self.action_collector._action_period)
        # print(self.ns + "step 1")
        # print(f"Linear: {action[0]}, Angular: {action[1]}")
        self._steps_curr_episode += 1
        # wait for new observations
        # merged_obs已经被normalize过
        # print(self.ns + "step 2")

        merged_obs, obs_dict = self.observation_collector.get_observations(last_action=self._last_action)
        # print(self.ns + "step 3")
        self._last_action = action_msg
        self.last_obs_dict  = obs_dict

        # calculate reward
        reward, reward_info = self.reward_calculator.get_reward(
            obs_dict["laser_scan"],
            obs_dict["goal_in_robot_frame"],
            action = (ax,ay),
            global_plan = obs_dict["global_plan"],
            robot_pose = obs_dict["robot_world_pose"],
            robot_vel = obs_dict["robot_world_vel"],
            plan_result = None,
        )
        # print(f"cum_reward: {reward}")
        done = reward_info["is_done"]

        # extended eval info
        if self._extended_eval:
            self._update_eval_statistics(obs_dict, reward_info)

        # info
        info = {}

        if done:
            info["done_reason"] = reward_info["done_reason"]
            info["is_success"] = reward_info["is_success"]

        if self._steps_curr_episode > self._max_steps_per_episode:
            done = True
            info["done_reason"] = 0
            info["is_success"] = 0

        # for logging
        if self._extended_eval and done:
            info["collisions"] = self._collisions
            info["distance_travelled"] = round(self._distance_travelled, 2)
            info["time_safe_dist"] = self._safe_dist_counter * self._action_frequency
            info["time"] = self._steps_curr_episode * self._action_frequency

        if done:
            if sum(self._done_hist) == 1000 and self.ns_prefix != "/eval_sim/":
                print(dt.now().strftime("%Y_%m_%d__%H_%M_%S"))
                print(
                    f"[ns: {self.ns_prefix}] Last 1000 Episodes: "
                    f"{self._done_hist[0]}x - {self._done_reasons[str(0)]}, "
                    f"{self._done_hist[1]}x - {self._done_reasons[str(1)]}, "
                    f"{self._done_hist[2]}x - {self._done_reasons[str(2)]}, "
                )
                self._done_hist = [0] * 3
            self._done_hist[int(info["done_reason"])] += 1
        # obs_{i+1},r_i,done_i,info_i
        return merged_obs, reward, done, info
   
    def reset(self):
        # set task
        # regenerate start position end goal position of the robot and change the obstacles accordingly
        self._episode += 1

        # self.action_collector.agent_action_pub.publish(Twist())
        # if self._is_train_mode:
        #     self._sim_step_client()
        self.observation_collector._globalplan = None
        # print(self.ns + "re1")
        self.action_collector.call_msg_takeSimStep(Twist(),self.step_size) #  + 0.00001
        # print(self.ns + "re2")
        # else:
        #     try:
        #         rospy.wait_for_message(f"{self.ns_prefix}next_cycle", Bool)
        #     except Exception:
        #         pass

        while True:
            start_pos,goal_pos = self.task.reset()
            # print(self.ns + "re3")
            self.observation_collector.reset(start_pos,goal_pos)

            # print(self.ns + "re7")

            obs, self.last_obs_dict  = self.observation_collector.get_observations()
            # print(self.ns + "re8")
            if self.last_obs_dict["laser_scan"].min() > 0.5:
                break

        # print(self.ns + "re4")
        self.action_collector.reset(start_pos,goal_pos)
        # print(self.ns + "re5")
        self.reward_calculator.reset()
        # print(self.ns + "re6")
        self._steps_curr_episode = 0
        self._last_action = Twist()
        # extended eval info
        if self._extended_eval:
            self._last_robot_pose = None
            self._distance_travelled = 0
            self._safe_dist_counter = 0
            self._collisions = 0

        return obs  # reward, done, info can't be included

if __name__ == "__main__":

    rospy.init_node("flatland_gym_env", anonymous=True, disable_signals=False)
    print("start")

    flatland_env = FlatlandEnv_d86()
    rospy.loginfo("======================================================")
    rospy.loginfo("CSVWriter initialized.")
    rospy.loginfo("======================================================")
    check_env(flatland_env, warn=True)

    # init env
    obs = flatland_env.reset()

    # run model
    n_steps = 200
    for _ in range(n_steps):
        # action, _states = model.predict(obs)
        action = flatland_env.action_space.sample()

        obs, rewards, done, info = flatland_env.step(action)

        time.sleep(0.1)
