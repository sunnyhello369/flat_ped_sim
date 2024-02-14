#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# reward_d86s.py
import numpy as np
import scipy.spatial

from numpy.lib.utils import safe_eval
from geometry_msgs.msg import Pose2D
from typing import Tuple
from copy import deepcopy
def NormalizeAngle(d_theta):# -pi-pi
    d_theta_normalize = d_theta
    while d_theta_normalize > np.pi:
        d_theta_normalize = d_theta_normalize - 2 * np.pi
    while d_theta_normalize < -np.pi:
        d_theta_normalize = d_theta_normalize + 2 * np.pi
    return d_theta_normalize


def NormalizeAngleTo2Pi(d_theta):# 0-2pi
    d_theta_normalize = d_theta
    while d_theta_normalize > 2 * np.pi:
        d_theta_normalize = d_theta_normalize - 2 * np.pi
    while d_theta_normalize < 0:
        d_theta_normalize = d_theta_normalize + 2 * np.pi
    return d_theta_normalize


VIS = True
import time
class RewardCalculator_d86:
    def __init__(
        self,
        holonomic: float,
        robot_radius: float,
        safe_dist: float,
        goal_radius: float,
        rule: str = "rule_00",
        extended_eval: bool = False,
    ):
        """
        A class for calculating reward based various rules.


        :param safe_dist (float): The minimum distance to obstacles or wall that robot is in safe status.
                                  if the robot get too close to them it will be punished. Unit[ m ]
        :param goal_radius (float): The minimum distance to goal that goal position is considered to be reached.
        """
        self.curr_reward = 0
        # additional info will be stored here and be returned alonge with reward.
        self.info = {}
        self.holonomic = holonomic
        self.robot_radius = robot_radius
        self.goal_radius = goal_radius
        self.last_goal_dist = None
        self.last_dist_to_path = None
        self.last_action = None
        self.last_proj_s = None
        self._curr_dist_to_path = None
        self.safe_dist = safe_dist
        self._extended_eval = extended_eval
        self._curr_action = None

        self.total = 0.
        self.total_smooth = 0.
        self.total_max_speed = 0.
        self.total_change = 0.
        self.total_track = 0.
        self.total_approach = 0.
        self.total_theta_approach = 0.
        self.total_time = 0.
        self.total_safe_penalty = 0.
        
        self.kdtree = None

        self._cal_funcs = {
            "rule_00": RewardCalculator_d86._cal_reward_rule_00,
            "rule_01": RewardCalculator_d86._cal_reward_rule_01,
            "rule_02": RewardCalculator_d86._cal_reward_rule_02,
            "rule_03": RewardCalculator_d86._cal_reward_rule_03,
            "rule_04": RewardCalculator_d86._cal_reward_rule_04,
            "rule_05": RewardCalculator_d86._cal_reward_rule_05,
           "rule_d86": RewardCalculator_d86. _cal_reward_rule_d86,
            # "barn": RewardCalculator_d86._cal_reward_rule_barn,
        }
        self.cal_func = self._cal_funcs[rule]

    def reset(self):
        """
        reset variables related to the episode
        """
        if VIS:
            for i in range(50):
                print("this eps total reward:"+str(self.total))
                print("this eps total smmoth reward:"+str(self.total_smooth))
                print("this eps total max speed reward:"+str(self.total_max_speed))
                print("this eps total approach reward:"+str(self.total_approach))
                print("this eps total theta approach reward:"+str(self.total_theta_approach))
                print("this eps total time reward:"+str(self.total_time))
                print("this eps total safe reward:"+str(self.total_safe_penalty))
                print("this eps total change reward:"+str(self.total_change))
                print("this eps total track reward:"+str(self.total_track))

        self.total = 0.
        self.total_smooth = 0.
        self.total_max_speed = 0.
        self.total_approach = 0.
        self.total_theta_approach = 0.
        self.total_change = 0.
        self.total_track = 0.
        self.total_time = 0.
        self.total_safe_penalty = 0.

        self._curr_action = None
        self.last_pose = None
        self.last_vel = None
        self.last_goal_dist_ltrajp = None
        self.last_goal_dist = None
        self.last_proj_s = None
        self.last_dist_to_path = None
        self.last_action = None
        self.kdtree = None
        self._curr_dist_to_path = None

    def _reset(self):
        """
        reset variables related to current step
        """
        self.curr_reward = 0
        self.info = {}

    def get_reward(self, laser_scan: np.ndarray, goal_in_robot_frame: Tuple[float, float], *args, **kwargs):
        """
        Returns reward and info to the gym environment.

        :param laser_scan (np.ndarray): laser scan data
        :param goal_in_robot_frame (Tuple[float,float]): position (rho, theta) of the goal in robot frame (Polar coordinate)
        """
        self._reset()
        self.cal_func(self, laser_scan, goal_in_robot_frame, *args, **kwargs)
        return self.curr_reward, self.info
    # 1 计算网络输出控制点的smooth程度,计算a累加
    # 2 计算网络输出的最后一个在global_path的s，要比上次s更接近
    # 3 固定step惩罚
    # 4 collision惩罚
    # 5 scan中最近距离的指数惩罚
    # 6 网络输出终态点的速度方向，与global path投影点切线方向的角度差

    def _cal_reward_rule_00(self, laser_scan: np.ndarray, goal_in_robot_frame: Tuple[float, float], *args, **kwargs):
        self._reward_goal_reached(goal_in_robot_frame)
        self._reward_safe_dist(laser_scan, punishment=0.25)
        self._reward_collision(laser_scan)
        self._reward_goal_approached(goal_in_robot_frame, reward_factor=0.3, penalty_factor=0.4)

    def _cal_reward_rule_01(self, laser_scan: np.ndarray, goal_in_robot_frame: Tuple[float, float], *args, **kwargs):
        self._reward_distance_traveled(kwargs["action"], consumption_factor=0.0075)
        self._reward_goal_reached(goal_in_robot_frame, reward=15)
        self._reward_safe_dist(laser_scan, punishment=0.25)
        self._reward_collision(laser_scan, punishment=10)
        self._reward_goal_approached(goal_in_robot_frame, reward_factor=0.3, penalty_factor=0.4)

    def _cal_reward_rule_02(self, laser_scan: np.ndarray, goal_in_robot_frame: Tuple[float, float], *args, **kwargs):
        self._set_current_dist_to_globalplan(kwargs["global_plan"], kwargs["robot_pose"])
        self._reward_distance_traveled(kwargs["action"], consumption_factor=0.0075)
        self._reward_following_global_plan(reward_factor=0.2, penalty_factor=0.3)
        self._reward_goal_reached(goal_in_robot_frame, reward=15)
        self._reward_safe_dist(laser_scan, punishment=0.25)
        self._reward_collision(laser_scan, punishment=10)
        self._reward_goal_approached(goal_in_robot_frame, reward_factor=0.3, penalty_factor=0.4)

    def _cal_reward_rule_03(self, laser_scan: np.ndarray, goal_in_robot_frame: Tuple[float, float], *args, **kwargs):
        self._set_current_dist_to_globalplan(kwargs["global_plan"], kwargs["robot_pose"])
        self._reward_following_global_plan(kwargs["action"])
        if laser_scan.min() > self.safe_dist:
            self._reward_distance_global_plan(
                reward_factor=0.2,
                penalty_factor=0.3,
            )
        else:
            self.last_dist_to_path = None
        self._reward_goal_reached(goal_in_robot_frame, reward=15)
        self._reward_safe_dist(laser_scan, punishment=0.25)
        self._reward_collision(laser_scan, punishment=10)
        self._reward_goal_approached(goal_in_robot_frame, reward_factor=0.3, penalty_factor=0.4)

    def _cal_reward_rule_04(self, laser_scan: np.ndarray, goal_in_robot_frame: Tuple[float, float], *args, **kwargs):
        self._set_current_dist_to_globalplan(kwargs["global_plan"], kwargs["robot_pose"])
        self._reward_following_global_plan(kwargs["action"])
        if laser_scan.min() > self.safe_dist + 0.35:
            self._reward_distance_global_plan(
                reward_factor=0.2,
                penalty_factor=0.3,
            )
            self._reward_abrupt_direction_change(kwargs["action"])
            self._reward_reverse_drive(kwargs["action"])
        else:
            self.last_dist_to_path = None
        self._reward_goal_reached(goal_in_robot_frame, reward=15)
        self._reward_safe_dist(laser_scan, punishment=0.25)
        self._reward_collision(laser_scan, punishment=10)
        self._reward_goal_approached(goal_in_robot_frame, reward_factor=0.3, penalty_factor=0.4)

    def _cal_reward_rule_05(self, laser_scan: np.ndarray, goal_in_robot_frame: Tuple[float, float], *args, **kwargs):
        self._curr_action = kwargs["action"]
        self._set_current_dist_to_globalplan(kwargs["global_plan"], kwargs["robot_pose"])
        # self._reward_following_global_plan(self._curr_action)
        if laser_scan.min() > self.safe_dist:
            self._reward_distance_global_plan(
                reward_factor=0.2,
                penalty_factor=0.3,
            )
            self._reward_abrupt_vel_change(vel_idx=0, factor=1.0)
            self._reward_abrupt_vel_change(vel_idx=-1, factor=0.5)
            if self.holonomic:
                self._reward_abrupt_vel_change(vel_idx=1, factor=0.5)
            self._reward_reverse_drive(self._curr_action, 0.0001)
        else:
            self.last_dist_to_path = None
        self._reward_goal_reached(goal_in_robot_frame, reward=17.5)
        self._reward_safe_dist(laser_scan, punishment=0.25)
        self._reward_collision(laser_scan, punishment=10)
        self._reward_goal_approached(goal_in_robot_frame, reward_factor=0.4, penalty_factor=0.6)
        self.last_action = self._curr_action

    # def _cal_reward_rule_d86(self, laser_scan: np.ndarray, goal_in_robot_frame: Tuple[float, float], *args, **kwargs):
    #     self._curr_action = kwargs["action"]
    #     laser_min = laser_scan.min()
    #     s = None
    #     # self.curr_reward -= 0.05

    #     # self.total_time -= 0.05

    #     if VIS:
    #         print(time.time(),(kwargs["robot_pose"].x,kwargs["robot_pose"].y))
    #         print("plan:",kwargs["global_plan"])
    #     if False and kwargs["plan_result"] is not None:
    #         plan_success,track_cost,ctrl_points = kwargs["plan_result"]
    #         s,dtheta = self._calc_robot_pos_s(kwargs["global_plan"], (ctrl_points[0][-1],ctrl_points[1][-1]), (ctrl_points[2][-1],ctrl_points[3][-1]))
    #         if plan_success:
    #             self. _reward_ctrl_points(
    #                 track_cost,
    #                 ctrl_points,
    #                 track_penalty_factor = 1./135.,
    #                 smooth_penalty_factor  = 1./145.,
    #                 output_change_penalty_factor = 1/500.
    #             )
    #         else:
    #             # 规划失败惩罚
    #             self.curr_reward -= 50.
    #     else:
    #         # s,dtheta = self._calc_robot_pos_s(kwargs["global_plan"], (kwargs["robot_pose"].x,kwargs["robot_pose"].y), (kwargs["robot_vel"].x,kwargs["robot_vel"].y))
    #         e2e_smooth_penalty = 0.
    #         if self.last_action is not None:
    #             e2e_smooth_penalty = abs(self._curr_action[0] - self.last_action[0])/0.1/2. + abs(self._curr_action[1] - self.last_action[1])/0.1/2.
    #         else:
    #             e2e_smooth_penalty = abs(self._curr_action[0])/0.1/2. + abs(self._curr_action[1])/0.1/2.
            
    #         # if self.last_action is not None:
    #         #     e2e_smooth_penalty = abs(self._curr_action[0] - self.last_action[0]) + abs(self._curr_action[1] - self.last_action[1])
    #         # else:
    #         #     e2e_smooth_penalty = abs(self._curr_action[0]) + abs(self._curr_action[1])
            

    #         # e2e_slow_speed_penalty = (1. - abs(kwargs["robot_vel"].x)/2.5 - abs(kwargs["robot_vel"].y)/2.5)
    #         e2e_slow_speed_penalty = (1. - np.hypot(self._curr_action[0],self._curr_action[1])/3.53554)
            
    #         # self.total_max_speed -= e2e_slow_speed_penalty/200.
    #         # self.total_smooth -= e2e_smooth_penalty/100.
    #         # self.total_max_speed -= e2e_slow_speed_penalty*0.00375
    #         self.total_max_speed -= e2e_slow_speed_penalty*0.1
    #         self.total_smooth -= e2e_smooth_penalty*0.0375

    #         self.curr_reward -= e2e_slow_speed_penalty*0.1
    #         self.curr_reward -= e2e_smooth_penalty*0.0375

    #         # self.curr_reward -= 1./10.*e2e_smooth_penalty
    #         if VIS:
    #             print("speed up penlty:",-e2e_slow_speed_penalty*0.05)
    #             print("smmoth penlty:",-e2e_smooth_penalty*0.0375)
    #         last_dist = self.last_goal_dist
    #         dist = goal_in_robot_frame
    #         if VIS:
    #             print("speed up penlty:",-e2e_slow_speed_penalty*0.1)
    #             print("smmoth penlty:",-e2e_smooth_penalty*0.0375)

    #     self._reward_goal_approached(
    #         last_dist,
    #         dist, 
    #         # reward_factor=5., 
    #         # penalty_factor=5.,
    #         # theta_penalty_factor = 0.5
    #         reward_factor=3.5, 
    #         penalty_factor=3.5,
    #         theta_penalty_factor = 0.
    #         )

    #     # if False and s is not None and laser_min > 0.:
    #     #     # 沿全局奖励
    #     #     # self._reward_s_global_plan(
    #     #     #     s,
    #     #     #     dtheta,
    #     #     #     reward_factor=0.5,
    #     #     #     penalty_factor=0.7,
    #     #     #     theta_penalty_factor = 1./10.
    #     #     # )
    #     #     self._reward_s_global_plan(
    #     #         s,
    #     #         dtheta,
    #     #         reward_factor=1.65,
    #     #         # penalty_factor=1.8,
    #     #         penalty_factor=1.65,
    #     #         theta_penalty_factor = 1./10.
    #     #     )
    #     #     if VIS and self.last_goal_dist is not None:
    #     #         print("d dist:",goal_in_robot_frame[0]-self.last_goal_dist)
    #     #     self.last_goal_dist = goal_in_robot_frame[0]
    #     # else:
    #     #     self.last_proj_s = None
    #     #     self._reward_goal_approached(goal_in_robot_frame, 
    #     #     reward_factor=3.33, 
    #     #     # reward_factor=0.6, 
    #     #     # penalty_factor=1.8,
    #     #     penalty_factor=3.33,
    #     #     # penalty_factor=0.9,
    #     #     theta_penalty_factor = 0)
            
            
    #     # 速度方向与global一致，固定step惩罚
    #     # self._reward_goal_reached(goal_in_robot_frame, reward=30.,robot_woeld_vel=self._curr_action)
    #     # self._reward_goal_reached(goal_in_robot_frame, reward=20.)
    #     self._reward_goal_reached(goal_in_robot_frame, reward=100.)
        
    #     # self._reward_safe_dist_d86(laser_min,punish_range = 0.5, punishment=3.3) # 靠近惩罚
    #     self._reward_safe_dist_d86(laser_min,punish_range = 0.5, punishment=1.5) # 靠近惩罚
    #     # self._reward_safe_dist_d86(laser_min,punish_range = 0.5, punishment=0.1) # 靠近惩罚

    #     self._reward_collision_d86(laser_min, punishment=40.) # 碰撞惩罚
    #     # 最小化输出控制点变动
    #     self.last_action = self._curr_action
    #     self.last_goal_dist = goal_in_robot_frame[0]
    #     if VIS:
    #         print("reward:")
    #         print(self.curr_reward)
    #     self.total += self.curr_reward


    def _cal_reward_rule_d862(self, laser_scan: np.ndarray, goal_in_robot_frame: Tuple[float, float],goal_in_robot_frame_ltrajp: Tuple[float, float], *args, **kwargs):
        self._curr_action = kwargs["action"]
        laser_min = laser_scan.min()
        s = None
        plan_success = False
        if VIS:
            print(time.time(),(kwargs["robot_pose"].x,kwargs["robot_pose"].y))
            print("plan:",kwargs["global_plan"])

        if kwargs["plan_result"] is not None and kwargs["plan_result"][0]:
            plan_success,track_cost,ctrl_points = kwargs["plan_result"]
            # s,dtheta = self._calc_robot_pos_s(kwargs["global_plan"], (ctrl_points[0][-1],ctrl_points[1][-1]), (ctrl_points[2][-1],ctrl_points[3][-1]))
            
            self. _reward_ctrl_points(
                track_cost,
                ctrl_points,
                track_penalty_factor = 0.0375,
                speedup_penalty_factor = 0.05,
                smooth_penalty_factor = 0.0375,
                output_change_penalty_factor = 0.075
            )
            # last_dist = self.last_goal_dist_ltrajp
            # dist = goal_in_robot_frame_ltrajp
            last_dist = self.last_goal_dist
            dist = goal_in_robot_frame
        else:
            # 规划失败惩罚
            # 本次规划失败惩罚加到下次
            if self.last_goal_dist is not None:
                self.curr_reward -= 0.3

            # s,dtheta = self._calc_robot_pos_s(kwargs["global_plan"], (kwargs["robot_pose"].x,kwargs["robot_pose"].y), (kwargs["robot_vel"].x,kwargs["robot_vel"].y))
            e2e_smooth_penalty = 0.
            if self.last_vel is not None:
                e2e_smooth_penalty = abs(kwargs["robot_vel"].x - self.last_vel.x)/0.1/2. + abs(kwargs["robot_vel"].y - self.last_vel.y)/0.1/2.
            else:
                e2e_smooth_penalty = abs(kwargs["robot_vel"].x)/0.1/2. + abs(kwargs["robot_vel"].y)/0.1/2.
            
            e2e_slow_speed_penalty = (1. - np.hypot(kwargs["robot_vel"].x,kwargs["robot_vel"].y)/3.53554)
            
            self.total_max_speed -= e2e_slow_speed_penalty*0.1
            self.total_smooth -= e2e_smooth_penalty*0.0375

            self.curr_reward -= e2e_slow_speed_penalty*0.1
            self.curr_reward -= e2e_smooth_penalty*0.0375
            
            last_dist = self.last_goal_dist
            dist = goal_in_robot_frame
            if VIS:
                print("speed up penlty:",-e2e_slow_speed_penalty*0.1)
                print("smmoth penlty:",-e2e_smooth_penalty*0.0375)

        self._reward_goal_approached(
            last_dist,
            dist, 
            # reward_factor=1.8, 
            # penalty_factor=1.8,
            # theta_penalty_factor = 0.5
            reward_factor=3.33, 
            penalty_factor=3.33,
            theta_penalty_factor = 0.8
            )


        # if False and s is not None and laser_min > 0.:
        #     # 沿全局奖励
        #     # self._reward_s_global_plan(
        #     #     s,
        #     #     dtheta,
        #     #     reward_factor=0.5,
        #     #     penalty_factor=0.7,
        #     #     theta_penalty_factor = 1./10.
        #     # )
        #     self._reward_s_global_plan(
        #         s,
        #         dtheta,
        #         reward_factor=1.65,
        #         # penalty_factor=1.8,
        #         penalty_factor=1.65,
        #         theta_penalty_factor = 1./10.
        #     )
        #     if VIS and self.last_goal_dist is not None:
        #         print("d dist:",goal_in_robot_frame[0]-self.last_goal_dist)
        #     self.last_goal_dist = goal_in_robot_frame[0]
        # else:
        #     self.last_proj_s = None
        #     self._reward_goal_approached(goal_in_robot_frame, 
        #     reward_factor=3.33, 
        #     penalty_factor=3.33,
        #     theta_penalty_factor = 0)

        # track cost权重调大 靠近奖励调大    
        # 速度方向与global一致，固定step惩罚
        # self._reward_goal_reached(goal_in_robot_frame, reward=30.,robot_woeld_vel=self._curr_action)
        # self._reward_goal_reached(goal_in_robot_frame, reward=20.)
        self._reward_goal_reached(goal_in_robot_frame, reward=100.)
        # self._reward_safe_dist_d86(laser_min,punish_range = 0.5, punishment=3.3) # 靠近惩罚
        self._reward_safe_dist_d86(laser_min,punish_range = 0.5, punishment=0.1) # 靠近惩罚
        self._reward_collision_d86(laser_min, punishment=40.) # 碰撞惩罚
        # 最小化输出控制点变动
        
        self.last_vel = kwargs["robot_vel"]
        self.last_goal_dist = goal_in_robot_frame[0]
        if plan_success:
            self.last_goal_dist_ltrajp = goal_in_robot_frame_ltrajp[0]
            # self.last_action = ctrl_points
            self.last_action = self._curr_action
        else:
            self.last_goal_dist_ltrajp = None
            self.last_action = None

        if VIS:
            print("reward:")
            print(self.curr_reward)
        self.total += self.curr_reward

    def _calc_robot_pos_s(self,global_plan,robot_pose,robot_vel):
        if global_plan is not None and len(global_plan.s) != 0:
            s,_ = global_plan.xy_to_sd_by_kdtree(robot_pose[0],robot_pose[1])
            wvx,wvy = robot_vel[0],robot_vel[1]
            # wvtheta = np.arctan2(rvy,rvx) + robot_pose.theta
            wvtheta = np.arctan2(wvy,wvx)
            rwtheta = global_plan.calc_yaw(s)            
            return s,abs(NormalizeAngle(wvtheta - rwtheta))
        else:
            return None,None


    def _cal_reward_rule_d86(self, laser_scan: np.ndarray, goal_in_robot_frame: Tuple[float, float],goal_in_robot_frame_ltrajp: Tuple[float, float], *args, **kwargs):
        self._curr_action = kwargs["action"]
        laser_min = laser_scan.min()
        s = None
        plan_success = False
        if VIS:
            print(time.time(),(kwargs["robot_pose"].x,kwargs["robot_pose"].y))
            print("plan:",kwargs["global_plan"])


        if False and kwargs["plan_result"][0] is not None and kwargs["plan_result"][0][0]:
            plan_success = True
            convex_plan_result,polar_action_and_opt_action = kwargs["plan_result"]
            success,cost,mpc_traj_robot,action_points = convex_plan_result      
            polar_action,opt_polar_action = polar_action_and_opt_action

            # s,dtheta = self._calc_robot_pos_s(kwargs["global_plan"], (ctrl_points[0][-1],ctrl_points[1][-1]), (ctrl_points[2][-1],ctrl_points[3][-1]))
            
            # self. _reward_ctrl_points(
            #     polar_action_and_opt_action,
            #     mpc_traj_robot,
            #     # track_penalty_factor = 0.0375,
            #     # speedup_penalty_factor = 0.05,
            #     # smooth_penalty_factor = 0.0375,
            #     # output_change_penalty_factor = 0.075
            #     track_penalty_factor = 0.05, # track dist error 1m -> 0.1
            #     # speedup_penalty_factor = 0.1, # avg speed 0.25 -> 0.05 
            #     speedup_penalty_factor = 0.01, # avg speed 0.25 -> 0.05 
            #     smooth_penalty_factor = 0.15, # all smooth error 10 -> 0.05
            #     output_change_penalty_factor = 0.025 # all change dist 1m -> 0.1
            # )


            last_dist = self.last_goal_dist_ltrajp
            dist = goal_in_robot_frame_ltrajp
            # last_dist = self.last_goal_dist
            # dist = goal_in_robot_frame

            self._reward_goal_approached(
                last_dist,
                dist, 
                # reward_factor=1.8, 
                # penalty_factor=1.8,
                # theta_penalty_factor = 0.5
                # reward_factor=3.3, 
                # penalty_factor=4.,
                reward_factor=5., 
                penalty_factor=5.,
                theta_penalty_factor = 0.5,
                close_penalty_limit = 1.5
                )
            # if not kwargs["galaxy_success"]:                
            #     self.curr_reward -= 0.2
        else:
            # 规划失败惩罚
            # if self.last_goal_dist is not None:
            #     self.curr_reward -= 0.4

            # s,dtheta = self._calc_robot_pos_s(kwargs["global_plan"], (kwargs["robot_pose"].x,kwargs["robot_pose"].y), (kwargs["robot_vel"].x,kwargs["robot_vel"].y))
            e2e_smooth_penalty = 0.
            if self.last_vel is not None:
                e2e_smooth_penalty = abs(kwargs["robot_vel"].x - self.last_vel.x)/0.1/2. + abs(kwargs["robot_vel"].y - self.last_vel.y)/0.1/2.
            else:
                e2e_smooth_penalty = abs(kwargs["robot_vel"].x)/0.1/2. + abs(kwargs["robot_vel"].y)/0.1/2.
            
            e2e_slow_speed_penalty = (1. - np.hypot(kwargs["robot_vel"].x,kwargs["robot_vel"].y)/3.53554)
            
            # self.total_max_speed -= e2e_slow_speed_penalty*0.1
            # self.total_smooth -= e2e_smooth_penalty*0.0375

            # self.curr_reward -= e2e_slow_speed_penalty*0.1
            # self.curr_reward -= e2e_smooth_penalty*0.0375

            self.total_max_speed -= e2e_slow_speed_penalty*0.005
            self.total_smooth -= e2e_smooth_penalty*0.00375

            self.curr_reward -= e2e_slow_speed_penalty*0.005
            self.curr_reward -= e2e_smooth_penalty*0.00375
            
            last_dist = self.last_goal_dist
            dist = goal_in_robot_frame
            if VIS:
                print("speed up penlty:",-e2e_slow_speed_penalty*0.1)
                print("smmoth penlty:",-e2e_smooth_penalty*0.0375)

            self._reward_goal_approached(
                last_dist,
                dist, 
                # reward_factor=1.8, 
                # penalty_factor=1.8,
                # theta_penalty_factor = 0.5

                # reward_factor=5., 
                # penalty_factor=5.,
                # theta_penalty_factor = 0.5,

                reward_factor=0.5, 
                penalty_factor=0.5,
                theta_penalty_factor = 0.,
                close_penalty_limit = 1.5
            )
        

        self._reward_goal_reached(goal_in_robot_frame, reward=125.)
        self._reward_safe_dist_d86(laser_min,punish_range = 2., punishment=1.) # 靠近惩罚
        self._reward_collision_d86(laser_min, punishment=5.) # 碰撞惩罚

        # self._reward_goal_reached(goal_in_robot_frame, reward=100.)
        # # self._reward_safe_dist_d86(laser_min,punish_range = 2., punishment=1.) # 靠近惩罚
        # self._reward_collision_d86(laser_min, punishment=40.) # 碰撞惩罚
        # 最小化输出控制点变动
        
        self.last_vel = kwargs["robot_vel"]
        self.last_goal_dist = goal_in_robot_frame[0]
        if plan_success:
            self.last_goal_dist_ltrajp = goal_in_robot_frame_ltrajp[0]
            # self.last_action = ctrl_points
            self.last_action = self._curr_action
        else:
            self.last_goal_dist_ltrajp = None
            self.last_action = None

        if VIS:
            print("reward:")
            print(self.curr_reward)
        self.total += self.curr_reward


    def _set_current_dist_to_globalplan(self, global_plan: np.ndarray, robot_pose: Pose2D):
        if global_plan is not None and len(global_plan) != 0:
            self._curr_dist_to_path, idx = self.get_min_dist2global_kdtree(global_plan, robot_pose)
    
    def _reward_ctrl_points2(self,
        track_cost,
        ctrl_points,
        track_penalty_factor: float = 1.,
        speedup_penalty_factor: float = 1.,
        smooth_penalty_factor: float = 1.,  
        output_change_penalty_factor: float = 1.,
    ):
        
        x,y,vx,vy,ax,ay,jx,jy = ctrl_points
        smooth_penalty = 0.
        speedup_penalty = 0.
        # 整体速度快
        for i in range(len(vx)-1):
            speedup_penalty -= (1. - np.hypot(vx[i],vy[i])/3.53554)

        for i in range(len(ax)-1):
            smooth_penalty -= np.hypot(ax[i],ay[i])/2.82843

        for i in range(len(jx)-1):
            smooth_penalty -= np.hypot(jx[i],jy[i])/5.656855

        # 终态速度尽量小
        smooth_penalty -= 2*np.hypot(vx[-1],vy[-1])/3.53554
        smooth_penalty -= 2*np.hypot(ax[-1],ay[-1])/2.82843
        smooth_penalty -= 2*np.hypot(jx[-1],jy[-1])/5.656855
        
        output_change_penalty = 0.
        if self.last_action is not None:
            # 最小化网络输出控制点的变化幅度，不希望每次重规划控制点变化太大
            for i in range(len(self.last_action)):
                output_change_penalty -= np.hypot(self._curr_action[i][0] - self.last_action[i][0],self._curr_action[i][1] - self.last_action[i][1])


        if VIS:
            print("track cost:")
            print(-track_penalty_factor*track_cost)
            print("smooth:")
            print(smooth_penalty_factor*smooth_penalty)
            print("output change:")
            print(output_change_penalty_factor*output_change_penalty)
        
        self.curr_reward -= track_penalty_factor*track_cost/200.
        self.curr_reward += speedup_penalty_factor*speedup_penalty/6.
        self.curr_reward += smooth_penalty_factor*smooth_penalty/12.
        self.curr_reward += output_change_penalty_factor*output_change_penalty/0.25/60.
        
        self.total_track-= track_penalty_factor*track_cost/200.
        self.total_max_speed += speedup_penalty_factor*speedup_penalty/6.
        self.total_smooth += smooth_penalty_factor*smooth_penalty/12.
        self.total_change += output_change_penalty_factor*output_change_penalty/0.25/60.

    
    def _reward_ctrl_points(self,
        polar_action_and_opt_action,
        mpc_traj_robot,
        track_penalty_factor: float = 1.,
        speedup_penalty_factor: float = 1.,
        smooth_penalty_factor: float = 1.,  
        output_change_penalty_factor: float = 1.,
    ):
        polar_action,opt_polar_action = polar_action_and_opt_action
        track_cost = 0.

        for i in range(len(polar_action)):
            x,y = polar_action[i][1]*np.cos(polar_action[i][0]),polar_action[i][1]*np.sin(polar_action[i][0])
            opt_x,opt_y = opt_polar_action[i][1]*np.cos(opt_polar_action[i][0]),opt_polar_action[i][1]*np.sin(opt_polar_action[i][0])
            track_cost += np.hypot(x-opt_x,y-opt_y)
        
        if abs(track_cost) > 100:
            print("track")
            print(polar_action)
            print(opt_polar_action)
        # 三阶
        rp,rv,ra,rj = mpc_traj_robot
        # 二阶
        # rp,rv,ra = mpc_traj_robot

        smooth_penalty = 0.
        speedup_penalty = 0.


        # 整体速度快
        for i in range(len(rv)-1):
            speedup_penalty -= (1. - np.hypot(rv[i][0],rv[i][1])/3.53554)

        for i in range(len(ra)-1):
            smooth_penalty -= np.hypot(ra[i][0],ra[i][1])/5.656855

        for i in range(len(rj)-1):
            smooth_penalty -= np.hypot(rj[i][0],rj[i][1])/11.3137085

        if abs(speedup_penalty) > 1000:
            print("speedup")
            print(rv)


        # 终态速度尽量小
        smooth_penalty -= 2*np.hypot(rv[-1][0],rv[-1][1])/3.53554
        smooth_penalty -= 2*np.hypot(ra[-1][0],ra[-1][1])/5.656855
        smooth_penalty -= 2*np.hypot(rj[-1][0],rj[-1][1])/11.3137085

        if abs(smooth_penalty) > 2000:
            print("smooth")
            print(rv,ra,rj)

        output_change_penalty = 0.
        if self.last_action is not None:
            # 最小化网络输出控制点的变化幅度，不希望每次重规划控制点变化太大
            for i in range(len(self.last_action)):
                output_change_penalty -= np.hypot(self._curr_action[i][0] - self.last_action[i][0],self._curr_action[i][1] - self.last_action[i][1])

        if abs(output_change_penalty) > 1000:
            print("change")
            print(self._curr_action)
            print(self.last_action)

        if VIS:
            print("track cost:")
            print(-track_cost)
            print(-track_penalty_factor*track_cost)
            print("smooth:")
            print(smooth_penalty)
            print(smooth_penalty_factor*smooth_penalty/21.)
            print("output change:")
            print(output_change_penalty)
            print(output_change_penalty_factor*output_change_penalty)
            print("speed up:")
            print(speedup_penalty)
            print(speedup_penalty_factor*speedup_penalty/9.)
            print("ctrl penalty:")
            print(output_change_penalty_factor*output_change_penalty - track_penalty_factor*track_cost + smooth_penalty_factor*smooth_penalty/21. + speedup_penalty_factor*speedup_penalty/9.)
        self.curr_reward -= track_penalty_factor*track_cost
        self.curr_reward += speedup_penalty_factor*speedup_penalty/9.
        self.curr_reward += smooth_penalty_factor*smooth_penalty/21.
        self.curr_reward += output_change_penalty_factor*output_change_penalty
        
        self.total_track-= track_penalty_factor*track_cost
        self.total_max_speed += speedup_penalty_factor*speedup_penalty/9.
        self.total_smooth += smooth_penalty_factor*smooth_penalty/21.
        self.total_change += output_change_penalty_factor*output_change_penalty



    def _reward_s_global_plan(
        self,
        now_s,
        dtheta,
        reward_factor: float = 0.1,
        penalty_factor: float = 0.15,
        theta_penalty_factor: float = 0.15
    ):
        """
        Reward for approaching/veering away the global plan. (Weighted difference between
        prior distance to global plan and current distance to global plan)

        :param global_plan: (np.ndarray): vector containing poses on global plan
        :param robot_pose (Pose2D): robot position
        :param reward_factor (float, optional): positive factor when approaching global plan. defaults to 0.1
        :param penalty_factor (float, optional): negative factor when veering away from global plan. defaults to 0.15
        """
        if now_s is not None:
            if self.last_proj_s is not None:
                if now_s < self.last_proj_s:
                    w = penalty_factor 
                else:
                    w = reward_factor

                self.curr_reward += w * (now_s - self.last_proj_s)
                if VIS:
                    if abs((now_s - self.last_proj_s)) > 5:
                        print("path s reward:")
                        print(w * (now_s - self.last_proj_s))

                        print("proj ds:")
                        print(now_s - self.last_proj_s)

            # self.curr_reward -= theta_penalty_factor*dtheta
            # if VIS:
            #     print("theta penalty:")
            #     print(-theta_penalty_factor*dtheta)
            self.last_proj_s = now_s

    def _reward_goal_reached(self, goal_in_robot_frame=Tuple[float, float], reward: float = 15,robot_woeld_vel = None):
        """
        Reward for reaching the goal.

        :param goal_in_robot_frame (Tuple[float,float]): position (rho, theta) of the goal in robot frame (Polar coordinate)
        :param reward (float, optional): reward amount for reaching. defaults to 15
        """
        if goal_in_robot_frame[0] < self.goal_radius:
            if robot_woeld_vel is None:
                self.curr_reward += reward
            else:
                self.curr_reward += reward * (1 - 0.375*(abs(robot_woeld_vel[0])/2.5+abs(robot_woeld_vel[1])/2.5))
            self.info["is_done"] = True
            self.info["done_reason"] = 2
            self.info["is_success"] = 1
        else:
            self.info["is_done"] = False

    def _reward_goal_approached(
        self,
        last_goal_dist,
        goal_in_robot_frame=Tuple[float, float],
        reward_factor: float = 0.3,
        penalty_factor: float = 0.5,
        theta_penalty_factor:float = 1./10.,
        close_penalty_limit:float = 0.8
    ):
        """
        Reward for approaching the goal.

        :param goal_in_robot_frame (Tuple[float,float]): position (rho, theta) of the goal in robot frame (Polar coordinate)
        :param reward_factor (float, optional): positive factor for approaching goal. defaults to 0.3
        :param penalty_factor (float, optional): negative factor for withdrawing from goal. defaults to 0.5
        """
        if last_goal_dist is not None:
            # goal_in_robot_frame : [rho, theta]

            # higher negative weight when moving away from goal
            # (to avoid driving unnecessary circles when train in contin. action space)
            if (last_goal_dist - goal_in_robot_frame[0]) > 0:
                w = reward_factor
            else:
                w = penalty_factor
            reward = w * (last_goal_dist - goal_in_robot_frame[0])
            if reward > close_penalty_limit:
                reward = close_penalty_limit
            if reward < -close_penalty_limit:
                reward = -close_penalty_limit
            self.curr_reward += reward
            self.total_approach += reward
            dtheta = goal_in_robot_frame[1]
            # print("dtheta,reward")
            # print(dtheta,-theta_penalty_factor*abs(dtheta/np.pi))
            self.curr_reward -= theta_penalty_factor*abs(dtheta/np.pi)
            self.total_theta_approach -= theta_penalty_factor*abs(dtheta/np.pi)
            if VIS:
                # if abs(self.last_goal_dist - goal_in_robot_frame[0]) > 5.:
                print(last_goal_dist,goal_in_robot_frame[0])
                print("close reward:",reward)
                print("vtheta penalty:",-theta_penalty_factor*abs(dtheta/np.pi))
            # print("reward_goal_approached:  {}".format(reward))

        # last_goal_dist = goal_in_robot_frame[0]

    def _reward_collision(self, laser_scan: np.ndarray, punishment: float = 10):
        """
        Reward for colliding with an obstacle.

        :param laser_scan (np.ndarray): laser scan data
        :param punishment (float, optional): punishment for collision. defaults to 10
        """
        if laser_scan.min() <= 0.:
            self.curr_reward -= punishment

            if not self._extended_eval:
                self.info["is_done"] = True
                self.info["done_reason"] = 1
                self.info["is_success"] = 0
            else:
                self.info["crash"] = True

    def _reward_collision_d86(self, laser_min: float, punishment: float = 10):
        """
        Reward for colliding with an obstacle.

        :param laser_scan (np.ndarray): laser scan data
        :param punishment (float, optional): punishment for collision. defaults to 10
        """
        if laser_min <= 1e-7:
            if VIS:
                for i in range(50):
                    print("col!!")
            self.curr_reward -= punishment

            if not self._extended_eval:
                self.info["is_done"] = True
                self.info["done_reason"] = 1
                self.info["is_success"] = 0
            else:
                self.info["crash"] = True

    def _reward_safe_dist(self, laser_scan: np.ndarray,punish_range: float = 0.5, punishment: float = 0.15):
        """
        Reward for undercutting safe distance.

        :param laser_scan (np.ndarray): laser scan data
        :param punishment (float, optional): punishment for undercutting. defaults to 0.15
        """
        if laser_scan.min() < punish_range:
            self.curr_reward -= punishment

            if self._extended_eval:
                self.info["safe_dist"] = True

    def _reward_safe_dist_d86(self, laser_min: float,punish_range: float = 0.2, punishment: float = 0.85):
        """
        Reward for undercutting safe distance.

        :param laser_scan (np.ndarray): laser scan data
        :param punishment (float, optional): punishment for undercutting. defaults to 0.15
        """
        if laser_min < punish_range:
            self.curr_reward -=  punishment*np.exp(-1.0*1.1*(laser_min - 0.))
            if VIS:
                print("safe dist:")
                print(-punishment*np.exp(-1.0*0.5*(laser_min - 0.)))
                self.total_safe_penalty -=  punishment*np.exp(-1.0*0.5*(laser_min - 0.))
            if self._extended_eval:
                self.info["safe_dist"] = True


    def _reward_not_moving(self, action: np.ndarray = None, punishment: float = 0.01):
        """
        Reward for not moving. Only applies half of the punishment amount
        when angular velocity is larger than zero.

        :param action (np.ndarray (,2)): [0] - linear velocity, [1] - angular velocity
        :param punishment (float, optional): punishment for not moving. defaults to 0.01
        """
        if action is not None and action[0] == 0.0:
            self.curr_reward -= punishment if action[1] == 0.0 else punishment / 2

    def _reward_distance_traveled(
        self,
        action: np.array = None,
        punishment: float = 0.01,
        consumption_factor: float = 0.005,
    ):
        """
        Reward for driving a certain distance. Supposed to represent "fuel consumption".

        :param action (np.ndarray (,2)): [0] - linear velocity, [1] - angular velocity
        :param punishment (float, optional): punishment when action can't be retrieved. defaults to 0.01
        :param consumption_factor (float, optional): weighted velocity punishment. defaults to 0.01
        """
        if action is None:
            self.curr_reward -= punishment
        else:
            lin_vel = action[0]
            ang_vel = action[-1]
            reward = (lin_vel + (ang_vel * 0.001)) * consumption_factor
        self.curr_reward -= reward

    def _reward_distance_global_plan(
        self,
        reward_factor: float = 0.1,
        penalty_factor: float = 0.15,
    ):
        """
        Reward for approaching/veering away the global plan. (Weighted difference between
        prior distance to global plan and current distance to global plan)

        :param global_plan: (np.ndarray): vector containing poses on global plan
        :param robot_pose (Pose2D): robot position
        :param reward_factor (float, optional): positive factor when approaching global plan. defaults to 0.1
        :param penalty_factor (float, optional): negative factor when veering away from global plan. defaults to 0.15
        """
        if self._curr_dist_to_path:
            if self.last_dist_to_path is not None:
                if self._curr_dist_to_path < self.last_dist_to_path:
                    w = reward_factor
                else:
                    w = penalty_factor

                self.curr_reward += w * (self.last_dist_to_path - self._curr_dist_to_path)
            self.last_dist_to_path = self._curr_dist_to_path

    def _reward_following_global_plan(
        self,
        action: np.array = None,
        dist_to_path: float = 0.5,
    ):
        """
        Reward for travelling on the global plan.

        :param global_plan: (np.ndarray): vector containing poses on global plan
        :param robot_pose (Pose2D): robot position
        :param action (np.ndarray (,2)): [0] = linear velocity, [1] = angular velocity
        :param dist_to_path (float, optional): applies reward within this distance
        """
        if self._curr_dist_to_path and action is not None and self._curr_dist_to_path <= dist_to_path:
            self.curr_reward += 0.1 * action[0]

    def get_min_dist2global_kdtree(self, global_plan: np.array, robot_pose: Pose2D):
        """
        Calculates minimal distance to global plan using kd tree search.

        :param global_plan: (np.ndarray): vector containing poses on global plan
        :param robot_pose (Pose2D): robot position
        """
        if self.kdtree is None:
            self.kdtree = scipy.spatial.cKDTree(global_plan)

        dist, index = self.kdtree.query([robot_pose.x, robot_pose.y])
        return dist, index

    def _reward_abrupt_direction_change(self, action: np.ndarray = None):
        """
        Applies a penalty when an abrupt change of direction occured.

        :param action: (np.ndarray (,2)): [0] = linear velocity, [1] = angular velocity
        """
        if self.last_action is not None:
            curr_ang_vel = action[-1]
            last_ang_vel = self.last_action[-1]

            vel_diff = abs(curr_ang_vel - last_ang_vel)
            self.curr_reward -= (vel_diff ** 4) / 50
        self.last_action = action

    def _reward_reverse_drive(self, action: np.array = None, penalty: float = 0.01):
        """
        Applies a penalty when an abrupt change of direction occured.

        :param action: (np.ndarray (,2)): [0] = linear velocity, [1] = angular velocity
        """
        if action is not None and action[0] < 0:
            self.curr_reward -= penalty

    def _reward_abrupt_vel_change(self, vel_idx: int, factor: float = 1):
        """
        Applies a penalty when an abrupt change of direction occured.

        :param action: (np.ndarray (,2)): [0] = linear velocity, [1] = angular velocity
        """
        if self.last_action is not None:
            curr_vel = self._curr_action[vel_idx]
            last_vel = self.last_action[vel_idx]

            vel_diff = abs(curr_vel - last_vel)
            self.curr_reward -= ((vel_diff ** 4) / 100) * factor
