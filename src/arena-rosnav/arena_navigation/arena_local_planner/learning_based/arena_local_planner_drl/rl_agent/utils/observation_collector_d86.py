#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import imp
import threading
from typing import Tuple
import os,sys
import subprocess
from numpy.core.numeric import normalize_axis_tuple
import rospy
import rosservice
import rospkg

import random
import numpy as np
from collections import deque

import time  # for debuging
import threading

# observation msgs
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D, PoseStamped, Pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from flatland_msg.msg import Galaxy2D
# services
from flatland_msgs.srv import StepWorld, StepWorldRequest
# from arena_plan_msgs.srv import MakeGlobalPlan,MakeGlobalPlanRequest
from arena_intermediate_planner.srv import Global_Kino_MakePlan,Global_Kino_MakePlanRequest

# message filter
import message_filters

# for transformations
from rl_agent.utils.transformations import *
# from tf.transformations import *

from gym import spaces
import numpy as np

from std_msgs.msg import Bool

from rl_agent.utils.debug import timeit
import sys
sys.path.append(r"/home/chen/desire_10086/flat_ped_sim/src/d86env")
from galaxy2d import galaxy_xyin_360out
from cubic_spline import Spline2D

def NormalizeAngle(d_theta):# -pi-pi
    d_theta_normalize = d_theta
    while d_theta_normalize > math.pi:
        d_theta_normalize = d_theta_normalize - 2 * math.pi
    while d_theta_normalize < -math.pi:
        d_theta_normalize = d_theta_normalize + 2 * math.pi
    return d_theta_normalize

# obs周期 robot_action_rate
class ObservationCollector_d86:
    def __init__(
        self,
        ns: str,
        num_lidar_beams: int,
        lidar_range: float,
        safe_dist:float,
        exp_dist:float,
        # start:Pose2D(),
        # goal:Pose2D(),
        laser_origin_in_base:list=[0.,0.,0.],
        plan_dynamic_limit:tuple=(2.5,2.,4.),
        stack_times:int = 4,
        external_time_sync: bool = False,
    ):
        self.ns = ns
        if ns is None or not ns:
            self.ns_prefix = ""
        else:
            self.ns_prefix = "/" + ns + "/"

        # define observation_space
        self.goal = None
        self.observation_space = None
        self.lidar_range = lidar_range
        self.stack_times = stack_times
        self.sigle_state_dim = num_lidar_beams + 2 + 1 + 1
        self.state_dim = self.stack_times*self.sigle_state_dim # + num_lidar_beams
        self.plan_dynamic_limit = plan_dynamic_limit
        self.laser_origin_in_base = laser_origin_in_base
        self.laser_num_beams = num_lidar_beams
        self.drad = 1./num_lidar_beams*2*np.pi
        self.safe_dist = safe_dist
        self.exp_dist =exp_dist

        self.new_state_come = False
        self.new_scan_come = False
        self.last_done = False
        self.scan_que = np.zeros((self.stack_times,self.laser_num_beams), dtype=float)
        self.state_que = np.zeros((self.stack_times,4), dtype=float)
        self.construct_obs_space()
        
        # for frequency controlling
        self._action_frequency = 1 / rospy.get_param("/robot_action_rate")

        self._clock = Clock()
        self.scan = None
        self._scan = LaserScan()
        self._scan_xy_robot_frame = []
        self.state = None
        self._robot_pose = Pose2D()
        self._robot_vel = Twist()
        # self._subgoal = Pose2D()
        self._globalplan = None
        self.merged_obs = None
        self.obs_dict = {
            "laser_scan": None,
            "goal_in_robot_frame": [None, None],
            "global_plan": self._globalplan,
            "robot_world_pose": self._robot_pose,
            "robot_world_vel": self._robot_vel.linear,
             "laser_convex":[None,None,None],
        }
        self.n_res = {
            "n_scan":None,
            "n_polar_convex":None,
            "ntheta":None,
            "nrho":None,
            "nrvx":None,
            "nrvy":None
        }
        self.convex_vertex_now = None
        # train mode?
        self._is_train_mode = rospy.get_param("/train_mode")
        # self._start_global_planner()
        
        # synchronization parameters
        self._ext_time_sync = external_time_sync
        self._first_sync_obs = (
            True  # whether to return first sync'd obs or most recent
        )
        self.max_deque_size = 10
        self._sync_slop = 0.05

        self._laser_deque = deque()
        self._rs_deque = deque()

        # subscriptions
        # ApproximateTimeSynchronizer appears to be slow for training, but with real robot, own sync method doesn't accept almost any messages as synced
        # need to evaulate each possibility
        # _ext_time_sync对于训练来说很慢，但是对于真实robot采取自己的同步方法(not _ext_time_sync)几乎接收不到任何信息
        #  ,也就是说真实机器人要采用_ext_time_sync，训练时采用not _ext_time_sync
        # 同步上，点云帧与此点云时刻对应的机器人状态
        if self._ext_time_sync: # 同步接收
            self._scan_sub = message_filters.Subscriber(
                f"{self.ns_prefix}scan", LaserScan
            )
            self._robot_state_sub = message_filters.Subscriber(
                f"{self.ns_prefix}odom", Odometry
            )

            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self._scan_sub, self._robot_state_sub],
                self.max_deque_size,
                slop=self._sync_slop,
            )
            # self.ts = message_filters.TimeSynchronizer([self._scan_sub, self._robot_state_sub], 10)
            self.ts.registerCallback(self.callback_odom_scan)
            
        else:
            self._scan_sub = rospy.Subscriber(
                f"{self.ns_prefix}scan",
                LaserScan,
                self.callback_scan,
                tcp_nodelay=True,
            )

            self._galaxy_sub = rospy.Subscriber(
                f"{self.ns_prefix}galaxy2d",
                Galaxy2D,
                self.callback_galaxy2d,
                tcp_nodelay=True,
            )

            # 仿真中将ground truth加了噪声后发布
            self._robot_state_sub = rospy.Subscriber(
                f"{self.ns_prefix}odom",
                Odometry,
                self.callback_robot_state,
                tcp_nodelay=True,
            )
            self.last_time = None

        # self._clock_sub = rospy.Subscriber(
        #     f'{self.ns_prefix}clock', Clock, self.callback_clock, tcp_nodelay=True)

        # self._subgoal_sub = rospy.Subscriber(
        #     f"{self.ns_prefix}subgoal", PoseStamped, self.callback_subgoal
        # )

        self._globalplan_sub = rospy.Subscriber(
            f"{self.ns_prefix}inter_planner/globalPlan", Path, self.callback_global_plan
        )

        # ret = self._wait_for_global_plan_service()
        # if not ret:
        #     for i in range(1000):
        #         print("global plan node fail!!!!!!!!!!!!!!!!!!!!!")

    def construct_obs_space(self):
        obs_space = []
        # 观测空间一旦设置后返回给环境，环境会用这个空间初始化网络，且只初始化一次
        # 不能多次重置,所以直接将需要动态改变的观测值 距离和动作中的点云距离设为-1~1 和 0~1
        # 再在collector乘上对应要求的长度
        # for i in range(self.stack_times):
        #     obs_space.extend([
        #         spaces.Box(
        #                 low=0,
        #                 high=self.lidar_range,
        #                 shape=(self.laser_num_beams,),
        #                 dtype=np.float32,
        #             ),
        #             spaces.Box(
        #                 low=-np.pi,
        #                 high=np.pi,
        #                 shape=(1,),
        #                 dtype=np.float32,  # dtheta 速度方向与终态直线角 之差
        #             ),                
        #             spaces.Box(
        #                 low=0, high=100., shape=(1,), dtype=np.float32
        #             ),  # dist2goal 为了更新距离上下限
        #             spaces.Box(
        #                 low=self.dynamic_space.low[0],
        #                 high=self.dynamic_space.high[0],
        #                 shape=(2,),
        #                 dtype=np.float32,  # linear vel
        #             ),
        #             # spaces.Box(
        #             #     low=self.dynamic_space.low[1],
        #             #     high=self.dynamic_space.high[1],
        #             #     shape=(1,),
        #             #     dtype=np.float32,  # angular vel
        #             # )
        #     ])
        obs_space.extend([
            spaces.Box(
                    low=0.,
                    high=1.,
                    shape=(self.stack_times*self.laser_num_beams,),
                    dtype=np.float32,
                ),
                spaces.Box(
                    low=0.,
                    high=1.,
                    shape=(self.stack_times,),
                    dtype=np.float32,  # dtheta 速度方向与终态直线角 之差
                ),                
                spaces.Box(
                    low=0., high=1., shape=(self.stack_times,), dtype=np.float32
                ),  # dist2goal 为了更新距离上下限
                spaces.Box(
                    low=0.,
                    high=1.,
                    shape=(2*self.stack_times,),
                    dtype=np.float32,  # linear vel
                ),
                # spaces.Box(
                #     low=self.dynamic_space.low[1],
                #     high=self.dynamic_space.high[1],
                #     shape=(1,),
                #     dtype=np.float32,  # angular vel
                # )
        ])


        # obs_space.append(spaces.Box(
        #             low=0.,
        #             high=1.,
        #             shape=(self.laser_num_beams,),
        #             dtype=np.float32,
        #         )
        #     )
        self.observation_space = ObservationCollector_d86._stack_spaces(tuple(obs_space))


    def reset(self,start_pos,goal_pos):
        # try:
        #     rospy.wait_for_service(f"{self.ns_prefix}scan",timeout=2.)
        # except rospy.ROSException:
        #     print(self.ns+"_scan timeout")
        # try:
        #     rospy.wait_for_service(f"{self.ns_prefix}odom",timeout=2.)
        # except rospy.ROSException:
        #     print(self.ns+"_odom timeout")

        self.goal = goal_pos
        # 5倍曼哈顿距离
        self.dist2goal_norm = np.hypot(goal_pos.x - start_pos.x,goal_pos.y - start_pos.y)

        self._scan_xy_robot_frame = [(0.,0.) for i in range(self.laser_num_beams)]
        # convex_vertex_now,polar_convex_now,polar_convex_now_theta = galaxy_xyin_360out(self._scan_xy_robot_frame,radius = self.lidar_range)
        self._robot_pose.x = start_pos.x
        self._robot_pose.y = start_pos.y
        self._robot_vel.linear.x = 0.
        self._robot_vel.linear.y = 0.
        # self.n_res["n_scan"] = np.zeros(self.laser_num_beams, dtype=float)
        # self.n_res["ntheta"] = 0.
        # self.n_res["nrho"] = self.dist2goal_norm/40.
        # self.n_res["nrvx"] = 0.
        # self.n_res["nrvy"] = 0.
        # self.n_res["n_polar_convex"] = np.zeros(self.laser_num_beams, dtype=float)
        # self.obs_dict["laser_scan"] = np.zeros(self.laser_num_beams, dtype=float)
        # self.obs_dict["laser_convex"] = [convex_vertex_now,polar_convex_now,polar_convex_now_theta]
        # self.obs_dict["goal_in_robot_frame"] = [self.dist2goal_norm, 0.]
        # self.obs_dict["robot_world_pose"] = self._robot_pose
        # self.obs_dict["robot_world_vel"] = self._robot_vel.linear
        # self.convex_vertex_now = None
        # define observation_space
        # scan,dist2goal,车辆yaw与 终点车辆连线所成角的差值,vx,vy,w
        # self.observation_space = self.get_observation_space()
        self.last_done = True


    def _start_global_planner(self):
        # Generate local planner node
        package = 'arena_bringup'
        launch_file = 'intermediate_planner.launch'
        if self.ns == "":
            arg1 = "ns:=_"
        else:
            arg1 = "ns:=" + self.ns


        # Use subprocess to execute .launch file
        if self._is_train_mode:
            self._global_planner_process = subprocess.Popen(["roslaunch", package, launch_file, arg1],
                                                            stdout=subprocess.DEVNULL,
                                                            stderr=subprocess.STDOUT)
        else:
            self._global_planner_process = subprocess.Popen(["roslaunch", package, launch_file, arg1],
                                                            stdout=subprocess.DEVNULL)

        self._global_plan_service = None
        # self._reset_global_costmap_service = None

    def _make_new_global_plan(self,start_pos, goal_pos):
        self._globalplan = None        
        req = Global_Kino_MakePlanRequest()
        goal_msg = PoseStamped()
        pose_goal = Pose()
        pose_goal.position.x = goal_pos.x
        pose_goal.position.y = goal_pos.y
        goal_msg.pose = pose_goal
        goal_msg.header.frame_id = "map"
        req.goal = goal_msg

        start_msg = PoseStamped()
        pose_start = Pose()
        pose_start.position.x = start_pos.x
        pose_start.position.y = start_pos.y
        start_msg.pose = pose_start
        start_msg.header.frame_id = "map"
        req.start = start_msg
        req.tolerance = 0.05

        globalplan_raw = None
        # try:
        #     rospy.wait_for_service(self.ns_prefix + "global_kino_make_plan",timeout=1.)
        #     globalplan_raw = self._global_plan_service(req).plan
        # except rospy.ServiceException as e:
        #     rospy.logwarn(e)
        #     # rospy.logdebug("step Service call failed: %s" % e)
        # except rospy.ROSException: 
        #     rospy.logwarn("make plan time out")
        #     # rospy.logdebug("time out")
        
        # unable to receive data from sender, check sender's logs for details
        if globalplan_raw is not None and not len(globalplan_raw.poses) == 0:
            # self._globalplan_raw = globalplan_raw
            self._globalplan = self.process_global_plan_msg(globalplan_raw)

            # change frame_id from /map to map
            # for pose in self._globalplan_raw.poses:
            #     pose.header.frame_id = "map"

    def _wait_for_global_plan_service(self) -> bool:
        # wait until service is available
        make_plan_service_name = self.ns_prefix + "global_kino_make_plan" 
        # reset_costmap_service_name = self.ns_prefix + "global_planner" + "/" + "resetGlobalCostmap"
        max_tries = 100
        service_list = rosservice.get_service_list()
        for i in range(max_tries):
            if make_plan_service_name in service_list:
                break
            else:
                time.sleep(0.3)
                service_list = rosservice.get_service_list()

        if make_plan_service_name in service_list:
            self._global_plan_service = rospy.ServiceProxy(make_plan_service_name,
                                                           Global_Kino_MakePlan,persistent=True)
            self.make_plan_service_name = make_plan_service_name
            # self._reset_global_costmap_service = rospy.ServiceProxy(reset_costmap_service_name,
            #                                                         std_srvs.srv.Empty,
            #                                                         persistent=True)
            return True
        else:
            rospy.wait_for_service(make_plan_service_name)
            return False

    def get_observation_space(self):
        return self.observation_space

    def wait_for_sensor_info(self):
        timeout = 20
        for i in range(timeout):
            if self.new_scan_come and self.new_state_come:
                self.new_scan_come = False
                self.new_state_come = False
                break
            else:
                if i == timeout - 1:
                    # print(self.ns + "timeout")
                    rospy.logwarn("action next sensor time out")
                    raise TimeoutError(
                        f"Timeout while trying to call '{self.ns_prefix}sensor'"
                    )
                time.sleep(0.005)
                continue

    def get_observations2(self, *args, **kwargs):
        # apply action time horizon
        # if self._is_train_mode:
        #     # 向前仿真到需要采取下个动作的周期
        #     self.call_service_takeSimStep(self._action_frequency)
        # else:
        #     try:
        #         rospy.wait_for_message(f"{self.ns_prefix}next_cycle", Bool)
        #     except Exception:
        #         pass
        
        if not self._ext_time_sync:
            self.wait_for_sensor_info()
            # try to retrieve sync'ed obs
            # laser_scan,scans_xy, robot_state = self.get_sync_obs()
            laser_scan,scans_xy, robot_state = None,None,None
            ##############
            # if len(self._rs_deque) > 0 and len(self._laser_deque) > 0:
            #     laser_scan,scans_xy = self.process_scan_msg(self._laser_deque[-1])
            #     robot_state= self.process_robot_state_msg(self._rs_deque[-1])
            
            if self.scan is not None and self.state is not None:
                laser_scan,scans_xy = self.process_scan_msg(self.scan)
                robot_state= self.process_robot_state_msg(self.state)
            ##############
            if laser_scan is not None and robot_state is not None:
                # print("Synced successfully")
                self._scan = laser_scan
                self._scan_xy_robot_frame = scans_xy
                self._robot_pose = robot_state[0]
                self._robot_vel = robot_state[1]
            self.scan = None
            self.state = None
            # else:
            #     print("Not synced")

        if len(self._scan.ranges) > 0:
            scan = self._scan.ranges.astype(np.float32)
        else:
            scan = np.zeros(self.laser_num_beams, dtype=float)

        # 输入xy逆时针 convex_vertex_now顶点顺时针 polar_convex_now散点顺时针
        
        # convex_vertex_now,polar_convex_now,polar_convex_now_theta = galaxy_xyin_360out(self._scan_xy_robot_frame,radius = self.lidar_range)
        convex_vertex_now,polar_convex_now,polar_convex_now_theta = None,None,None
        # rho, theta = ObservationCollector_d86._get_goal_pose_in_robot_frame(
        #     self.goal, self._robot_pose,self._robot_vel
        # )
        
        # rho, dtheta = self._get_goal_pose_in_robot_frame(
        #     self.goal, self._robot_pose,self._robot_vel
        # )
        rho, dtheta, prho, pdtheta= self._get_goal_pose_in_robot_frame(
            self.goal, self._robot_pose,self._robot_vel,None
        )
        n_scan,n_polar_convex,ndtheta,nrho,nrvx,nrvy = self.normalize_obs(scan,polar_convex_now,dtheta,rho,self._robot_vel.linear.x,self._robot_vel.linear.y)
        # obs_dict用于记录统计并且是reward计算需要的信息，mergerd用于环境返回
        # 测试一下看回调传回的速度是在机器人坐标系下的
        if self.last_done:
            self.last_done = False
            for i in range(self.stack_times):
                self.scan_que[i,:] = n_scan
                self.state_que[i,:] = np.array([ndtheta,nrho,nrvx,nrvy])
        else:
            self.scan_que = np.roll(self.scan_que, -1, axis=0)
            self.scan_que[-1, :] = n_scan
            self.state_que = np.roll(self.state_que, -1, axis=0)
            self.state_que[-1, :] = np.array([ndtheta,nrho,nrvx,nrvy])
        
        if convex_vertex_now is not None and len(polar_convex_now) == self.laser_num_beams:
            merged_obs = self.scan_que.flatten()
            merged_obs = np.append(merged_obs,self.state_que.flatten())
            # merged_obs = np.append(merged_obs,n_polar_convex)
        else:
            merged_obs = self.scan_que.flatten()
            merged_obs = np.append(merged_obs,self.state_que.flatten())
            # merged_obs = np.append(merged_obs,n_scan)
            convex_vertex_now = None
        
        obs_dict = {
            "laser_scan": scan,
            "goal_in_robot_frame": [rho, dtheta],
            "global_plan": self._globalplan,
            "robot_world_pose": self._robot_pose,
            "robot_world_vel": self._robot_vel.linear,
             "laser_convex":[convex_vertex_now,polar_convex_now,polar_convex_now_theta],
        }

        # self._laser_deque.clear()
        # self._rs_deque.clear()
        return merged_obs, obs_dict

    def get_observations(self, last_action_point_in_robot):
        # apply action time horizon
        # if self._is_train_mode:
        #     # 向前仿真到需要采取下个动作的周期
        #     self.call_service_takeSimStep(self._action_frequency)
        # else:
        #     try:
        #         rospy.wait_for_message(f"{self.ns_prefix}next_cycle", Bool)
        #     except Exception:
        #         pass
        
        if not self._ext_time_sync:
            self.wait_for_sensor_info()
            # try to retrieve sync'ed obs
            # laser_scan,scans_xy, robot_state = self.get_sync_obs()
            laser_scan,scans_xy, robot_state = None,None,None
            ##############
            # if len(self._rs_deque) > 0 and len(self._laser_deque) > 0:
            #     laser_scan,scans_xy = self.process_scan_msg(self._laser_deque[-1])
            #     robot_state= self.process_robot_state_msg(self._rs_deque[-1])
            
            if self.scan is not None and self.state is not None:
                laser_scan,scans_xy = self.process_scan_msg(self.scan)
                robot_state= self.process_robot_state_msg(self.state)
            ##############
            if laser_scan is not None and robot_state is not None:
                # print("Synced successfully")
                self._scan = laser_scan
                self._scan_xy_robot_frame = scans_xy
                self._robot_pose = robot_state[0]
                self._robot_vel = robot_state[1]
            self.scan = None
            self.state = None
            # else:
            #     print("Not synced")

        if len(self._scan.ranges) > 0:
            scan = self._scan.ranges.astype(np.float32)
        else:
            scan = np.zeros(self.laser_num_beams, dtype=float)

        # 输入xy逆时针 convex_vertex_now顶点顺时针 polar_convex_now散点顺时针
        
        convex_vertex_now,polar_convex_now,polar_convex_now_theta = galaxy_xyin_360out(self._scan_xy_robot_frame,radius = self.lidar_range)
        # convex_vertex_now,polar_convex_now,polar_convex_now_theta = None,None,None
        # rho, theta = ObservationCollector_d86._get_goal_pose_in_robot_frame(
        #     self.goal, self._robot_pose,self._robot_vel
        # )
        
        rho, dtheta, prho, pdtheta= self._get_goal_pose_in_robot_frame(
            self.goal, self._robot_pose,self._robot_vel,last_action_point_in_robot
        )
        n_scan,n_polar_convex,ndtheta,nrho,nrvx,nrvy = self.normalize_obs(scan,polar_convex_now,dtheta,rho,self._robot_vel.linear.x,self._robot_vel.linear.y)
        # obs_dict用于记录统计并且是reward计算需要的信息，mergerd用于环境返回
        # 测试一下看回调传回的速度是在机器人坐标系下的
        if convex_vertex_now is not None and len(polar_convex_now) == self.laser_num_beams:
            process_scan = n_polar_convex
        else:
            process_scan = n_scan

        if self.last_done:
            self.last_done = False
            for i in range(self.stack_times):
                self.scan_que[i,:] = process_scan
                self.state_que[i,:] = np.array([ndtheta,nrho,nrvx,nrvy])
        else:
            self.scan_que = np.roll(self.scan_que, -1, axis=0)
            self.scan_que[-1, :] = process_scan
            self.state_que = np.roll(self.state_que, -1, axis=0)
            self.state_que[-1, :] = np.array([ndtheta,nrho,nrvx,nrvy])
        
        merged_obs = self.scan_que.flatten()
        merged_obs = np.append(merged_obs,self.state_que.flatten())

        # if convex_vertex_now is not None and len(polar_convex_now) == self.laser_num_beams:
        #     merged_obs = self.scan_que.flatten()
        #     merged_obs = np.append(merged_obs,self.state_que.flatten())
        #     # merged_obs = np.append(merged_obs,n_polar_convex)
        # else:
        #     merged_obs = self.scan_que.flatten()
        #     merged_obs = np.append(merged_obs,self.state_que.flatten())
        #     # merged_obs = np.append(merged_obs,n_scan)
        #     convex_vertex_now = None
        
        obs_dict = {
            "laser_scan": scan,
            "goal_in_robot_frame": [rho, dtheta],
            "goal_in_robot_frame_ltrajp": [prho, pdtheta],
            "global_plan": self._globalplan,
            "robot_world_pose": self._robot_pose,
            "robot_world_vel": self._robot_vel.linear,
            "laser_convex":[convex_vertex_now,polar_convex_now,polar_convex_now_theta],
        }

        # self._laser_deque.clear()
        # self._rs_deque.clear()
        return merged_obs, obs_dict

    def normalize_obs(self,scan,polar_convex,theta,rho,wvx,wvy):
        # 全部归一化到0-1
        frame_dist = np.hypot(self.laser_origin_in_base[0],self.laser_origin_in_base[1])
        scan_min = - self.safe_dist
        scan_max = self.lidar_range + frame_dist - self.safe_dist
        n_scan = []
        n_polar_convex = []
        for i in range(len(scan)):
            scan_i = (scan[i] - scan_min)/(scan_max - scan_min)
            if scan_i > 1.:
                scan_i = 1.
            if scan_i < 0.:
                scan_i = 0.
            n_scan.append(scan_i)

        if polar_convex is not None and len(polar_convex) == self.laser_num_beams:
            for i in range(len(polar_convex)):
                scan_i = (polar_convex[i] - scan_min)/(scan_max - scan_min)
                if scan_i > 1.:
                    scan_i = 1.
                if scan_i < 0.:
                    scan_i = 0.
                n_polar_convex.append(scan_i)

        # ntheta = (theta + np.pi)/2./np.pi
        ntheta = theta/np.pi
        nrho = rho/40.
        # nwvx = (wvx + self.plan_dynamic_limit[0])/self.plan_dynamic_limit[0]/2.
        # nwvy = (wvy + self.plan_dynamic_limit[0])/self.plan_dynamic_limit[0]/2.
        nwvx = wvx / self.plan_dynamic_limit[0]
        nwvy = wvy / self.plan_dynamic_limit[0]

        return np.array(n_scan),np.array(n_polar_convex),ntheta,nrho,nwvx,nwvy

    def normalize_scan_obs(self,scan,polar_convex):
        # 全部归一化到0-1
        frame_dist = np.hypot(self.laser_origin_in_base[0],self.laser_origin_in_base[1])
        scan_min = - self.safe_dist
        scan_max = self.lidar_range + frame_dist - self.safe_dist
        n_scan = []
        n_polar_convex = []
        for i in range(len(scan)):
            n_scan.append((scan[i] - scan_min)/(scan_max - scan_min))
        if polar_convex is not None and len(polar_convex) == self.laser_num_beams:
            for i in range(len(polar_convex)):
                n_polar_convex.append((polar_convex[i] - scan_min)/(scan_max - scan_min))

        return np.array(n_scan),np.array(n_polar_convex)

    def normalize_state_obs(self,theta,rho,wvx,wvy):
        # 全部归一化到0-1

        ntheta = (theta + np.pi)/2./np.pi
        nrho = rho/50.
        nwvx = (wvx + self.plan_dynamic_limit[0])/self.plan_dynamic_limit[0]/2.
        nwvy = (wvy + self.plan_dynamic_limit[0])/self.plan_dynamic_limit[0]/2.
        return ntheta,nrho,nwvx,nwvy

    # @staticmethod
    @staticmethod
    def robotp2worldp(point_in_robort,robot_pose_in_world):
        rx,ry= point_in_robort
        dist = np.hypot(rx,ry)
        p_theta_world = robot_pose_in_world[2] + np.arctan2(ry,rx)
        return robot_pose_in_world[0] + dist*np.cos(p_theta_world) ,robot_pose_in_world[1] + dist*np.sin(p_theta_world)

    def _get_goal_pose_in_robot_frame(self,goal_pos: Pose2D, robot_pos: Pose2D,robot_vel:Twist,last_action_point_in_robot):
        y_relative = goal_pos.y - robot_pos.y
        x_relative = goal_pos.x - robot_pos.x
        rho = np.hypot(x_relative,y_relative)
        # theta = (
        #     np.arctan2(y_relative, x_relative) - robot_pos.theta + 4 * np.pi
        # ) % (2 * np.pi) - np.pi
        # 速度朝向  终点连线的角度差(世界坐标系下)
        # odom返回速度为世界坐标系下

        # v_heading = np.arctan2(robot_vel.linear.y,robot_vel.linear.x) + robot_pos.theta
        v_heading = np.arctan2(robot_vel.linear.y,robot_vel.linear.x)
        # print("robot pose use:",robot_pos.x,robot_pos.y)
        # if len(self._rs_deque) > 0 and len(self._laser_deque) > 0:
        #     laser_scan,scans_xy = self.process_scan_msg(self._laser_deque[-1])
        #     robot_state= self.process_robot_state_msg(self._rs_deque[-1])
        #     print("robot pose record new:",robot_state[0].x,robot_state[0].y)
        # print("dist:",rho)
        # theta = (
        #     v_heading - np.arctan2(y_relative, x_relative)  + 4 * np.pi
        # ) % (2 * np.pi)
        dtheta = NormalizeAngle(v_heading - np.arctan2(y_relative, x_relative))

        prho = None
        pdtheta = None
        if last_action_point_in_robot is not None:
            last_p = last_action_point_in_robot[-1]
            last_p_1 = last_action_point_in_robot[-2]
            last_p_w = ObservationCollector_d86.robotp2worldp(last_p,(robot_pos.x,robot_pos.y,robot_pos.theta))
            last_p_w_1 = ObservationCollector_d86.robotp2worldp(last_p_1,(robot_pos.x,robot_pos.y,robot_pos.theta))
            
            y_relative = goal_pos.y - last_p_w[1]
            x_relative = goal_pos.x - last_p_w[0]
            prho = np.hypot(x_relative,y_relative)

            v_heading = np.arctan2(last_p_w[1] - last_p_w_1[1],last_p_w[0] - last_p_w_1[0])
            pdtheta = NormalizeAngle(v_heading - np.arctan2(y_relative, x_relative))
        
        return rho, dtheta,prho,pdtheta

    def get_sync_obs(self):
        laser_scan = None
        robot_state = None
        scans_xy = None
        # print(f"laser deque: {len(self._laser_deque)}, robot state deque: {len(self._rs_deque)}")
        while len(self._rs_deque) > 0 and len(self._laser_deque) > 0:
            # 同步的是最老的雷达或状态信息，状态信息频率较快100hz 队尾的和最新的最多差0.1s
            # scan 5hz state 100hz ok
            laser_scan_msg = self._laser_deque.popleft()
            robot_state_msg = self._rs_deque.popleft()

            laser_stamp = laser_scan_msg.header.stamp.to_sec()
            robot_stamp = robot_state_msg.header.stamp.to_sec()

            while abs(laser_stamp - robot_stamp) > self._sync_slop:
                if laser_stamp > robot_stamp:
                    if len(self._rs_deque) == 0:
                        return laser_scan,scans_xy, robot_state
                    robot_state_msg = self._rs_deque.popleft()
                    robot_stamp = robot_state_msg.header.stamp.to_sec()
                else:
                    if len(self._laser_deque) == 0:
                        return laser_scan,scans_xy, robot_state
                    laser_scan_msg = self._laser_deque.popleft()
                    laser_stamp = laser_scan_msg.header.stamp.to_sec()

            laser_scan,scans_xy = self.process_scan_msg(laser_scan_msg)
            robot_state= self.process_robot_state_msg(robot_state_msg)

            if self._first_sync_obs:
                break

        # print(f"Laser_stamp: {laser_stamp}, Robot_stamp: {robot_stamp}")
        return laser_scan, scans_xy,robot_state



    def callback_odom_scan(self, scan, odom):
        self._scan,self._scan_xy_robot_frame = self.process_scan_msg(scan)
        self._robot_pose, self._robot_vel = self.process_robot_state_msg(odom)


    def callback_clock(self, msg_Clock):
        self._clock = msg_Clock.clock.to_sec()
        return

    def callback_subgoal(self, msg_Subgoal):
        self._subgoal = self.process_subgoal_msg(msg_Subgoal)
        return

    def callback_global_plan(self, msg_global_plan):
        # 不考虑随机静态和动态障碍物，只考虑地图因素
        # print("get:")
        # print(time.time())
        self._globalplan = ObservationCollector_d86.process_global_plan_msg(
            msg_global_plan
        )
        self.obs_dict["global_plan"] = self._globalplan
        return

    # def callback_scan(self, msg_laserscan):
    #     # append在右加新元素，pop left将左边的老元素出队
    #     if len(self._laser_deque) == self.max_deque_size:
    #         self._laser_deque.popleft()
    #     self._laser_deque.append(msg_laserscan)

    def callback_scan(self, msg_laserscan):
        self.new_scan_come = True
        # append在右加新元素，pop left将左边的老元素出队
        # if len(self._laser_deque) == self.max_deque_size:
        #     self._laser_deque.popleft()

        # self._laser_deque.append(msg_laserscan)
        self.scan = msg_laserscan
        # print("msg scan:")
        # print(self.scan)
        # if not self._ext_time_sync:
        #     # try to retrieve sync'ed obs
        #     # laser_scan,scans_xy, robot_state = self.get_sync_obs()

        #     laser_scan,scans_xy = self.process_scan_msg(msg_laserscan)

        #     self._scan = laser_scan
        #     self._scan_xy_robot_frame = scans_xy

        #     # else:
        #     #     print("Not synced")

        # if len(self._scan.ranges) > 0:
        #     scan = self._scan.ranges.astype(np.float32)
        # else:
        #     scan = np.zeros(self.laser_num_beams, dtype=float)

        # # 输入xy逆时针 convex_vertex_now顶点顺时针 polar_convex_now散点顺时针
        
        # self.convex_vertex_now,polar_convex_now,polar_convex_now_theta = galaxy_xyin_360out(self._scan_xy_robot_frame,radius = self.lidar_range)


        # self.n_res["n_scan"],self.n_res["n_polar_convex"] = self.normalize_scan_obs(scan,polar_convex_now)

        # # obs_dict用于记录统计并且是reward计算需要的信息，mergerd用于环境返回
        # # 测试一下看回调传回的速度是在机器人坐标系下的

        # self.obs_dict["laser_scan"] = scan
        # self.obs_dict["laser_convex"] = [self.convex_vertex_now,polar_convex_now,polar_convex_now_theta]


    def callback_robot_state(self, msg_robotstate):
        self.new_state_come = True
        # if len(self._rs_deque) == self.max_deque_size:
        #     self._rs_deque.popleft()
        
        # self._rs_deque.append(msg_robotstate)
        self.state = msg_robotstate
        # print("msg state:")
        # print(self.state)


        # 地图坐标系下的位置和速度
        # now = time.time()
        # pose2d = self.pose3D_to_pose2D(msg_robotstate.pose.pose)
        # twist = msg_robotstate.twist.twist
        # print(now," robot pose:",pose2d.x,pose2d.y)
       
        # print("robot vel:")
        # print(twist.linear.x,twist.linear.x)
        # self.last_time = now



        # if not self._ext_time_sync:
        #     robot_state= self.process_robot_state_msg(msg_robotstate)
        #     self._robot_pose = robot_state[0]
        #     self._robot_vel = robot_state[1]



        # # 输入xy逆时针 convex_vertex_now顶点顺时针 polar_convex_now散点顺时针
        
        # rho, theta = self._get_goal_pose_in_robot_frame(
        #     self.goal, self._robot_pose,self._robot_vel
        # )

        # self.n_res["ntheta"],self.n_res["nrho"],self.n_res["nrvx"],self.n_res["nrvy"] = self.normalize_state_obs(theta,rho,self._robot_vel.linear.x,self._robot_vel.linear.y)

        # # obs_dict用于记录统计并且是reward计算需要的信息，mergerd用于环境返回
        # # 测试一下看回调传回的速度是在机器人坐标系下的
        # self.obs_dict["goal_in_robot_frame"] = [rho, theta]
        # self.obs_dict["robot_world_pose"] = self._robot_pose
        # self.obs_dict["robot_world_vel"] = self._robot_vel.linear


        # pose2d1 = self.pose3D_to_pose2D(self._rs_deque[0].pose.pose)
        # pose2d2 = self.pose3D_to_pose2D(self._rs_deque[-1].pose.pose)
        # print("fl:",pose2d1.x,"  ",pose2d1.y," ",pose2d2.x,"  ",pose2d2.y)

    def callback_observation_received(
        self, msg_LaserScan, msg_RobotStateStamped
    ):
        # process sensor msg
        self._scan,self._scan_xy_robot_frame = self.process_scan_msg(msg_LaserScan)
        self._robot_pose, self._robot_vel = self.process_robot_state_msg(
            msg_RobotStateStamped
        )
        self.obs_received = True
        return

    def process_scan_msg(self, msg_LaserScan: LaserScan):
        self._scan_stamp = msg_LaserScan.header.stamp.to_sec()
        scan = np.array(msg_LaserScan.ranges)
        scan[np.isnan(scan)] = msg_LaserScan.range_max
        
        # laser2base
        scans_xy = []
        # laser坐标系下scan换为base坐标系下 并且scan变为到圆形机器人边缘的距离
        # scan本身是逆时针的
        for i in range(self.laser_num_beams):
            laser_frame_theta = i*self.drad
            base_frame_theta = laser_frame_theta + self.laser_origin_in_base[2]
            base_frame_x = scan[i]*np.cos(base_frame_theta) + self.laser_origin_in_base[0]
            base_frame_y = scan[i]*np.sin(base_frame_theta) + self.laser_origin_in_base[1]
            scan[i] = np.hypot(base_frame_x,base_frame_y) - self.safe_dist
            scans_xy.append((scan[i]*np.cos(base_frame_theta),scan[i]*np.sin(base_frame_theta)))

        msg_LaserScan.ranges = scan
        return msg_LaserScan,scans_xy

    def process_robot_state_msg(self, msg_Odometry):
        
        pose3d = msg_Odometry.pose.pose
        twist = msg_Odometry.twist.twist
        return self.pose3D_to_pose2D(pose3d), twist

    def process_pose_msg(self, msg_PoseWithCovarianceStamped):
        # remove Covariance
        pose_with_cov = msg_PoseWithCovarianceStamped.pose
        pose = pose_with_cov.pose
        return self.pose3D_to_pose2D(pose)

    def process_subgoal_msg(self, msg_Subgoal):
        return self.pose3D_to_pose2D(msg_Subgoal.pose)

    # @staticmethod
    # def process_global_plan_msg(globalplan):
    #     global_plan_2d = list(
    #         map(
    #             lambda p: ObservationCollector_d86.pose3D_to_pose2D(p.pose),
    #             globalplan.poses,
    #         )
    #     )

    #     return np.array(list(map(lambda p2d: [p2d.x, p2d.y], global_plan_2d)))
    @staticmethod
    def process_global_plan_msg(globalplan):
        x = []
        y = []
        for p in globalplan.poses:
            pose = ObservationCollector_d86.pose3D_to_pose2D(p.pose)
            if len(x) > 0 and math.hypot(pose.x - x[-1], pose.y - y[-1]) <= 1e-1:
                continue
            x.append(pose.x)
            y.append(pose.y)
    #     return np.array(list(map(lambda p2d: [p2d.x, p2d.y], global_plan_2d)))

        # 1 构造全局路径的样条 直接用三次样条吧 方便
        return Spline2D(x,y)

    @staticmethod
    def pose3D_to_pose2D(pose3d):
        pose2d = Pose2D()
        pose2d.x = pose3d.position.x
        pose2d.y = pose3d.position.y
        quaternion = (
            pose3d.orientation.x,
            pose3d.orientation.y,
            pose3d.orientation.z,
            pose3d.orientation.w,
        )
        euler = euler_from_quaternion(*quaternion)
        yaw = euler[2]
        pose2d.theta = yaw
        return pose2d

    @staticmethod
    def _stack_spaces(ss: Tuple[spaces.Box]):
        low = []
        high = []
        for space in ss:
            low.extend(space.low.tolist())
            high.extend(space.high.tolist())
        return spaces.Box(np.array(low).flatten(), np.array(high).flatten())


if __name__ == "__main__":

    rospy.init_node("states", anonymous=True)
    print("start")


    goal = Pose2D()
    goal.x = 10.
    goal.y = 10.
    goal.theta = 0.
    dynamic_space =  spaces.Box(
        low=np.array(
            [
                -2.,
                -2.,
               2.,
            ]
        ),
        high=np.array(
            [
                2.,
                2.,
               2.,
            ]
        ),
        dtype=np.float32,
    )
    state_collector = ObservationCollector_d86("sim1", 360, 10,0.25,goal,10.,dynamic_space)
    i = 0
    r = rospy.Rate(100)
    while i <= 100:
        i = i + 1
        obs = state_collector.get_observations()
        time.sleep(0.001)
