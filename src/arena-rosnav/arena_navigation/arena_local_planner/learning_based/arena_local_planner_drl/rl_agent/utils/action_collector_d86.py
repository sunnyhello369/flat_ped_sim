from copy import deepcopy

import matplotlib
import rospy
import sys,os
import numpy as np
import time  # for debuging

from geometry_msgs.msg import Twist,PolygonStamped,Point32,Point
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path,Odometry

# from tf.transformations import *
from rl_agent.utils.transformations import *
from gym import spaces

# services
from flatland_msgs.srv import StepWorld, StepWorldRequest
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose2D, PoseStamped, PoseWithCovarianceStamped

sys.path.append(r"/home/chen/desire_10086/flat_ped_sim/src/d86env")
from generate_ref_from_convex_polygon import quniform_one_bspline_optim
from SO2_mpc import SO2MPC
from cubic_bspline import quniform_clamped_bspline2D


VIS = True
class ActionCollector_d86:
    def __init__(self,
    ns: str,
    lidar_num:int,
    output_points_num:int,
    plan_dynamic_limit=(3.,3.,4.)):
        self.ns = ns
        if ns is None or not ns:
            self.ns_prefix = ""
        else:
            self.ns_prefix = "/" + ns + "/"
        self._is_train_mode = rospy.get_param("/train_mode")
        self._action_period = 1. / rospy.get_param("/robot_action_rate")
        self._plan_period = 1. / rospy.get_param("/robot_plan_rate")
        self.traj_time_span = rospy.get_param("/traj_time_span")

        self.construct_action_space(lidar_num,output_points_num)
        # action agent publisher
        if self._is_train_mode:
            self.agent_action_pub = rospy.Publisher(f"{self.ns_prefix}cmd_vel", Twist, queue_size=1)
        else:
            self.agent_action_pub = rospy.Publisher(f"{self.ns_prefix}cmd_vel_pub", Twist, queue_size=1)

        # self.action_library = {
        #     0: {"linear": 0.0, "angular": -self.w_max_},
        #     1: {"linear": self.v_max_, "angular": 0.0},
        #     2: {"linear": 0.0, "angular": self.w_max_},
        #     3: {"linear": self.v_max_, "angular": self.w_max_ / 2},
        #     4: {"linear": self.v_max_, "angular": -self.w_max_ / 2},
        #     5: {"linear": 0.0, "angular": 0.0},
        # }
        # self.N_DISCRETE_ACTIONS = len(self.action_library)
        # self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)
        
        self.planner =  quniform_one_bspline_optim(one_spline_ctrl_points_num=8,degrees=5,fix_end_pos_state=False
            ,v_limit=plan_dynamic_limit[0]
            ,a_limit=plan_dynamic_limit[1]
            ,j_limit=plan_dynamic_limit[2])
        
        # mpc_state_num = int(5*self._plan_period/self._action_period)
        # mpc_state_num = 30
        mpc_state_num = 4
        # self.controller = SO2MPC(N= mpc_state_num,control_period=self._action_period
        #     ,v_limit=plan_dynamic_limit[0]
        #     ,a_limit=plan_dynamic_limit[1]
        #     ,j_limit=plan_dynamic_limit[2])   
        self.controller = SO2MPC(N= mpc_state_num,control_period=self._action_period
            ,v_limit=plan_dynamic_limit[0]
            ,a_limit=plan_dynamic_limit[1]
            # ,a_limit=plan_dynamic_limit[1]*4.
            ,j_limit=plan_dynamic_limit[2]*2000.)   


        self.traj_spline_world = None
        self.need_bark = False
        self.exc_time = 0.
        self.goal_pos_world = None
        self.start_state_world = None
        self.start_pos = None
        self.msg_Odometry  = None   
        self.msg_Odometry_last  = None   
        self.new_state_come = False

        # service clients
        if self._is_train_mode:
            self._service_name_step = f"{self.ns_prefix}step_world"
            self._sim_step_client = rospy.ServiceProxy(
                self._service_name_step, StepWorld,persistent=True
            )
        
        self._next_step_sub = rospy.Subscriber(
            f"{self.ns_prefix}next_step_signal",
            Bool,
            self.callback_next_step,
            tcp_nodelay=True,
        )
        self.step_finish = False

        self._robot_state_sub = rospy.Subscriber(
            f"{self.ns_prefix}odom",
            Odometry,
            self.callback_robot_state,
            tcp_nodelay=True,
        )

        if VIS:
            self.marker_net_pub = rospy.Publisher(f"{self.ns_prefix}ctrl_network_points", Marker, queue_size=10)
            self.marker_opt_pub = rospy.Publisher(f"{self.ns_prefix}ctrl_opt_points", Marker, queue_size=10)
            self.poly_pub = rospy.Publisher(f"{self.ns_prefix}convex_polygon", PolygonStamped, queue_size=10)
            self.path_pub = rospy.Publisher(f"{self.ns_prefix}drl_bspl_path", Path, queue_size=10)
            self.mpc_pub = rospy.Publisher(f"{self.ns_prefix}mpc_path", Path, queue_size=10)

    def construct_action_space(self,lidar_num,output_points_num):
        action_space = []

        action_space.extend([
            spaces.Box(
                    low=-1.,
                    high=1.,
                    shape=(output_points_num,),
                    dtype=np.float32,
                ),
                spaces.Box(
                    low=-1.,
                    high=1.,
                    shape=(output_points_num,),
                    dtype=np.float32,  # dist
                ),                
        ])
        low = []
        high = []
        for space in action_space:
            low.extend(space.low.tolist())
            high.extend(space.high.tolist())
        self.output_points_num = output_points_num
        self.action_space = spaces.Box(np.array(low).flatten(), np.array(high).flatten())
    
    def get_action_space(self):
        return self.action_space

    def wait_for_sensor_info(self):
        timeout = 20
        for i in range(timeout):
            if self.new_state_come:
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

    def callback_robot_state(self,msg_Odometry):
        self.new_state_come = True
        self.msg_Odometry_last = self.msg_Odometry
        self.msg_Odometry = msg_Odometry

    def callback_next_step(self,next_step_signal):
        self.step_finish = next_step_signal.data

    def plan_step(self,action_points,laser_convex,robot_pose):
        # 因为是仿真中可以将规划计算和规划执行写成下面这种串行的形式，但实际中是并行关系
        # 实际 在本帧中使用上帧(当前时刻+规划周期)时刻状态作为规划起点的轨迹规划计算中，跟踪的是上帧轨迹

        # reset后的第一帧，因为不存在上帧轨迹选择bark不动作
        if not self.need_bark:
            # 执行轨迹是在上帧robot_pose坐标系下规划得到的
            if VIS:
                self.vis_plan_path()
            self.control_exce_multi_step()
        else:
            self.need_bark = False
            self.bark()

        # action_points [(x1,y1),..,(x5,y5)]
        # 1 判断终点是否在多边形内,并且直线距离小于，如果在则直接将终态设为全局终态
        # robot_pose为规划坐标系原点
        goal_in_robot_frame = ActionCollector_d86.worldp2robotp((self.goal_pos_world[0],self.goal_pos_world[1]),(robot_pose.x,robot_pose.y,robot_pose.theta))
        start_state_robot = ActionCollector_d86.worlds2robots(self.start_state_world,(robot_pose.x,robot_pose.y,robot_pose.theta))

        if len(action_points)!=0:
            if ActionCollector_d86.is_in_convex(goal_in_robot_frame,laser_convex):
                action_points.pop()
                action_points.append(goal_in_robot_frame)
                end = [goal_in_robot_frame,(0.,0.),(0.,0.),(0.,0.)]
                success,spline,track_cost,ctrl_points = self.planner.formulate(self.traj_time_span,laser_convex,start_state_robot,end = end,ref_ctrl_points=action_points)

            # # action_points，以及多边形顶点不在世界坐标系下(其y轴方向与第一束激光方向一致)，需要转到世界坐标下再用于规划。start_state为世界坐标系下状态
            # # 假设第一束激光方向
            # # 坐标系需要再确认，pub 消息linear x,y指的是小车坐标系吗? 以及obs中的回调机器人速度是小车坐标系下，还是世界坐标系
            # # 2 规划
            # # 先判断初态是否在约束内,如果初态不在当前约束内做紧急刹车

            # # 规划是否成功,规划轨迹
            else:
                success,spline,track_cost,ctrl_points = self.planner.formulate(self.traj_time_span,laser_convex,start_state_robot,ref_ctrl_points=action_points)
        else:
            success,spline,track_cost,ctrl_points = False,None,None,None
        
        if VIS:
            self.vis_pub_action(laser_convex,action_points,robot_pose)

        if success:
            self.exc_time = 0.
            ctrlx = []
            ctrly = []
            for i in range(len(spline.sx.ctrl_points)):  
                x,y = ActionCollector_d86.robotp2worldp((spline.sx.ctrl_points[i],spline.sy.ctrl_points[i]),(robot_pose.x,robot_pose.y,robot_pose.theta))
                ctrlx.append(x)
                ctrly.append(y)
            
            self.traj_spline_world = quniform_clamped_bspline2D(ctrlx,ctrly,spline.sx.p,spline.sx.T)
            # if VIS:
            #     with open("/home/tou/desire_10086/flat_ped_sim/src/ctrl.txt", "a+") as path_file:
            #         # for x in self.traj_spline_world.sx.ctrl_points:
            #         #     path_file.write(str(x))
            #         #     path_file.write("\t")
            #         # for i in range(len(self.traj_spline_world.sy.ctrl_points)):
            #         #     path_file.write(str(self.traj_spline_world.sy.ctrl_points[i]))
            #         #     if i == len(self.traj_spline_world.sy.ctrl_points) - 1:
            #         #         path_file.write("\n")
            #         #     else:
            #         #         path_file.write("\t")
            #         for x in spline.sx.ctrl_points:
            #             path_file.write(str(x))
            #             path_file.write("\t")
            #         for i in range(len(spline.sy.ctrl_points)):
            #             path_file.write(str(spline.sy.ctrl_points[i]))
            #             if i == len(spline.sy.ctrl_points) - 1:
            #                 path_file.write("\n")
            #             else:
            #                 path_file.write("\t")
        else:
            if VIS:
                print("fail 1!!!!!!!!!!!!!!!!!!!!!")
                print("fail 1!!!!!!!!!!!!!!!!!!!!!")
                print("fail 1!!!!!!!!!!!!!!!!!!!!!")
                print("fail 1!!!!!!!!!!!!!!!!!!!!!")
                print("fail 1!!!!!!!!!!!!!!!!!!!!!")
                print("fail 1!!!!!!!!!!!!!!!!!!!!!")
                print("fail 1!!!!!!!!!!!!!!!!!!!!!")
                print("fail 1!!!!!!!!!!!!!!!!!!!!!")
                print("fail 1!!!!!!!!!!!!!!!!!!!!!")
                print("fail 1!!!!!!!!!!!!!!!!!!!!!")
                print("fail 1!!!!!!!!!!!!!!!!!!!!!")

            # 执行上次的轨迹 前提有轨迹可执行
            if self.traj_spline_world is not None and self.exc_time + self._plan_period < self.traj_time_span:
                # self.vis_plan_path(robot_pose)
                pass
            elif len(action_points)==0:
                self.need_bark = True
            else:
                success2,spline,track_cost2,ctrl_points = self.planner.formulate(self.traj_time_span,laser_convex,start_state_robot)
                if not success2:
                    if VIS:
                        print("plan convex stop trajfail")
                        print("fail 2!!!!!!!!!!!!!!!!!!!!!")
                        print("fail 2!!!!!!!!!!!!!!!!!!!!!")
                        print("fail 2!!!!!!!!!!!!!!!!!!!!!")
                        print("fail 2!!!!!!!!!!!!!!!!!!!!!")
                        print("fail 2!!!!!!!!!!!!!!!!!!!!!")
                        print("fail 2!!!!!!!!!!!!!!!!!!!!!")
                        print("fail 2!!!!!!!!!!!!!!!!!!!!!")
                        print("fail 2!!!!!!!!!!!!!!!!!!!!!")
                        print("fail 2!!!!!!!!!!!!!!!!!!!!!")
                        print("fail 2!!!!!!!!!!!!!!!!!!!!!")
                        print("fail 2!!!!!!!!!!!!!!!!!!!!!")
                    success3,spline,track_cost3,ctrl_points = self.planner.formulate(self.traj_time_span,None,start_state_robot)
                    if not success3:
                        if VIS:
                            print("plan no convex stop trajfail")
                            print("fail 3!!!!!!!!!!!!!!!!!!!!!")
                            print("fail 3!!!!!!!!!!!!!!!!!!!!!")
                            print("fail 3!!!!!!!!!!!!!!!!!!!!!")
                            print("fail 3!!!!!!!!!!!!!!!!!!!!!")
                            print("fail 3!!!!!!!!!!!!!!!!!!!!!")
                            print("fail 3!!!!!!!!!!!!!!!!!!!!!")
                            print("fail 3!!!!!!!!!!!!!!!!!!!!!")
                            print("fail 3!!!!!!!!!!!!!!!!!!!!!")
                            print("fail 3!!!!!!!!!!!!!!!!!!!!!")
                            print("fail 3!!!!!!!!!!!!!!!!!!!!!")
                            print("fail 3!!!!!!!!!!!!!!!!!!!!!")
                        self.need_bark = True

                if spline is not None:
                    self.exc_time = 0.
                    ctrlx = []
                    ctrly = []
                    for i in range(len(spline.sx.ctrl_points)):
                        x,y = ActionCollector_d86.robotp2worldp((spline.sx.ctrl_points[i],spline.sy.ctrl_points[i]),(robot_pose.x,robot_pose.y,robot_pose.theta))
                        ctrlx.append(x)
                        ctrly.append(y)
                    
                    self.traj_spline_world = quniform_clamped_bspline2D(ctrlx,ctrly,spline.sx.p,spline.sx.T)

        return success,track_cost,ctrl_points

    # def control_exce_multi_step(self,robot_pose):
    #     action_msg = Twist()

    #     # # 这里偷懒了 本来应该按控制周期来求解MPC每50ms求解一次取u[0]作为控制量。
    #     # # 目前mpc求解速度太慢，只求解一次，取u序列在对应时刻发布
    #     # if not success:
    #     #     return 

    #     # for v in zip(vx_opt,vy_opt):

    #     # 下帧规划起点状态为本帧轨迹起始时间exc_time+_plan_period
    #     print("odom:")
    #     state = self.get_now_ws_from_odom()
    #     print(state)

    #     # 跟踪效果太差，重新选择规划起点
    #     if np.hypot(self.start_state_world[0][0] - state[0][0],self.start_state_world[0][1] - state[0][1]) > 0.2:
    #         print("reloc!")
    #         # 运动学模型递推_plan_period
    #         self.start_state_world = ActionCollector_d86.intergral_2d_model(state[0],state[1],state[2],self.traj_time_span)
    #         # self.start_state_world = state
    #     else:
    #         self.start_state_world = [(self.traj_spline_world.calc_position_u(self.exc_time+self.traj_time_span)),(self.traj_spline_world.calcd(self.exc_time+self.traj_time_span)),(self.traj_spline_world.calcdd(self.exc_time+self.traj_time_span))]        
    #     print("traj:")
    #     print(self.start_state_world[0],self.start_state_world[1])

    #     # 在本帧规划计算中，跟踪的是上帧轨迹
    #     exce_time_thisloop = int(self._plan_period/self._action_period)
    #     exce_time_thisloop = int(self.traj_time_span/self._action_period)

        
    #     # traj = ActionCollector_d86.calc_traj_info(self.traj_spline_world, self.exc_time,self._action_period,self.controller.N)
    #     # print("track info:")
    #     # print(traj)
    #     # now_state = self.get_now_ws_from_odom()
    #     # print("now state:")
    #     # print(now_state)
    #     # success,x_opt,y_opt,vx_opt,vy_opt,ax_opt,ay_opt,jx_opt,jy_opt  = self.controller.formulate(now_state,traj)
     
    #     for i in range(exce_time_thisloop):
    #     # for i in range(self.controller.N):
    #         # 没有考虑位置误差的直接跟踪

    #         # 将轨迹转到自车坐标系下
                    
    #         traj = ActionCollector_d86.calc_traj_info(self.traj_spline_world, self.exc_time,self._action_period,self.controller.N)
    #         print("***************************")
    #         print("track info:")
    #         print(traj)
    #         now_state = self.get_now_ws_from_odom()
    #         print("now state:")
    #         print(now_state)
    #         success,x_opt,y_opt,vx_opt,vy_opt,ax_opt,ay_opt,jx_opt,jy_opt  = self.controller.formulate(now_state,traj)
            
    #         self.exc_time += self._action_period
            
    #         # 起步阶段不能使用mpc
    #         # 参考轨迹段起点在当前机器人位置后面，取u[0]会使机器人后退
    #         if not success :#  or i < 10
    #             print("control fail")
    #             wvx,wvy = self.traj_spline_world.calcd(self.exc_time)
    #         else:
    #             if VIS:
    #                 plan_path = Path()
    #                 plan_path.header.stamp = rospy.Time.now()
    #                 plan_path.header.frame_id = 'map'
    #                 # plan_path.header.frame_id = "sim_1_base_footprint"

                    
    #                 for p in zip(x_opt,y_opt):
    #                     pose = PoseStamped()
    #                     pose.pose.position.x = p[0]
    #                     pose.pose.position.y = p[1]
    #                     pose.pose.position.z = 0.
    #                     plan_path.poses.append(pose)
    #                 self.mpc_pub.publish(plan_path)

    #                 print("mpc:")
    #                 print([[(s[0],s[1]),(s[2],s[3]),(s[4],s[5])] for s in zip(x_opt,y_opt,vx_opt,vy_opt,ax_opt,ay_opt)])
    #                 print("cmd wv")
    #                 print(vx_opt[0],vy_opt[0])

    #             wvx,wvy = vx_opt[0],vy_opt[0]
    #             # wvx,wvy = vx_opt[i],vy_opt[i]

    #         # self.exc_time += self._action_period
    #         # wvx,wvy = self.traj_spline_world.calcd(self.exc_time)

    #         # 轨迹是世界坐标系下速度，但是发布的是自车坐标系下速度
    #         # 将下个控制周期的期望速度转到当前机器人坐标系下
    #         # now_yaw = self.traj_spline_world.calc_yaw_u(self.exc_time) # 当前世界坐标系下yaw角
    #         wx,wy,now_yaw = self.get_now_wpos_from_odom()
    #         rvx,rvy = ActionCollector_d86.worldva2robotva(wvx,wvy,now_yaw)
    #         print("cmd rv")
    #         print(rvx,rvy)
    #         print("***************************")

    #         action_msg.linear.x = rvx
    #         action_msg.linear.y = rvy
    #         action_msg.angular.z = 0.
    #         self.agent_action_pub.publish(action_msg)
            
    #         if self._is_train_mode:
    #             # 向前仿真到需要采取下个动作的周期
    #             self.call_service_takeSimStep(self._action_period)
    #         else:
    #             try:
    #                 rospy.wait_for_message(f"{self.ns_prefix}next_cycle", Bool)
    #             except Exception:
    #                 pass
        
    #     state = self.get_now_ws_from_odom()
    #     # 跟踪效果太差，重新选择规划起点
    #     if np.hypot(self.start_state_world[0][0] - state[0][0],self.start_state_world[0][1] - state[0][1]) > 0.2:
    #         # 运动学模型递推_plan_period
    #         # self.start_state_world = ActionCollector_d86.intergral_2d_model(state[0],state[1],state[2],self.traj_time_span)
    #         print("reloc!")
    #         self.start_state_world = state
    #     else:
    #         self.start_state_world = [(self.traj_spline_world.calc_position_u(self.exc_time)),(self.traj_spline_world.calcd(self.exc_time)),(self.traj_spline_world.calcdd(self.exc_time))]        



    def control_exce_multi_step(self):
        action_msg = Twist()

        # # 这里偷懒了 本来应该按控制周期来求解MPC每50ms求解一次取u[0]作为控制量。
        # 下帧规划起点状态为本帧轨迹起始时间exc_time+_plan_period
        state = self.get_now_ws_from_odom()

        # 跟踪效果太差，重新选择规划起点

        # if np.hypot(self.start_state_world[0][0] - state[0][0],self.start_state_world[0][1] - state[0][1]) > 0.2:
        #     if VIS:
        #         print("reloc!")
        #     # 运动学模型递推_plan_period
        #     self.start_state_world = ActionCollector_d86.intergral_2d_model(state[0],state[1],state[2],self._plan_period)
        # else:
        #     self.start_state_world = [(self.traj_spline_world.calc_position_u(self.exc_time+self._plan_period)),(self.traj_spline_world.calcd(self.exc_time+self._plan_period)),(self.traj_spline_world.calcdd(self.exc_time+self._plan_period))]        
        #     # self.start_state_world = [(self.traj_spline_world.calc_position_u(self.exc_time+self.traj_time_span)),(self.traj_spline_world.calcd(self.exc_time+self.traj_time_span)),(self.traj_spline_world.calcdd(self.exc_time+self.traj_time_span))]        

        # if VIS:
        #     print("odom:")
        #     print(state)
        #     print("traj:")
        #     print(self.start_state_world[0],self.start_state_world[1])

        # 在本帧规划计算中，跟踪的是上帧轨迹
        exce_time_thisloop = int(self._plan_period/self._action_period)
        # exce_time_thisloop = int(self.traj_time_span/self._action_period)
        # exce_time_thisloop = int(10./self._action_period)

        for i in range(exce_time_thisloop):
        # for i in range(self.controller.N):
            # 没有考虑位置误差的直接跟踪

            # 将轨迹转到自车坐标系下
                    
            w_traj_state = ActionCollector_d86.calc_traj_info(self.traj_spline_world, self.exc_time,self._action_period,self.controller.N)
            # w_traj_state = ActionCollector_d86.calc_traj_info(self.traj_spline_world, self.exc_time,self._action_period,30)

            now_state = self.get_now_ws_from_odom()
            wx,wy,now_yaw = self.get_now_wpos_from_odom()

            vx,vy = ActionCollector_d86.worldva2robotva(now_state[1][0],now_state[1][1],now_yaw)
            ax,ay = ActionCollector_d86.worldva2robotva(now_state[2][0],now_state[2][1],now_yaw)
            robot_now_state = [(0.,0.),(vx,vy),(ax,ay)]
            r_traj_state = []
            for s in w_traj_state:
                r_traj_state.append(ActionCollector_d86.worlds2robots(s,(wx,wy,now_yaw)))

            # if VIS:
            #     print("now state:")
            #     print(now_state)
            #     print("***************************")
            #     print("track info:")
            #     print(r_traj_state)
            success,x_opt,y_opt,vx_opt,vy_opt,ax_opt,ay_opt,jx_opt,jy_opt  = self.controller.formulate(robot_now_state,r_traj_state)

            self.exc_time += self._action_period
            
            if not success :#  or i < 10
                if VIS:
                    print("control fail")
                # wvx,wvy = self.traj_spline_world.calcd(self.exc_time)
                # wx,wy,now_yaw = self.get_now_wpos_from_odom()
                # rvx,rvy = ActionCollector_d86.worldva2robotva(wvx,wvy,now_yaw)

                wax,way = self.traj_spline_world.calcdd(self.exc_time)
                wx,wy,now_yaw = self.get_now_wpos_from_odom()
                # rax,ray = ActionCollector_d86.worldva2robotva(wax,way,now_yaw)
            else:
                if VIS:
                    plan_path = Path()
                    plan_path.header.stamp = rospy.Time.now()
                    plan_path.header.frame_id = 'map'
                    # plan_path.header.frame_id = "sim_1_base_footprint"

                    
                    for p in zip(x_opt,y_opt):
                        wp = ActionCollector_d86.robotp2worldp(p,(wx,wy,now_yaw))
                        pose = PoseStamped()
                        pose.pose.position.x = wp[0]
                        pose.pose.position.y = wp[1]
                        pose.pose.position.z = 0.
                        plan_path.poses.append(pose)
                    self.mpc_pub.publish(plan_path)

                    print("rmpc:")
                    print([[(s[0],s[1]),(s[2],s[3]),(s[4],s[5])] for s in zip(x_opt,y_opt,vx_opt,vy_opt,ax_opt,ay_opt)])
                    wx_opt = []
                    wy_opt = []
                    wvx_opt = []
                    wvy_opt = []
                    wax_opt = []
                    way_opt = []
                    print("wmpc:")
                    for p in zip(x_opt,y_opt):
                        ox,oy  = ActionCollector_d86.robotp2worldp(p,(wx,wy,now_yaw))
                        wx_opt.append(ox)
                        wy_opt.append(oy)
                    for v in zip(vx_opt,vy_opt):
                        ovx,ovy  = ActionCollector_d86.robotva2worldva(v[0],v[1],now_yaw)
                        wvx_opt.append(ovx)
                        wvy_opt.append(ovy)
                    for a in zip(ax_opt,ay_opt):
                        oax,oay  = ActionCollector_d86.robotva2worldva(a[0],a[1],now_yaw)
                        wax_opt.append(oax)
                        way_opt.append(oay)
                    print([[(s[0],s[1]),(s[2],s[3]),(s[4],s[5])] for s in zip(wx_opt,wy_opt,wvx_opt,wvy_opt,wax_opt,way_opt)])

                rax,ray = ax_opt[0],ay_opt[0]
                wx,wy,now_yaw = self.get_now_wpos_from_odom()
                wax,way = ActionCollector_d86.robotva2worldva(rax,ray,now_yaw)
                # rvx,rvy = vx_opt[0],vy_opt[0]

            # self.exc_time += self._action_period
            # wvx,wvy = self.traj_spline_world.calcd(self.exc_time)

            # 轨迹是世界坐标系下速度，但是发布的是自车坐标系下速度
            # 将下个控制周期的期望速度转到当前机器人坐标系下
            # now_yaw = self.traj_spline_world.calc_yaw_u(self.exc_time) # 当前世界坐标系下yaw角

            if False and VIS:
                print("cmd wa")
                print(wax,way)
                print("***************************")

            action_msg.linear.x = wax
            action_msg.linear.y = way
            action_msg.angular.z = 0.
            # 因为我们的控制只到v这层，控不了速度和加速度。可以从odom反馈看到返回的加速度都是0，导致
            # 达不到我们的轨迹状态要求，没办法。
            self.call_msg_takeSimStep(action_msg,self._action_period)
            self.wait_for_sensor_info()

        now_state = self.get_now_ws_from_odom()
        self.start_state_world = now_state

        # state = self.get_now_ws_from_odom()
        # # 跟踪效果太差，重新选择规划起点
        # if np.hypot(self.start_state_world[0][0] - state[0][0],self.start_state_world[0][1] - state[0][1]) > 0.2:
        #     # 运动学模型递推_plan_period
        #     # self.start_state_world = ActionCollector_d86.intergral_2d_model(state[0],state[1],state[2],self.traj_time_span)
        #     print("reloc!")
        #     self.start_state_world = state
        # else:
        #     self.start_state_world = [(self.traj_spline_world.calc_position_u(self.exc_time)),(self.traj_spline_world.calcd(self.exc_time)),(self.traj_spline_world.calcdd(self.exc_time))]        


    def bark(self):
        # action_msg = Twist()
        # action_msg.linear.x = 0.
        # action_msg.linear.y = 0.
        # action_msg.angular.z = 0.
        
        # self.agent_action_pub.publish(action_msg)

        # if self._is_train_mode:
        #     # 向前仿真到需要采取下个动作的周期
        #     self.call_service_takeSimStep(self._plan_period)
        # else:
        #     try:
        #         rospy.wait_for_message(f"{self.ns_prefix}next_cycle", Bool)
        #     except Exception:
        #         pass
        self.call_msg_takeSimStep(Twist(),self._plan_period)
        self.wait_for_sensor_info()
        self.start_state_world  = self.get_now_ws_from_odom()


    def get_now_wpos_from_odom(self):
        odom = deepcopy(self.msg_Odometry)
        if odom is not None:
            quaternion = (
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
                odom.pose.pose.orientation.w,
            )
            euler = euler_from_quaternion(*quaternion)
            return odom.pose.pose.position.x,odom.pose.pose.position.y,euler[2]
        return self.start_pos.x,self.start_pos.y,self.start_pos.theta

    def get_now_ws_from_odom(self):
        odom_last = deepcopy(self.msg_Odometry_last)
        odom = deepcopy(self.msg_Odometry)
        if odom is not None:
            quaternion = (
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
                odom.pose.pose.orientation.w,
            )
            # euler = euler_from_quaternion(*quaternion)

            # 里程回调的速度为世界坐标系下速度！！！！！
            vx = odom.twist.twist.linear.x
            vy = odom.twist.twist.linear.y
            # 修改了仿真环境的返回，借用这个结构体返回加速度
            ax = odom.twist.twist.angular.x
            ay = odom.twist.twist.angular.y
            # wvx,wvy = ActionCollector_d86.robotva2worldva(vx,vy,euler[2])

            return [(odom.pose.pose.position.x,odom.pose.pose.position.y),(vx,vy),(ax,ay)]
        return self.start_state_world

    @staticmethod
    def intergral_2d_model(p,v,a,dt): 
        npx = p[0] +v[0]*dt + a[0]*dt**2/2.
        npy = p[1] +v[1]*dt + a[1]*dt**2/2.
        nvx = v[0] + a[0]*dt
        nvy = v[1] + a[1]*dt
        return [(npx,npy),(nvx,nvy),(a[0],a[1])]

    @staticmethod
    def calc_traj_info(traj,now_time,control_period,state_num):
        traj_info = []
        # now = deepcopy(now_time)
        x,y = traj.calc_position_u(now_time)
        vx,vy = traj.calcd(now_time)
        ax,ay = traj.calcdd(now_time)
        for i in range(state_num):
            now_time += control_period
            if now_time > traj.sx.T:
                traj_info.append([(x,y),(vx,vy),(ax,ay)])
            else:
                x,y = traj.calc_position_u(now_time)
                vx,vy = traj.calcd(now_time)
                ax,ay = traj.calcdd(now_time)
                traj_info.append([(x,y),(vx,vy),(ax,ay)])
        # print("*********")
        # print(traj_info)
        # print("*********")
        return traj_info

    def vis_pub_action(self,convex_polygon,action_points,robot_pose):
        convex_p = PolygonStamped()
        convex_p.header.stamp = rospy.Time.now()
        convex_p.header.frame_id = 'map'
        # convex_p.header.frame_id = "sim_1_base_footprint"

        for p in convex_polygon:
            wx,wy = ActionCollector_d86.robotp2worldp(p,(robot_pose.x,robot_pose.y,robot_pose.theta))
            p32 = Point(wx,wy,0.)
            # p32 = Point(p[0],p[1],0.)
            convex_p.polygon.points.append(p32)
        # 最后一个点重复
        # convex_p.polygon.points.pop()
        self.poly_pub.publish(convex_p)

        network_output_ctrlp = Marker()
        network_output_ctrlp.header.stamp = rospy.Time.now()
        network_output_ctrlp.header.frame_id = 'map'
        # network_output_ctrlp.header.frame_id = "sim_1_base_footprint"
        network_output_ctrlp.pose.orientation.w = 1.0
        network_output_ctrlp.action = Marker.ADD
        # line
        network_output_ctrlp.id = 0
        network_output_ctrlp.type = Marker.LINE_STRIP
        network_output_ctrlp.color.a = 0.5
        network_output_ctrlp.scale.x = 0.1
        network_output_ctrlp.color.b = 1.0



        tp1 = self.traj_time_span*(self.planner.u[self.planner.p + 1] - self.planner.u[1])/self.planner.p
        tp2 = self.traj_time_span*(self.planner.u[self.planner.p + 2] - self.planner.u[2])/self.planner.p
        tv1 =  self.traj_time_span*(self.planner.u[self.planner.p + 1] - self.planner.u[2])/(self.planner.p-1.)        
        p32 = Point(self.start_state_world[0][0],self.start_state_world[0][1],0.)

        network_output_ctrlp.points.append(p32)
        p32 = Point(self.start_state_world[1][0]*tp1 +network_output_ctrlp.points[0].x,self.start_state_world[1][1]*tp1 +network_output_ctrlp.points[0].y,0.)
        network_output_ctrlp.points.append(p32)
        p32 = Point((self.start_state_world[2][0]*tv1+self.start_state_world[1][0])*tp2+network_output_ctrlp.points[1].x,(self.start_state_world[2][1]*tv1+self.start_state_world[1][1])*tp2+network_output_ctrlp.points[1].y,0.)
        network_output_ctrlp.points.append(p32)

        for p in action_points:
            wx,wy = ActionCollector_d86.robotp2worldp(p,(robot_pose.x,robot_pose.y,robot_pose.theta))
            p32 = Point(wx,wy,0.)
            # p32 = Point(p[0],p[1],0.)
            network_output_ctrlp.points.append(p32)
        self.marker_net_pub.publish(network_output_ctrlp)

        # points
        network_output_ctrlp.id = 1
        network_output_ctrlp.type = Marker.POINTS
        network_output_ctrlp.color.a = 0.5
        network_output_ctrlp.scale.x = 0.2
        network_output_ctrlp.scale.y = 0.2
        network_output_ctrlp.color.g = 1.0
        self.marker_net_pub.publish(network_output_ctrlp)

    def vis_plan_path(self):
        plan_path = Path()
        plan_path.header.stamp = rospy.Time.now()
        plan_path.header.frame_id = 'map'
        # plan_path.header.frame_id = "sim_1_base_footprint"

        
        for i in range(51):
            rx,ry = self.traj_spline_world.calc_position_u(i*self.traj_time_span/50.)
            # wx,wy = ActionCollector_d86.robotp2worldp((rx,ry),(robot_pose.x,robot_pose.y,robot_pose.theta))
            pose = PoseStamped()
            pose.pose.position.x = rx
            pose.pose.position.y = ry
            pose.pose.position.z = 0.
            plan_path.poses.append(pose)
        self.path_pub.publish(plan_path)

        network_output_ctrlp = Marker()
        network_output_ctrlp.header.stamp = rospy.Time.now()
        network_output_ctrlp.header.frame_id = 'map'
        # network_output_ctrlp.header.frame_id = "sim_1_base_footprint"
        network_output_ctrlp.pose.orientation.w = 1.0
        network_output_ctrlp.action = Marker.ADD
        # line
        network_output_ctrlp.id = 0
        network_output_ctrlp.type = Marker.LINE_STRIP
        network_output_ctrlp.color.a = 1.0
        network_output_ctrlp.scale.x = 0.1
        network_output_ctrlp.color.b = 1.0

        for p in zip(self.traj_spline_world.sx.ctrl_points,self.traj_spline_world.sy.ctrl_points):
            # wx,wy = ActionCollector_d86.robotp2worldp(p,(robot_pose.x,robot_pose.y,robot_pose.theta))
            # p32 = Point(wx,wy,0.)
            p32 = Point(p[0],p[1],0.)
            network_output_ctrlp.points.append(p32)
        self.marker_opt_pub.publish(network_output_ctrlp)

        # points
        network_output_ctrlp.id = 1
        network_output_ctrlp.type = Marker.POINTS
        network_output_ctrlp.color.a = 1.0
        network_output_ctrlp.scale.x = 0.2
        network_output_ctrlp.scale.y = 0.2
        network_output_ctrlp.color.g = 1.0
        self.marker_opt_pub.publish(network_output_ctrlp)

    @staticmethod
    def worlds2robots(ws,rs_in_world):
        p,v,a = ws[0],ws[1],ws[2]
        rx,ry = ActionCollector_d86.worldp2robotp(p,rs_in_world)
        rvx,rvy = ActionCollector_d86.worldva2robotva(v[0],v[1],rs_in_world[2])
        rax,ray = ActionCollector_d86.worldva2robotva(a[0],a[1],rs_in_world[2])
        return [(rx,ry),(rvx,rvy),(rax,ray)]

    @staticmethod
    def robots2worlds(rs,rs_in_world):
        p,v,a = rs[0],rs[1],rs[2]
        wx,wy = ActionCollector_d86.robotp2worldp(p,rs_in_world)
        wvx,wvy = ActionCollector_d86.robotva2worldva(v[0],v[1],rs_in_world[2])
        wax,way = ActionCollector_d86.robotva2worldva(a[0],a[1],rs_in_world[2])
        return [(wx,wy),(wvx,wvy),(wax,way)]

    @staticmethod
    def worldva2robotva(wvax,wvay,now_wyaw):
        va =np.hypot(wvax,wvay)
        vayaw = np.arctan2(wvay,wvax)
        vax = np.cos(vayaw - now_wyaw)*va
        vay = np.sin(vayaw - now_wyaw)*va
        return vax,vay

    @staticmethod
    def worldp2robotp(point_in_world,robot_pose_in_world):
        wx,wy = point_in_world
        dist = np.hypot(wx - robot_pose_in_world[0],wy - robot_pose_in_world[1])
        p_theta_world = np.arctan2(wy - robot_pose_in_world[1],wx - robot_pose_in_world[0])
        return  dist*np.cos(p_theta_world - robot_pose_in_world[2]) ,dist*np.sin(p_theta_world - robot_pose_in_world[2])

    @staticmethod
    def robotp2worldp(point_in_robort,robot_pose_in_world):
        rx,ry= point_in_robort
        dist = np.hypot(rx,ry)
        p_theta_world = robot_pose_in_world[2] + np.arctan2(ry,rx)
        return robot_pose_in_world[0] + dist*np.cos(p_theta_world) ,robot_pose_in_world[1] + dist*np.sin(p_theta_world)

    @staticmethod
    def robotva2worldva(rvax,rvay,now_wyaw):
        va =np.hypot(rvax,rvay)
        vayaw = np.arctan2(rvay,rvax)
        vax = np.cos(vayaw + now_wyaw)*va
        vay = np.sin(vayaw + now_wyaw)*va
        return vax,vay


    @staticmethod
    def is_in_convex(point,convex):
        # convex为顺时针
        for i in range(1,len(convex)):
            cross = convex[i-1][0]*convex[i][1] - convex[i-1][1]*convex[i][0] +  (convex[i][0] - convex[i-1][0])*point[1]+ (convex[i-1][1] - convex[i][1])*point[0]
            if cross > 0:
                return False
        return True

    def call_msg_takeSimStep(self,action_msg, t=None):
        timeout = 20
        for i in range(timeout):
            action_msg.angular.z = t
            self.agent_action_pub.publish(action_msg)
            # 因为我们的控制只到v这层，控不了速度和加速度。可以从odom反馈看到返回的加速度都是0，导致
            # 达不到我们的轨迹状态要求，没办法。
            
            # try:
            #     rospy.wait_for_message(f"{self.ns_prefix}next_step_signal", Bool,timeout=0.02)
            # except rospy.ROSException:
            #         # print(self.ns+"_step timeout")
            #         if self.step_finish:
            #             self.step_finish = False
            #             break
                    
            #         rospy.logdebug("time out")
            #         if i == timeout - 1:
            #             raise TimeoutError(
            #                 f"Timeout while trying to call '{self.ns_prefix}next_step_signal'"
            #             )
            #         time.sleep(0.33)
            #         continue
            
            if self.step_finish:
                self.step_finish = False
                break
            else:
                if i == timeout - 1:
                    # print(self.ns + "timeout")
                    rospy.logwarn("action next step time out")
                    raise TimeoutError(
                        f"Timeout while trying to call '{self.ns_prefix}next_step_signal'"
                    )
                # print("took step")
                time.sleep(0.005)
                continue



    def call_service_takeSimStep(self, t=None):
        request = StepWorldRequest() if t is None else StepWorldRequest(t)
        timeout = 12

        try:
            for i in range(timeout):
                try:
                    rospy.wait_for_service(self._service_name_step,timeout=2.)
                except rospy.ROSException:
                    print(self.ns+"_step timeout")
                    rospy.logdebug("time out")
                    if i == timeout - 1:
                        raise TimeoutError(
                            f"Timeout while trying to call '{self.ns_prefix}step_world'"
                        ) 
                    continue
 
                response = self._sim_step_client(request)
            
                # rospy.logdebug("step service=", response)
                if response.success:
                    break
                if i == timeout - 1:
                    raise TimeoutError(
                        f"Timeout while trying to call '{self.ns_prefix}step_world'"
                    )
                # print("took step")
                time.sleep(0.33)

        except rospy.ServiceException as e:
            rospy.logdebug("step Service call failed: %s" % e)


    def reset(self,start_pos,goal_pos):
        self.start_state_world = [(start_pos.x,start_pos.y),(0.,0.),(0.,0.)]
        self.goal_pos_world = (goal_pos.x,goal_pos.y)
        self.exc_time = 0.
        self.traj_spline_world = None
        self.need_bark = True
        self.msg_Odometry  = None   
        self.msg_Odometry_last  = None   
        self.start_pos = start_pos


class ActionCollector_d86_simplify(ActionCollector_d86):
    def __init__(self,
    ns: str,
    lidar_num:int,
    output_points_num:int,
    plan_dynamic_limit=(3.,3.,4.)):

        super(ActionCollector_d86_simplify, self).__init__(
            ns,
            lidar_num,
            output_points_num,
            plan_dynamic_limit
        )

        self.planner =  quniform_one_bspline_optim(one_spline_ctrl_points_num=8,degrees=5,fix_end_pos_state=False
            ,v_limit=plan_dynamic_limit[0]
            ,a_limit=plan_dynamic_limit[1]
            ,j_limit=plan_dynamic_limit[2])
        
        # mpc_state_num = 30
        mpc_state_num = 4
        # self.controller = SO2MPC(N= mpc_state_num,control_period=self._action_period
        #     ,v_limit=plan_dynamic_limit[0]
        #     ,a_limit=plan_dynamic_limit[1]
        #     ,j_limit=plan_dynamic_limit[2])   
        self.controller = SO2MPC(N= mpc_state_num,control_period=self._action_period
            ,v_limit=plan_dynamic_limit[0]
            ,a_limit=plan_dynamic_limit[1]
            # ,a_limit=plan_dynamic_limit[1]*4.
            ,j_limit=plan_dynamic_limit[2]*2000.)

        
               



if __name__ == "__main__":
    action_collector = ActionCollector()
    print(action_collector.get_cmd_vel(1))

    box = spaces.Box(low=3.0, high=4, shape=(2, 2))
    print(box)
    box.seed(4)
    for _ in range(1):
        print(box.sample())

    min_position = 0
    max_position = 10
    max_speed = 2
    goal_position = 0.5
    low = np.array([min_position, -max_speed])
    high = np.array([max_position, max_speed])
    action_space = spaces.Discrete(3)  # action space
    observation_space = spaces.Box(low, high)  #
    print("*" * 10)
    print(observation_space)
    for _ in range(2):
        print(observation_space.sample())

    observation_space = spaces.Tuple(
        (
            spaces.Box(low=0, high=10, shape=(10,), dtype=np.float32),
            spaces.Box(low=-10, high=0, shape=(3 + 2,), dtype=np.float32),
        )
    )
    print("2" * 10)
    print(observation_space.sample())
    print(type(observation_space.sample()))

    reward = spaces.Discrete(4)
    print(type(reward.sample()))
