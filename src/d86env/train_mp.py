# -*- coding: utf-8 -*-
import os, rospy
import sys
import time
from datetime import datetime as dt
import torch
import numpy as np
import numpy.random as rd
import random
from copy import deepcopy
sys.path.append(r"./ElegantRL")
from ElegantRL.agent import *
from ElegantRL.run import train_and_evaluate, train_and_evaluate_mp
sys.path.append(r"/home/dmz/flat_ped_sim/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/tools")
from train_agent_utils_d86 import *

PI = 3.1415926535898


class d86_env_Arguments:
    def __init__(self, if_on_policy=False):
        self.env = None  # the environment for training
        self.agent = None  # Deep Reinforcement Learning algorithm

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -14  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        self.if_on_policy = if_on_policy
        if self.if_on_policy:  # (on-policy)
            self.net_dim = 2 ** 9  # the network width
            # num of transitions sampled from replay buffer.
            self.batch_size = self.net_dim * 2
            self.repeat_times = 2 ** 3  # collect target_step, then update network
            self.target_step = 2 ** 12  # repeatedly update network to keep critic's loss small
            self.max_memo = self.target_step  # capacity of replay buffer
            # GAE for on-policy sparse reward: Generalized Advantage Estimation.
            self.if_per_or_gae = False
        else:
            self.net_dim = 2 ** 8  # the network width
            # num of transitions sampled from replay buffer.
            self.batch_size = self.net_dim
            self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
            self.target_step = 2 ** 10  # collect target_step, then update network
            self.max_memo = 2 ** 21  # capacity of replay buffer
            # PER for off-policy sparse reward: Prioritized Experience Replay.
            self.if_per_or_gae = False

        '''Arguments for device'''
        self.env_num = 1  # The Environment number for each worker. env_num == 1 means don't use VecEnv.
        # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.worker_num = 2
        # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)
        self.thread_num = 8
        # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.visible_gpu = '0'
        self.random_seed = 10086  # initialize random seed in self.init_before_training()
        ################
        '''Arguments for evaluate and save'''
        self.cwd = None  # current work directory. None means set automatically
        # remove the cwd folder? (True, False, None:ask me)
        self.if_remove = False
        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        # allow break training when reach goal (early termination)
        self.if_allow_break = True

        # the environment for evaluating. None means set automatically.
        self.eval_env = None
        self.eval_gap = 2 ** 7  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2 ** 3  # number of times that get episode return in first
        self.eval_times2 = 2 ** 4  # number of times that get episode return in second
        self.eval_device_id = -1  # -1 means use cpu, >=0 means use GPU  ---new---
        # env
        self.arena_param = None
        self.arena_path = None
        # 360 sensor size + 4 self state size 环境单步返回的状态的维度为360个雷达信息+4自身运动状态信息
        self.env_sigle_state_dim = 360 + 4
        self.env_state_stack_times = 4
        self.env_state_dim = self.env_sigle_state_dim * self.env_state_stack_times
        
        self.which_env = -1 # 0 空环境 1 静态障碍 2 动态障碍
        self.env_get_goal_dist_accuracy = 0.4  # 位置精度-->判断是否到达目标的距离
        self.env_target_return = 21
        self.env_max_step = 1000
        self.env_action_dim = 10

        AGENT_NAME = "e2e_" + dt.now().strftime("%Y_%m_%d__%H_%M_%S")
        PATHS = get_path(AGENT_NAME)
        self.arena_path = PATHS
        print("Load Path: ")
        for k,v in PATHS.items():
            if  v is not None:
                print(k+"   "+v)
            else:
                print(k+"   None")
        print("________ STARTING TRAINING WITH:  %s ________\n" % AGENT_NAME)

        # for training with start_arena_flatland.launch
        ns_for_nodes = "/single_env" not in rospy.get_param_names()

        # check if simulations are booted
        wait_for_nodes(with_ns=ns_for_nodes, n_envs=self.env_num, timeout=5) # env_num用于训练的环境数量

        # initialize hyperparameters (save to/ load from json)
        self.arena_param = initialize_hyperparameter(
            PATHS=PATHS,
            load_target = None,
            n_envs=self.env_num,
        )
        
        # self.env_action_max = [2, PI * 2 / 3]  # v,w m/s rad/s
        # self.env_accel_max = [1, 2]
        # self.env_min_dist2obstacle = 0.11  # 离障碍的最近距离 由雷达检测 如果小于这个距离也认为发生碰撞 直接结束
        # self.env_max_linear_aceleration = 4  # 最大加速度 大于这个值认为发生碰撞
        # self.env_time_penalty = - 0.05  # 单步时间惩罚
        # self.env_goal_reward = 20  # 到达目标点最大奖励
        # self.env_after_reset2dect_col = 0.07  # 为了避免reset后一定会触发碰撞，设置一个延时时间在reset后的这段时间不检测碰撞
        # self.env_wait_for_envinit = 0.8  # reset中调用setstate初始化车体状态后多少秒后再去get车的初始状态
        # self.env_collision_penalty = -10  # 碰撞惩罚

    def init_before_training(self, if_main,id=0):
        random.seed(self.random_seed+id)
        np.random.seed(self.random_seed+id)
        os.environ['PYTHONHASHSEED'] = str(self.random_seed+id)
        torch.manual_seed(self.random_seed+id)
        # torch.manual_seed_all(self.random_seed+id)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)

        '''env'''
        # if self.env is None:
        #     raise RuntimeError(f'\n| Why env=None? For example:'
        #                        f'\n| args.env = XxxEnv()'
        #                        f'\n| args.env = str(env_name)'
        #                        f'\n| args.env = build_env(env_name), from elegantrl.env import build_env')
        # if not (isinstance(self.env, str) or hasattr(self.env, 'env_name')):
        #     raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env).')
    # generate agent name and model specific paths




        '''agent'''
        if self.agent is None:
            raise RuntimeError(f'\n| Why agent=None? Assignment `args.agent = AgentXXX` please.')
        if not hasattr(self.agent, 'init'):
            raise RuntimeError(f"\n| why hasattr(self.agent, 'init') == False"
                               f'\n| Should be `agent=AgentXXX()` instead of `agent=AgentXXX`.')
        if self.agent.if_on_policy != self.if_on_policy:
            raise RuntimeError(f'\n| Why bool `if_on_policy` is not consistent?'
                               f'\n| self.if_on_policy: {self.if_on_policy}'
                               f'\n| self.agent.if_on_policy: {self.agent.if_on_policy}')

        '''cwd'''
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            env_name = getattr(self.env, 'env_name', self.env)
            self.cwd = f'./{agent_name}_{env_name}_{self.visible_gpu}'
        if if_main:
            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
            elif self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Remove cwd: {self.cwd}")
            os.makedirs(self.cwd, exist_ok=True)


def args_init():
    # hyper-parameters of on-policy is different from off-policy
    args = d86_env_Arguments(if_on_policy=True)
    args.if_remove = False  # 是否要删除之前训练的网络 如果不删除会载入之前训练好的网络
    args.random_seed = 10086
    args.visible_gpu = '0'  # 使用的gpu编号
    # args.visible_gpu = '1'  # 使用的gpu编号
    args.worker_num = 6  # 每个learner下有两个worker来采集轨迹
    args.thread_num = 12  # 114服务器是12核的
    args.env_num = 1  # 每个worker下只有一个env 每个env的random_seed都不一样

    # e2e 0 convex 1 
    args.which_env = 1 
    # 参数的设定 网络的结构 网络的加载
    # Actor 和 Critic 的层数有待斟酌
    # 翻车检查不出来 要加懒惰惩罚
    '''choose an DRL algorithm'''
    from agent import AgentPPO
    args.agent = AgentPPO()
    args.if_per_or_gae = True
    args.agent.if_use_dn = False  # 类似resnet shortcut机制的网络
    # args.agent.cri_target = False  # 是否使用target critic 来软更新
    args.agent.if_use_cri_target = True
    args.agent.if_use_act_target = False
    args.learning_rate = 1e-4  # 3e-5 1e-4
    #####
    # 因为在envmodel中用了 rospy 去init node,ros规定一个进程内只能初始化一个ros node 所以不能用deepcopy去拷贝环境
    # 现在开了多进程 我们要保证每个env ros_master_url 和 随机种子都不一样
    # train_env = envmodel("train_env")
    # # eval_env = train_env  # envmodel("eval_env")
    # train_env.target_return = 20.2  # 期望策略回报
    # train_env.max_step = 1000  # 环境规定的单回合最大步数 若某个回合的步数大于这个值 直接结束此回合
    # args.env = PreprocessEnv(env=train_env)
    # args.eval_env = args.env  # PreprocessEnv(env=eval_env)
    # args.cwd = f'./{args.env.env_name}_{args.agent.__class__.__name__}/'  # 保存和读取网络的路径
    args.cwd = f'./my_env_AgentPPO/'  # 保存和读取网络的路径
    # args.if_allow_break = False  # 当某次的eval得分到达了target_return 是否就不训练了 直接训练结束
    # 360 sensor size + 4 self state size 环境单步返回的状态的维度为360个雷达信息+4自身运动状态信息
    
    # e2e 
    # args.env_sigle_state_dim = 360 + 2 + 2 
    max_vertex_num = 60
    ctrl_points_num = 2
    args.env_sigle_state_dim = max_vertex_num*2 + 2 + 1 + 1 + 2*ctrl_points_num*2 + 1


    args.env_state_stack_times = 3
    args.env_action_dim = 4
    # args.env_action_dim = 10
    args.env_action_max = [1., 1.]  # v,w m/s rad/s
    
    args.env_state_dim = args.env_sigle_state_dim * args.env_state_stack_times # + 360
    args.env_get_goal_dist_accuracy = 0.25  # 位置精度-->判断是否到达目标的距离
    args.env_target_return = 300.
    # 训练前期400 
    args.env_max_step = 300 # 1/(1-gama)  gama=0.98 maxstep = 500 gama=0.99 maxstep = 1000

    # args.env_accel_max = [1, 2] # m/s^2 rad/s^2
    # args.env_min_dist2obstacle = 0.11  # 离障碍的最近距离 由雷达检测 如果小于这个距离也认为发生碰撞 直接结束
    # args.env_max_linear_aceleration = 4  # 最大加速度 大于这个值认为发生碰撞 m/s^2
    # args.env_time_penalty = - 0.05  # 单步时间惩罚
    # args.env_goal_reward = 20  # 到达目标点最大奖励
    # args.env_after_reset2dect_col = 0.07  # 为了避免reset后一定会触发碰撞，设置一个延时时间在reset后的这段时间不检测碰撞
    # args.env_collision_penalty = -10  # 碰撞惩罚

    # 智能体的结算reward的0.1倍一定要大于日常reward才能避免被稀释。
    # 日常惩罚总和调整到-1~1之间
    # args.env_time_penalty = - 0.05  # 单步时间惩罚
    
    # gama = 0.98 0.98^(100 无障碍环境预计100步内能到终点) = 0.1326 1/0.1326 = 7.54---设置为无障碍环境+-13   0.98^60 = 0.2975 1/0.2975 = 3.36 -- 设置为+-6?
    # 障碍环境预计130步到终点 0.98^130 = 0.0723  1/0.0723 = 13.823 --- 给奖励20
    # 障碍环境130步到终点算快的 如果按160步内到 0.98^160 = 0.039 1/0.039 = 25.3 -- 给奖励30
    # args.env_goal_reward = 20  # 到达目标点最大奖励 8
    # args.env_wait_for_envinit = 0.45  # reset中调用setstate初始化车体状态后多少秒后再去get车的初始状态
    # args.env_after_reset2dect_col = 0.08  # 为了避免reset后一定会触发碰撞，设置一个延时时间在reset后的这段时间不检测碰撞
    # args.env_collision_penalty = -20  # 碰撞惩罚 一定要是负的！！！ 因为奖励函数那写成+这个惩罚
    #####
    # 如果我希望考虑接下来的max_step 步，那么我让第t步的reward占现在这一步的Q值的 0.1
    # gama = 0.1^(1/max_step)   0.9 ~= 0.1 ** (1/  22)  0.96  ~= 0.1 ** (1/  56)  0.98  ~= 0.1 ** (1/ 114) 0.99  ~= 0.1 ** (1/ 229) 0.999 ~= 0.1 ** (1/2301)  # 没必要，DRL目前无法预测这么长的MDPs过程
    args.gamma = 0.98  # 0.985  # 未来回报折扣因子 越小越不看重未来收益(短视) 大看的远 在DRL中agent“看得远”表面上指的是向前考虑的步数多，实质上是指agent向前考虑的系统动态演化跨度大
    args.reward_scale = 1.  # 奖励缩放系数
    # 更新tar模型 保留新模型self.cri参数的tau*100 %  cri_target (1-tau)*100 % 不变
    # args.soft_update_tau = 2 ** -11  # 软更新参数  改小 方差大但收敛于更高的奖励
    args.agent.explore_rate = 0.9  # 探索 在选择动作时对噪声缩放系数选取的概率
    args.agent.lambda_a_value = 1.
    # buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
    # buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (self.lambda_a_value / (buf_adv_v.std() + 1e-5))

    # 0.005-0.05 推荐0.01 越大新旧策略相差越大 探索越大
    # args.agent.lambda_entropy = 0.01  # 0.006 探索熵系数 越大 越鼓励探索 略微加大探索  改大平均得分更高 但是方差更大 改低 得分略低 方差小
    # args.agent.lambda_entropy = 0.005  # 0.006 探索熵系数 越大 越鼓励探索 略微加大探索  改大平均得分更高 但是方差更大 改低 得分略低 方差小
    # args.agent.lambda_gae_adv = 0.98  # 0.96-0.99 0.98 gae /\系数  代表这我们对于方差与偏差的取舍。降低此参数可减少critic方差但会微增偏差它也是一个“衰减因子”  一般情况下，训练刚开始的时候，我们的Critic效果较差，即其估计出的V(s)会与现实情况有较大的偏差。此时，应该将tau设计较大
    # args.agent.lambda_gae_adv = 0.95  # 0.96-0.99 0.98 gae /\系数  代表这我们对于方差与偏差的取舍。降低此参数可减少critic方差但会微增偏差它也是一个“衰减因子”  一般情况下，训练刚开始的时候，我们的Critic效果较差，即其估计出的V(s)会与现实情况有较大的偏差。此时，应该将tau设计较大
    args.agent.ratio_clip = 0.2 # 0.1-0.3 Cliprange越小训练越稳定，越大越节省数据。一般在训练早期取较小的值，比如0.2，保证训练平稳进行；到训练后期可以适当放大，因为此时policy已经足够优秀，所采集数据中正样本比例非常高，可以放心利用。
    ####
    # args.net_dim = 2 ** 8  # actor和critic全连接中间层的维度
    args.net_dim = 2 ** 8  # actor和critic全连接中间层的维度
    # args.batch_size =  2 ** 14 # args.net_dim * 4  # 单次策略优化时 从经验池中sample batch_size步的数据进行 网络的更新
    # args.batch_size = 2**14 # args.net_dim * 4  # 单次策略优化时 从经验池中sample batch_size步的数据进行 网络的更新
    # args.batch_size = 2**10 # args.net_dim * 4  # 单次策略优化时 从经验池中sample batch_size步的数据进行 网络的更新

    # 我们放宽对新旧策略的限制（如，把clip改大），鼓励探索（如，把 lambda_entropy 调大），那么我们就一定需要减少 repeat_times，防止过度更新导致新旧策略差异过大。
    # args.soft_update_tau = 2 ** -11
    # args.agent.lambda_entropy = 0.005
    # args.agent.lambda_gae_adv = 0.95
    # args.batch_size = 2**14 
    # args.repeat_times = 5  # 每次update时 利用当前经验池中的数据 重复几次策略优化(网络更新)
    # args.target_step = 14000 # args.env_max_step * 8

    # args.soft_update_tau = 2 ** -9
    # args.agent.lambda_entropy = 0.05
    # args.agent.lambda_gae_adv = 0.97
    # args.batch_size = 2**10 
    # args.repeat_times = 2**4  # 每次update时 利用当前经验池中的数据 重复几次策略优化(网络更新)
    # args.target_step = 4000 # args.env_max_step * 8

    args.soft_update_tau = 2 ** -11
    # args.agent.lambda_entropy = 0.008
    # args.agent.lambda_gae_adv = 0.95
    args.agent.lambda_gae_adv = 0.98
    # args.batch_size = 2**4 
    # args.repeat_times = 3  # 每次update时 利用当前经验池中的数据 重复几次策略优化(网络更新)
    # args.target_step = 11000 # args.env_max_step * 8

    args.agent.lambda_entropy = 0.05
    args.batch_size = 2**15 
    args.repeat_times = 16  # 每次update时 利用当前经验池中的数据 重复几次策略优化(网络更新)
    args.target_step = args.env_max_step * 16 * 4

    # args.agent.lambda_entropy = 0.04
    # args.batch_size = 2**9
    # args.repeat_times = 8  # 每次update时 利用当前经验池中的数据 重复几次策略优化(网络更新)
    # args.target_step = args.env_max_step * 16


    # args.env.max_step * 8  2 ** 12  # 采集到不多不少target_step步的经验就会对策略进行一次的update

    args.break_step = int(2e7)  # 当总的经验池历史收集步数 综合达到 这个步数 结束训练
    # args.break_step = int(1e7)  # 当总的经验池历史收集步数 综合达到 这个步数 结束训练
    args.max_memo = args.target_step*args.worker_num  # 经验池大小 = max_memo + max_step
    ####
    args.if_allow_break = False  # 不允许提前终止
    args.eval_gap = 2 ** 8  # for Recorder 两次验证的最短时间差 单位s 2**5
    # for Recorder 一次验证中跑几个回合(如果结果并没比当前最好的结果还好 就没有后续的验证了) 2**2
    args.eva_size1 = 2 ** 2
    # for Recorder 2**4 如果在eva_size1次的结果奖励 大于当前最大奖励 则用当前模型继续跑eva_size2-eva_size1个回合 检查当前模型是否确实还可以
    args.eva_size2 = 2 ** 4
    '''train and evaluate'''
    # train_and_evaluate_PPO(args)
    # args.rollout_num = 2
    # train_and_evaluate_mp(args)
    return args

def d86_train():
    train_and_evaluate_mp(args_init())


if __name__ == '__main__':
    d86_train()
