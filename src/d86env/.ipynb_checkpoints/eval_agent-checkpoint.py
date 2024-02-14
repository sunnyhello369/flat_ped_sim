#!/home/chen-ubuntu/anaconda3/envs/torch16/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import time
import torch
import numpy as np
import numpy.random as rd
import random
from copy import deepcopy
sys.path.append(r"./ElegantRL")
from ElegantRL.agent import *

from ElegantRL.env import build_env, build_gazebo_env,build_arena_env
from ElegantRL.evaluator import Evaluator

PI = 3.1415926535898
def sigmoid(x):
    if x < 0:
        return np.exp(x)/(1.+np.exp(x))
    else:
        return 1./(1.+np.exp(-x))

def get_episode_return_and_step(env, act, device) -> (float, int):
    episode_step = 1
    episode_return = 0.0  # sum of rewards in an episode

    max_step = env.max_step
    
    if_discrete = env.if_discrete

    state = env.reset()

    time.sleep(2.)

    for episode_step in range(300):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)  # 直接用actor的forward得到tanh后的mean(-1,1)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        # not need detach(), because with torch.no_grad() outside
        action = a_tensor.detach().cpu().numpy()[0]
        state, reward, done, _ = env.step(sigmoid(action))  # 在step中将(-1,1)乘上action max
        # state, reward, done, _ = env.step((-1.,-1.))  # 在step中将(-1,1)乘上action max

        # print("step reward:",reward)

        episode_return += reward
        if done:
            break
    episode_return = getattr(env, 'episode_return', episode_return)
    print("steps:",episode_step,"  reward:",episode_return)
    return episode_return, episode_step


def get_r_avg_std_s_avg_std(rewards_steps_list):
    rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)
    # average of episode return and episode step
    r_avg, s_avg = rewards_steps_ary.mean(axis=0)
    # standard dev. of episode return and episode step
    r_std, s_std = rewards_steps_ary.std(axis=0)
    sum_r, total_step = rewards_steps_ary.sum(axis=0)
    return r_avg, r_std, s_avg, s_std, sum_r, total_step


def save_learning_curve(recorder=None, cwd='.', save_title='learning curve', fig_name='plot_test_curve.jpg'):
    if recorder is None:
        recorder = np.load(f"{cwd}/recorder.npy")

    recorder = np.array(recorder)
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]

    # self.total_step--[0], r_avg--[1], r_std--[2], r_exp--[3], *log_tuple(critic_LOSS--[4], actor_LOSS--[5])
    # 希望critic_LOSS接近于0 此loss是critic网络对gae或raw_reward的拟合 希望拟合的越准越好
    # 希望actor_LOSS越小越好 此loss是-advantage * clamp ratio + policy entropy(因为熵是负的 所以lambda_entropy越大则优化器偏向增大熵)

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)

    '''axs[0]'''
    ax00 = axs[0]
    ax00.cla()

    ax01 = axs[0].twinx()
    color0 = 'lightcoral'
    ax00.set_ylabel('Episode Return')
    ax00.plot(steps, r_avg, label='Episode Return', color=color0)
    ax00.fill_between(steps, r_avg - r_std, r_avg +
                      r_std, facecolor=color0, alpha=0.3)
    ax00.grid()

    '''axs[1]'''

    '''plot save'''
    plt.title(
        save_title, y=2.3)  # save_title = f"step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"
    plt.savefig(f"{cwd}/{fig_name}")
    # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    plt.close('all')
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()


def utility_evaluate_(args, eps=10,env_id=0, agent_id=0):
    args.init_before_training(if_main=False)
    seed = 10000
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    # env = build_gazebo_env(args, env_id)  # 用自己的ros环境
    env = build_arena_env(args, env_id)  # 用自己的ros环境
    time.sleep(3.)
    agent = args.agent
    agent.init(args.net_dim, args.env_state_dim,
               args.env_action_dim, agent_id=args.eval_device_id)
    agent.save_or_load_agent(args.cwd, if_save=False)
    act = agent.act
    act.eval()  # 调整模型到验证模式
    [setattr(param, 'requires_grad', False) for param in act.parameters()]
    recorder = list()  # total_step, r_avg, r_std, obj_c, ...
    time.sleep(1.)
    rewards_steps_list = [get_episode_return_and_step(env, act, agent.device) for _ in
                          range(eps)]  # eps表示验证多少个episode
    r_avg, r_std, s_avg, s_std, sum_r, total_step = get_r_avg_std_s_avg_std(
        rewards_steps_list)
    recorder.append((total_step, r_avg, r_std))  # update recorder
    save_learning_curve(recorder, save_title='sum reward:' +
                                             str(sum_r), fig_name=args.cwd)
    print(sum_r/eps)


def utility_evaluate():
    agent_cwd = f'./my_env_AgentPPO'
    # hyper-parameters of on-policy is different from off-policy
    from train_mp import d86_env_Arguments
    args = d86_env_Arguments(if_on_policy=True)
    args.eval_device_id = -1  # GPU 1号
    from ElegantRL.agent import AgentPPO
    args.agent = AgentPPO()
    args.if_per_or_gae = False
    args.agent.if_use_dn = False  # 类似resnet shortcut机制的网络
    # args.agent.cri_target = False  # 是否使用target critic 来软更新
    args.agent.if_use_cri_target = False
    args.agent.if_use_act_target = False
    args.learning_rate = 5e-5
    args.random_seed = 100
    
    args.which_env = 1
    args.env_action_dim = 2
    # args.which_env = 1
    # args.env_action_dim = 10
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
    args.cwd = agent_cwd  # 保存和读取网络的路径
    # args.if_allow_break = False  # 当某次的eval得分到达了target_return 是否就不训练了 直接训练结束
    # 360 sensor size + 4 self state size 环境单步返回的状态的维度为360个雷达信息+4自身运动状态信息
    
    # args.which_env = 0 # 0 空环境 1 静态障碍 2 动态障碍
    ctrl_points_num = 1
    args.env_sigle_state_dim = 360 + 2 + 1 + 1 + 2*ctrl_points_num*2 + 1
    args.env_state_stack_times = 4
    args.env_state_dim = args.env_sigle_state_dim * args.env_state_stack_times # + 360
    args.env_get_goal_dist_accuracy = 0.25  # 位置精度-->判断是否到达目标的距离
    args.env_target_return = 21
    args.env_max_step = 600

    args.env_action_max = [3, 4]  # v,w m/s rad/s
    args.env_min_dist2obstacle = 0.11  # 离障碍的最近距离 由雷达检测 如果小于这个距离也认为发生碰撞 直接结束
    args.env_max_linear_aceleration = 4  # 最大加速度 大于这个值认为发生碰撞 m/s^2

    args.env_time_penalty = - 0.05  # 单步时间惩罚
    # gama = 0.98 0.98^(100 无障碍环境预计100步内能到终点) = 0.1326 1/0.1326 = 7.54---设置为无障碍环境+-13   0.98^60 = 0.2975 1/0.2975 = 3.36 -- 设置为+-6?
    # 障碍环境预计130步到终点 0.98^130 = 0.0723  1/0.0723 = 13.823 --- 给奖励20
    # 障碍环境130步到终点算快的 如果按160步内到 0.98^160 = 0.039 1/0.039 = 25.3 -- 给奖励30
    args.env_goal_reward = 20  # 到达目标点最大奖励 8
    args.env_wait_for_envinit = 0.45  # reset中调用setstate初始化车体状态后多少秒后再去get车的初始状态
    args.env_after_reset2dect_col = 0.08  # 为了避免reset后一定会触发碰撞，设置一个延时时间在reset后的这段时间不检测碰撞
    args.env_collision_penalty = -20  # 碰撞惩罚 一定要是负的！！！ 因为奖励函数那写成+这个惩罚

    ####
    args.net_dim = 2 ** 8  # actor和critic全连接中间层的维度
    args.batch_size = 2 ** 8  # 单次策略优化时 从经验池中sample batch_size步的数据进行 网络的更新
    args.repeat_times = 2 ** 3  # 每次update时 利用当前经验池中的数据 重复几次策略优化(网络更新)
    # args.env.max_step * 8  2 ** 12  # 采集到不多不少target_step步的经验就会对策略进行一次的update
    args.target_step = args.env_max_step * 4
    args.break_step = int(8e6)  # 当总的经验池历史收集步数 综合达到 这个步数 结束训练
    # args.max_memo = args.env.max_step * 4  # 经验池大小 = max_memo + max_step
    ####
    utility_evaluate_(args, eps=10,env_id=1)


if __name__ == '__main__':
    utility_evaluate()
