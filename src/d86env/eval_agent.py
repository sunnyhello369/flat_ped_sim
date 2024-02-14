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
from train_mp import args_init
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
    
# 230528 data analyze
from geometry_msgs.msg import PointStamped
import rospy
from nav_msgs.msg import Odometry
import json
step_reward_pub = rospy.Publisher('/sim_1/step_reward', PointStamped, queue_size=10)
eps_num = 0
action_odom = Odometry()
def robot_state_callback(msg):
    global action_odom
    action_odom = msg
rospy.Subscriber("/sim_1/action_odom", Odometry, robot_state_callback)

def get_parent_directory(target_string):
    current_file = os.path.abspath(__file__)
    path_parts = current_file.split(os.path.sep)

    if target_string in path_parts:
        target_index = path_parts.index(target_string)
        target_path = os.path.sep.join(path_parts[:target_index+1]) + os.path.sep
        return target_path
    else:
        return None
d86path_json_path = get_parent_directory("flat_ped_sim") + "src/data_collected_analyze.json"
# ===

def get_episode_return_and_step(env, act, device) -> (float, int):
    global eps_num
    eps_num += 1
    episode_step = 1
    episode_return = 0.0  # sum of rewards in an episode

    max_step = env.max_step
    
    if_discrete = env.if_discrete

    state = env.reset()

    time.sleep(2.)

    for episode_step in range(1000):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)  # 直接用actor的forward得到tanh后的mean(-1,1)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        # not need detach(), because with torch.no_grad() outside
        # action = a_tensor.detach().cpu().numpy()[0]

        # for record
        env.action_collector.steps = episode_step 
        env.action_collector.record()
        # ===

        action = torch.sigmoid(a_tensor).detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        state, reward, done, info = env.step(action)

        # state, reward, done, _ = env.step(sigmoid(action))  # 在step中将(-1,1)乘上action max
        # state, reward, done, _ = env.step((-1.,-1.))  # 在step中将(-1,1)乘上action max

        # print("step reward:",reward)

        # 230528 data analyze
        # need to update env _max_steps_per_episode
        success = False
        timeout = False
        collide = False
        if    done:
            if info["is_success"]:
                success = True
            elif info["done_reason"] == 0:
                timeout = True
            else:
                collide = True 
        robot_pos = action_odom.pose.pose.position
        robot_x = robot_pos.x
        robot_y = robot_pos.y
        vel_x = action_odom.twist.twist.linear.x 
        vel_y = action_odom.twist.twist.linear.y

        data_entry = {
                "position": (robot_x, robot_y),
                "velocity": (vel_x, vel_y),
                "time": episode_step*0.1,
                "collide": collide,
                "timeout": timeout,
                "success": success,
                "eps": eps_num
        }
        # with open(d86path_json_path, 'a') as f:
        #     f.write(json.dumps(data_entry) + '\n')
        # ===

        # for record
        step_reward = PointStamped()
        step_reward.header.stamp = rospy.Time.now() # 设置时间戳为当前 ROS 时间
        step_reward.header.frame_id = "map" # 设置参考坐标系 ID 为 "map"
        # step_reward.header.seq = env.action_collector.steps
        step_reward.point.x = episode_step
        step_reward.point.y = reward
        step_reward.point.z = 0
        step_reward_pub.publish(step_reward)
        # for record ===

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
    utility_evaluate_(args_init(), eps=1000,env_id=1)


if __name__ == '__main__':
    utility_evaluate()
