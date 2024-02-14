import os
import time

import gym
import torch
import random
import numpy as np
import numpy.random as rd
import multiprocessing as mp

from copy import deepcopy
from env import build_env, build_gazebo_env,build_arena_env
from replay import ReplayBuffer, ReplayBufferMP
from evaluator import Evaluator

"""[ElegantRL.2021.09.01](https://github.com/AI4Finance-Foundation/ElegantRL)"""


class Arguments:
    def __init__(self, if_on_policy=False):
        self.env = None  # the environment for training
        self.agent = None  # Deep Reinforcement Learning algorithm

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -14  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3
        self.max_step = 200
        
        self.if_on_policy = if_on_policy
        if self.if_on_policy:  # (on-policy)
            self.net_dim = 2 ** 9  # the network width
            self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 3  # collect target_step, then update network
            self.target_step = 2 ** 12  # repeatedly update network to keep critic's loss small
            self.max_memo = self.target_step  # capacity of replay buffer
            self.if_per_or_gae = False  # GAE for on-policy sparse reward: Generalized Advantage Estimation.
        else:
            self.net_dim = 2 ** 8  # the network width
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
            self.target_step = 2 ** 10  # collect target_step, then update network
            self.max_memo = 2 ** 21  # capacity of replay buffer
            self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.

        '''Arguments for device'''
        self.env_num = 1  # The Environment number for each worker. env_num == 1 means don't use VecEnv.
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.thread_num = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)
        self.visible_gpu = '0'  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.random_seed = 0  # initialize random seed in self.init_before_training()

        self.env_state_dim = 3
        self.env_action_dim = 2

        '''Arguments for evaluate and save'''
        self.cwd = None  # current work directory. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.if_allow_break = True  # allow break training when reach goal (early termination)

        self.eval_env = None  # the environment for evaluating. None means set automatically.
        self.eval_gap = 2 ** 7  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2 ** 3  # number of times that get episode return in first
        self.eval_times2 = 2 ** 4  # number of times that get episode return in second
        self.eval_device_id = -1  # -1 means use cpu, >=0 means use GPU

    def init_before_training(self, if_main,id=0):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)

        '''env'''
        if self.env is None:
            raise RuntimeError(f'\n| Why env=None? For example:'
                               f'\n| args.env = XxxEnv()'
                               f'\n| args.env = str(env_name)'
                               f'\n| args.env = build_env(env_name), from elegantrl.env import build_env')
        if not (isinstance(self.env, str) or hasattr(self.env, 'env_name')):
            raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env).')

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


'''single processing training'''


def train_and_evaluate(args, agent_id=0):  # 单GPU版本没有更新 和之前的是一样的
    args.init_before_training(if_main=True)

    env = build_env(args.env, if_print=False)

    '''init: Agent'''
    agent = args.agent
    agent.init(args.net_dim, env.state_dim, env.action_dim, args.learning_rate, args.if_per_or_gae, args.env_num)
    agent.save_or_load_agent(args.cwd, if_save=False)

    '''init Evaluator'''
    eval_env = build_env(env) if args.eval_env is None else args.eval_env
    evaluator = Evaluator(args.cwd, agent_id, agent.device, eval_env,
                          args.eval_gap, args.eval_times1, args.eval_times2)
    # evaluator.save_or_load_recoder(if_save=False)

    '''init ReplayBuffer'''
    if agent.if_on_policy:
        buffer = list()
    else:
        buffer = ReplayBuffer(max_len=args.max_memo, state_dim=env.state_dim,
                              action_dim=1 if env.if_discrete else env.action_dim,
                              if_use_per=args.if_per_or_gae)
        buffer.save_or_load_history(args.cwd, if_save=False)

    """start training"""
    cwd = args.cwd
    gamma = args.gamma
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    reward_scale = args.reward_scale
    if_allow_break = args.if_allow_break
    soft_update_tau = args.soft_update_tau
    del args

    '''choose update_buffer()'''
    if agent.if_on_policy:
        assert isinstance(buffer, list)

        def update_buffer(_trajectory):
            _trajectory = list(map(list, zip(*_trajectory)))  # 2D-list transpose
            ten_state = torch.as_tensor(_trajectory[0])
            ten_reward = torch.as_tensor(_trajectory[1], dtype=torch.float32) * reward_scale
            ten_mask = (1.0 - torch.as_tensor(_trajectory[2], dtype=torch.float32)) * gamma  # _trajectory[2] = done
            ten_action = torch.as_tensor(_trajectory[3])
            ten_noise = torch.as_tensor(_trajectory[4], dtype=torch.float32)

            buffer[:] = (ten_state, ten_action, ten_noise, ten_reward, ten_mask)

            _steps = ten_reward.shape[0]
            _r_exp = ten_reward.mean()
            return _steps, _r_exp
    else:
        assert isinstance(buffer, ReplayBuffer)

        def update_buffer(_trajectory_list):
            _steps = 0
            _r_exp = 0
            for _trajectory in _trajectory_list:
                ten_state = torch.as_tensor([item[0] for item in _trajectory], dtype=torch.float32)
                ary_other = torch.as_tensor([item[1] for item in _trajectory])
                ary_other[:, 0] = ary_other[:, 0] * reward_scale  # ten_reward
                ary_other[:, 1] = (1.0 - ary_other[:, 1]) * gamma  # ten_mask = (1.0 - ary_done) * gamma

                buffer.extend_buffer(ten_state, ary_other)

                _steps += ten_state.shape[0]
                _r_exp += ary_other[:, 0].mean()  # other = (reward, mask, action)
            return _steps, _r_exp

    '''init ReplayBuffer after training start'''
    agent.states = [env.reset(), ]
    if not agent.if_on_policy:
        if_load = buffer.save_or_load_history(cwd, if_save=False)

        if not if_load:
            trajectory = explore_before_training(env, target_step)
            trajectory = [trajectory, ]
            steps, r_exp = update_buffer(trajectory)
            evaluator.total_step += steps

    '''start training loop'''
    if_train = True
    while if_train:
        with torch.no_grad():
            trajectory = agent.explore_env(env, target_step)
            steps, r_exp = update_buffer(trajectory)

        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
        with torch.no_grad():
            if_reach_goal, if_save = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple)
            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))

    print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

    env.close()
    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if not agent.if_on_policy else None
    evaluator.save_or_load_recoder(if_save=True)


def explore_before_training(env, target_step):  # for off-policy only
    trajectory = list()

    if_discrete = env.if_discrete
    action_dim = env.action_dim

    state = env.reset()
    step = 0
    while True:
        if if_discrete:
            action = rd.randint(action_dim)  # assert isinstance(action_int)
            next_s, reward, done, _ = env.step(action)
            other = (reward, done, action)
        else:
            action = rd.uniform(-1, 1, size=action_dim)
            next_s, reward, done, _ = env.step(action)
            other = (reward, done, *action)

        trajectory.append((state, other))
        state = env.reset() if done else next_s

        step += 1
        if done and step > target_step:
            break
    return trajectory


def explore_before_training_vec_env(env, target_step) -> list:  # for off-policy only
    # plan to be elegant: merge this function to explore_before_training()
    assert hasattr(env, 'env_num')
    env_num = env.env_num

    trajectory_list = [list() for _ in range(env_num)]

    if_discrete = env.if_discrete
    action_dim = env.action_dim

    states = env.reset()
    step = 0
    while True:
        if if_discrete:
            actions = rd.randint(action_dim, size=env_num)
            s_r_d_list = env.step(actions)

            next_states = list()
            for env_i in range(env_num):
                next_s, reward, done = s_r_d_list[env_i]
                trajectory_list[env_i].append((states[env_i], (reward, done, actions[env_i])))
                next_states.append(next_s)
        else:
            actions = rd.uniform(-1, 1, size=(env_num, action_dim))
            s_r_d_list = env.step(actions)

            next_states = list()
            for env_i in range(env_num):
                next_s, reward, done = s_r_d_list[env_i]
                trajectory_list[env_i].append((states[env_i], (reward, done, *actions[env_i])))
                next_states.append(next_s)
        states = next_states

        step += 1
        if step > target_step:
            break
    return trajectory_list


'''multiple processing training'''


class WorkerPipe:  # 采集轨迹的进程
    def __init__(self, env_num, worker_num):
        self.env_num = env_num
        self.worker_num = worker_num
        # 单个worker与主进程workvec通信的管道 发送自己探索得到的轨迹 接收update后的actor网络
        self.pipes = [mp.Pipe() for _ in range(worker_num)]
        # 主进程workvec与多个worker进程通信的管道 发送update后的actor网络 接收每个进程探索得到的轨迹并总结
        self.pipe1s = [pipe[1] for pipe in self.pipes]

    def explore(self, agent):  # 主进程 管理多个worker进程
        # 在learner中调用的是这个函数
        # learner将其update后的新模型传进来 并等待worker探索完将总结的轨迹传回
        act_dict = agent.act.state_dict()
        # print("ok10")
        for worker_id in range(self.worker_num):
            self.pipe1s[worker_id].send(act_dict)
        # print("ok11")
        trajectory_lists = [pipe1.recv() for pipe1 in self.pipe1s]
        # print("ok12")
        return trajectory_lists

    def run(self, args, comm_env, worker_id, learner_id):  # env_pipe, worker_id, learner_id
        # print(f'| os.getpid()={os.getpid()} PipeExplore.run {learner_id}')
        random_id = self.worker_num * learner_id + worker_id + 1
        torch.manual_seed(int(random_id)%int(1e6))
        args.init_before_training(if_main=False,id=random_id)
        # worker的随机种子都是一样的! 这样在sample action时的随机noise 相同? 熵产生的随机动作相同
        # 如果传入的是comm_env 则是外部的vec env pipe否则使用的就是下面初始化的env

        '''init Agent'''
        # env = build_env(args.env, if_print=False)
        # 两个learner 每个learner下两个worker
        env = build_arena_env(args, env_id=self.worker_num * learner_id + worker_id + 1)  # 用自己的ros环境 0号留给验证环境
        
        agent = args.agent
        agent.init(args.net_dim, args.env_state_dim, args.env_action_dim,
                   args.learning_rate, args.if_per_or_gae, args.env_num, learner_id)

        '''loop'''
        gamma = args.gamma
        target_step = args.target_step
        reward_scale = args.reward_scale
        if args.if_on_policy:
            def convert_trajectory(_trajectory):  # on-policy
                _trajectory = list(map(list, zip(*_trajectory)))  # 2D-list transpose

                ten_state = torch.as_tensor(_trajectory[0])
                ten_reward = torch.as_tensor(_trajectory[1], dtype=torch.float32) * reward_scale
                ten_mask = (1 - torch.as_tensor(_trajectory[2], dtype=torch.float32)) * gamma
                ten_action = torch.as_tensor(_trajectory[3])
                ten_noise = torch.as_tensor(_trajectory[4], dtype=torch.float32)
                return ten_state, ten_action, ten_noise, ten_reward, ten_mask
        else:
            def convert_trajectory(_trajectory):  # off-policy
                temp = list()
                for item_trajectory in _trajectory:
                    ten_state = torch.as_tensor([item[0] for item in item_trajectory])
                    ten_other = torch.as_tensor([item[1] for item in item_trajectory])
                    ten_other[:, 0] = ten_other[:, 0] * reward_scale  # ten_reward
                    ten_other[:, 1] = (1.0 - ten_other[:, 1]) * gamma  # ten_mask = (1.0 - ten_done) * gamma

                    temp.append((ten_state, ten_other))
                return temp
        del args

        if comm_env:  # 一个worker下有多个env
            env = comm_env
            agent.states = env.reset()  # 调用的是主vecenv进程下的 reset 将vecenv其下的所有env都reset了并总结
        else:  # 一个worker下一个env
            agent.states = [env.reset(), ]  # 单个env

        with torch.no_grad():
            while True:
                # 等待接收 update后的actor网络
                # print(worker_id+1,"s1")
                act_dict = self.pipes[worker_id][0].recv() # 2 3
                # print(worker_id+1,"s2")
                agent.act.load_state_dict(act_dict)
                # print(worker_id+1,"s3")
                # 接收到新actor 后开始探索轨迹 探索到指定的步数后将轨迹发出去
                # 这个explore_env不是上面的那个explore 就是一个agent在一个环境中探索
                trajectory = agent.explore_env(env,
                                               target_step)  # 在agent初始化时会根据其传入的envnum参数来自动选择explore_env是one_env还是vec_env
                # print(worker_id+1,"s4")
                trajectory = convert_trajectory(trajectory)
                # 将这个worker探索到的轨迹发给workvec主进程 让其总结
                # print("ok2")
                # print(worker_id+1,"s5")
                self.pipes[worker_id][0].send(trajectory)
                # print("ok3")
                # print(worker_id+1,"s6")
                random_id += 1
                random.seed(int(random_id)%int(1e6))
                np.random.seed(int(random_id)%int(1e6))
                # torch.manual_seed(int(random_id)%int(1e6))

def get_comm_data(agent):
    act = list(agent.act.parameters())
    cri_optim = get_optim_parameters(agent.cri_optim)

    if agent.cri is agent.act:
        cri = None
        act_optim = None
    else:
        cri = list(agent.cri.parameters())
        act_optim = get_optim_parameters(agent.act_optim)

    act_target = list(agent.act_target.parameters()) if agent.if_use_act_target else None
    cri_target = list(agent.cri_target.parameters()) if agent.if_use_cri_target else None
    return act, act_optim, cri, cri_optim, act_target, cri_target  # data


class LearnerPipe:
    def __init__(self, learner_num):
        self.learner_num = learner_num  # learner数量一定要时2的幂次数
        self.round_num = int(np.log2(learner_num))  # 对数量取2的对数 2个learner--log2(2) = 1

        # 用于传输轨迹数据的管道 与woker相连
        self.pipes = [mp.Pipe() for _ in range(learner_num)]
        pipes = [mp.Pipe() for _ in range(learner_num)]
        # 用于传输网络和优化器的管道 与其他learner相连
        self.pipe0s = [pipe[0] for pipe in pipes]
        self.pipe1s = [pipe[1] for pipe in pipes]
        self.device_list = [torch.device(f'cuda:{i}') for i in range(learner_num)]  # 0,1 learner数量由GPU数量决定

        if learner_num == 1:
            self.idx_l = None
        elif learner_num == 2:
            self.idx_l = [(1,), (0,), ]
        elif learner_num == 4:
            self.idx_l = [(1, 2), (0, 3),
                          (3, 0), (2, 1), ]
        elif learner_num == 8:
            self.idx_l = [(1, 2, 4), (0, 3, 5),
                          (3, 0, 6), (2, 1, 7),
                          (5, 6, 0), (4, 7, 1),
                          (7, 4, 2), (6, 5, 3), ]
        else:
            print(f"| LearnerPipe, ERROR: learner_num {learner_num} should in (1, 2, 4, 8)")
            exit()

    def comm_data(self, data, learner_id, round_id):
        if round_id == -1:  # data为轨迹数据时传入的round_id就是-1
            # 传入的data为轨迹数据
            learner_jd = self.idx_l[learner_id][round_id]
            # 每个learner进程会将自己common_explore采集到的轨迹发给其他进程
            self.pipes[learner_jd][0].send(data)
            # 在这里等待接收其他进程发来的他们的common_explore采集到的轨迹
            return self.pipes[learner_id][1].recv()
        else:  # data为 网络参数时 发送自己的网络参数 并等待其他learner的网络参数
            learner_jd = self.idx_l[learner_id][round_id]
            self.pipe0s[learner_jd].send(data)
            return self.pipe1s[learner_id].recv()

    # def comm_network_optim(self, agent, learner_id):
    #
    #     device = self.device_list[learner_id]
    #
    #     for round_id in range(self.round_num):
    #         act = list(agent.act.parameters())
    #         act_optim = get_optim_parameters(agent.act_optim)
    #
    #         if agent.cri is agent.act:
    #             cri = None
    #             cri_optim = None
    #         else:
    #             cri = list(agent.cri.parameters())
    #             cri_optim = get_optim_parameters(agent.cri_optim)
    #
    #         act_target = list(agent.act_target.parameters()) if agent.if_use_act_target else None
    #         cri_target = list(agent.cri_target.parameters()) if agent.if_use_cri_target else None
    #
    #         data = (act, act_optim, cri, cri_optim, act_target, cri_target)
    #         data = self.comm_data(data, learner_id, round_id)
    #         # 将当前learner训练的网络和优化器的一样以comm_data发给 其他learner进程
    #         # 并且等待接收其他进程发来的网络和优化器 并将自己的网络和优化器 与 其他进程发来的网络和优化器 参数 加权平局作为自己网络和优化器的新参数
    #
    #         if data:
    #             avg_update_net(agent.act, data[0], device)
    #             avg_update_optim(agent.act_optim, data[1], device)
    #
    #             avg_update_net(agent.cri, data[2], device) if data[2] else None
    #             avg_update_optim(agent.cri_optim, data[3], device) if data[3] else None
    #
    #             avg_update_net(agent.act_target, data[4], device) if agent.if_use_act_target else None
    #             avg_update_net(agent.cri_target, data[5], device) if agent.if_use_cri_target else None

    def comm_network_optim(self, agent, learner_id):
        device = self.device_list[learner_id]

        for round_id in range(self.round_num):
            data = get_comm_data(agent)
            data = self.comm_data(data, learner_id, round_id)

            if data:
                avg_update_net(agent.act, data[0], device)
                avg_update_optim(agent.act_optim, data[1], device) if data[1] else None

                avg_update_net(agent.cri, data[2], device) if data[2] else None
                avg_update_optim(agent.cri_optim, data[3], device)

                avg_update_net(agent.act_target, data[4], device) if agent.if_use_act_target else None
                avg_update_net(agent.cri_target, data[5], device) if agent.if_use_cri_target else None

    def run(self, args, comm_eva, comm_exp, learner_id=0):  # 单个learner进程的工作从 evaluator_pipe, worker_pipe, learner_id
        # 所有learner都是共用一个evaluator
        # 每个learner只和它下属的work通信
        # print(f'| os.getpid()={os.getpid()} PipeLearn.run, {learner_id}')
        random_id = learner_id
        torch.manual_seed(int(random_id)%int(1e6))
        args.init_before_training(if_main=learner_id == 0,id=random_id)

        # env = build_env(args.env, if_print=False)
        if_on_policy = args.if_on_policy

        '''init Agent'''
        agent = args.agent
        agent.init(args.net_dim, args.env_state_dim, args.env_action_dim,
                   args.learning_rate, args.if_per_or_gae, args.env_num, learner_id)
        agent.save_or_load_agent(args.cwd, if_save=False)

        '''init ReplayBuffer'''
        if if_on_policy:
            buffer = list()

            def update_buffer(_trajectory_lists):  # on-policy
                # _trajectory_lists = ((ten_state, ten_action, ten_noise, ten_reward, ten_mask), ...)
                _trajectory_lists = list(map(list, zip(*_trajectory_lists)))
                buffer[:] = [torch.cat(tensors, dim=0) for tensors in _trajectory_lists]

                _steps = buffer[3].shape[0]  # buffer[3] = ary_reward
                _r_exp = buffer[3].mean().item()  # buffer[3] = ary_reward
                return _steps, _r_exp

        else:
            buffer_num = args.worker_num * args.env_num
            if self.learner_num > 1:
                buffer_num *= 2

            buffer = ReplayBufferMP(max_len=args.max_memo, state_dim=args.env_state_dim,
                                    action_dim=1 if args.if_discrete else args.env_action_dim,
                                    if_use_per=args.if_per_or_gae,
                                    buffer_num=buffer_num, gpu_id=learner_id)

            def update_buffer(_trajectory_lists):  # off-policy
                # trajectory_lists = ((ten_state, ten_other), ...)
                _trajectory_lists = sum(_trajectory_lists, list())  # list2d.flatten()

                _steps = 0
                _r_exp = 0
                for idx_buffer in range(buffer.worker_num):
                    ten_state, ten_other = _trajectory_lists[idx_buffer]
                    buffer.buffers[idx_buffer].extend_buffer(ten_state, ten_other)

                    _steps += ten_other.shape[0]
                    _r_exp += ten_other[:, 0].sum()  # other = (reward, mask, *action)
                _r_exp /= _steps
                return _steps, _r_exp

            buffer.save_or_load_history(args.cwd, if_save=False)

        '''start training'''
        cwd = args.cwd
        batch_size = args.batch_size
        repeat_times = args.repeat_times
        soft_update_tau = args.soft_update_tau
        del args

        if_train = True
        while if_train:
            # print("ok3")
            trajectory_lists = comm_exp.explore(agent)  # 调的是worker env主进程的explore 接收总结后的轨迹数据(所有worker探索后的轨迹集合)
            # print("ok4")
            if self.learner_num > 1:
                data = self.comm_data(trajectory_lists, learner_id, round_id=-1)
                # data为worker进程采集到的轨迹数据 将其加入到自己的轨迹buffer中
                trajectory_lists.extend(data)
            # print("ok5")
            # print("learner"+str(learner_id)+" update")
            steps, r_exp = update_buffer(trajectory_lists)  # 将数据按类型分组到buffer中 每个learner进程有属于它自己的buffer
            # print("ok6")
            logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)  # 更新网络
            # print("ok7")
            # print("learner"+str(learner_id)+" update finish")
            if self.learner_num > 1:
                self.comm_network_optim(agent, learner_id)
                # print("ok8")
                # 将当前learner训练的网络和优化器的一样以comm_data发给 其他learner进程
                # 并且等待接收其他进程发来的网络和优化器 并将自己的网络和优化器 与 其他进程发来的网络和优化器 参数 加权平局作为自己网络和优化器的新参数
            if comm_eva:
                
                # evaluate pipe中会进行 是否要继续训练的判断
                # if_train = comm_eva.evaluate_and_save(agent.act, steps, r_exp, logging_tuple, if_train)
                if_train, if_save = comm_eva.evaluate_and_save_mp(agent.act, steps, r_exp, logging_tuple)
            random_id += 1
            random.seed(int(random_id)%int(1e6))
            np.random.seed(int(random_id)%int(1e6))
            # torch.manual_seed(int(random_id)%int(1e6))
            # print("ok9")
        agent.save_or_load_agent(cwd, if_save=True)  # 一个learner进程会保存一个网络
        if not if_on_policy:
            print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
            buffer.save_or_load_history(cwd, if_save=True)


class EvaluatorPipe:  # 验证进程的pipe 只会存在一个验证进程
    def __init__(self):
        super().__init__()
        # 最后一个learner进程通信
        self.pipe0, self.pipe1 = mp.Pipe()  # 管道的两端

    # def evaluate_and_save(self, agent_act, steps, r_exp, logging_tuple, if_train):
    #     if self.pipe1.poll():  # if_evaluator_idle
    #         if_train = self.pipe1.recv()  # 环境反馈是否结束
    #
    #         act_cpu_dict = {k: v.cpu() for k, v in agent_act.state_dict().items()}
    #     else:
    #         act_cpu_dict = None
    #
    #     self.pipe1.send((act_cpu_dict, steps, r_exp, logging_tuple))  # 主进程将
    #     return if_train

    def evaluate_and_save_mp(self, agent_act, steps, r_exp, logging_tuple):
        if self.pipe1.poll():  # if_evaluator_idle
            if_train, if_save = self.pipe1.recv()
            act_cpu_dict = {k: v.cpu() for k, v in agent_act.state_dict().items()}
        else:
            if_train, if_save = True, False
            act_cpu_dict = None
 
        self.pipe1.send((act_cpu_dict, steps, r_exp, logging_tuple))
        return if_train, if_save

    def run(self, args, agent_id):
        
        # print(f'| os.getpid()={os.getpid()} PipeEvaluate.run {agent_id}')
        random_id = agent_id
        torch.manual_seed(int(random_id)%int(1e6))
        args.init_before_training(if_main=False,id=agent_id)

        '''init: Agent'''
        # env = build_env(args.env, if_print=False)
        env = build_arena_env(args, env_id=0)  # 用自己的ros环境 为验证专门开一个新环境 是否有点不值得

        agent = args.agent
        agent.init(args.net_dim, args.env_state_dim, args.env_action_dim, args.learning_rate,
                   args.if_per_or_gae, args.env_num, agent_id=args.eval_device_id)
        agent.save_or_load_agent(args.cwd, if_save=False)

        act_cpu = agent.act
        act_cpu.eval()  # 调整模型到验证模式
        [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]
        
        '''init Evaluator'''
        # eval_env = args.eval_env if args.eval_env else build_env(env, if_print=False)
        eval_env = env
        evaluator = Evaluator(args.cwd, agent_id, agent.device, eval_env,
                              args.eval_gap, args.eval_times1, args.eval_times2)
        # evaluator.save_or_load_recoder(if_save=False)
        del agent
        # del env

        '''loop'''
        cwd = args.cwd
        break_step = args.break_step
        if_allow_break = args.if_allow_break
        del args

        if_save = False
        if_train = True
        if_reach_goal = False

        with torch.no_grad():
            while if_train:
                # eval_env.pauseSim()  # 暂停仿真环境 因为验证只在规定的时间差内才进行一次 在这么长的时间内一直开着仿真环境 浪费算力

                act_cpu_dict, steps, r_exp, logging_tuple = self.pipe0.recv()  # 等待接收learner传来的训练后的 policy
                # eval_env.unpauseSim()


                if act_cpu_dict:
                    act_cpu.load_state_dict(act_cpu_dict)  # 更新policy网络
                    # 注意这里的evaluator是普通的evaluator而不是这个多进程版本的evaluator
                    if_reach_goal, if_save = evaluator.evaluate_and_save(act_cpu, steps, r_exp, logging_tuple)
                else:
                    evaluator.total_step += steps

                if_train = not ((if_allow_break and if_reach_goal)
                                or evaluator.total_step > break_step
                                or os.path.exists(f'{cwd}/stop'))
                # self.pipe0.send(if_train)  # 发送 当前是否可结束训练的标识符 给learner
                self.pipe0.send((if_train, if_save))
                random_id += 1
                random.seed(int(random_id)%int(1e6))
                np.random.seed(int(random_id)%int(1e6))
                # torch.manual_seed(int(random_id)%int(1e6))

        print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')
        evaluator.save_or_load_recoder(if_save=True)


## 此类暂时没有修改 要用的话记得改
class VectorEnvPipe:  # 多环境多进程的pipe
    def __init__(self, args):
        self.env_num = args.env_num
        # 用于单个环境与VectorEnv主进程通信 发送单个环境反馈状态 接收单个环境要做的动作
        self.pipes = [mp.Pipe() for _ in range(self.env_num)]
        # 用于VectorEnv主进程和单个env进程通信 发送每个环境要做的动作 接收每个环境反馈的状态并总结
        self.pipe0s = [pipe[0] for pipe in self.pipes]  # 上面pipes中每个管道的一段 另一端pipe1没有写成成员参数而是写在函数中pipe1

        env = build_env(args.eval_env)  # 只是为了获得环境的参数 在得到参数后将环境对象删除
        self.max_step = env.max_step
        self.env_name = env.env_name
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.action_max = env.action_max
        self.if_discrete = env.if_discrete
        self.target_return = env.target_return
        del env

        self.process = list()
        for env_id in range(args.env_num):
            self.process.append(mp.Process(target=self.run, args=(args, env_id)))  # 每一个run会创建一个env
            args.random_seed += 1  # set different for each env 每个环境的随机种子都不一样 --- 很重要
        # [p.start() for p in self.process]

    ######## 多环境进程总结reset step函数
    # 对外是使用这两个函数的 
    # step函数接收learn对多个环境的指令 再分开来一一向各个环境分发其对应的指令  send action
    # step函数等待每个环境在act后saras 并做总结作为step的返回值
    # reset是对所有环境全部reset 一般只在初始化时使用
    def reset(self):
        vec_state = [pipe0.recv() for pipe0 in self.pipe0s]  # 等待pipe1发来 reset后的state
        return vec_state

    def step(self, vec_action):  # pipe0_step 这个step和下面  env.step(action) 不是一个函数
        for i in range(self.env_num):
            self.pipe0s[i].send(vec_action[i])  # 发送要做的动作
        return [pipe0.recv() for pipe0 in self.pipe0s]  # list of (state, reward, done) 等待pipe1发来 step后的sarsa

    ######## 单个环境进程
    def run(self, args, env_id):  # 单个环境进程
        np.random.seed(args.random_seed)

        # env = build_env(args.eval_env, if_print=False)  # 这里写自己的环境
        env = build_arena_env(args, env_id=0)  # 用自己的ros环境 为验证专门开一个新环境 是否有点不值得
        pipe1 = self.pipes[env_id][1]
        del args

        state = env.reset()
        pipe1.send(state)

        while True:
            action = pipe1.recv()  # 多环境进程总结step发送的action
            state, reward, done, _ = env.step(action)  # 在一个环境中做step
            pipe1.send((env.reset() if done else state, reward, done))  # 将step后的sara发给总结step

    # def check(self):
    #     vec_state = self.reset()
    #     ten_state = np.array(vec_state)
    #     print(ten_state.shape)
    #
    #     vec_action = np.array(((0.0, 1.0, 0.0),
    #                            (0.0, 0.5, 0.0),
    #                            (0.0, 0.1, 0.0),))[:self.env_num]
    #     assert self.env_num <= 3
    #
    #     trajectory_list = list()
    #     for _ in range(8):
    #         s_r_d_list = self.step(vec_action)
    #         ten_state = np.array([s_r_d[0] for s_r_d in s_r_d_list])
    #         print(ten_state.shape)
    #         trajectory_list.append(s_r_d_list)
    #
    #     trajectory_list = list(map(list, zip(*trajectory_list)))  # 2D-list transpose
    #     print('| shape of trajectory_list:', len(trajectory_list), len(trajectory_list[0]))


def train_and_evaluate_mp(args, agent_id=0):  # 不仅仅多进程探索采集轨迹 并且多GPU来加速训练
    """Don't set method='fork' when send tensor in GPU"""
    torch.multiprocessing.set_start_method(method='forkserver', force=True)  # method in {'spawn', 'forkserver'}
    # mp.set_start_method(method='spawn',force=True)
    process = list()

    '''learner'''
    learner_num = get_num_learner(args.visible_gpu)  # 根据传入的GPU数量来决定弄几个 learner
    learner_pipe = LearnerPipe(learner_num)
    # 一个learner下分配多个Worker进程加速探索
    # 一个worker下又有多个env
    for learner_id in range(learner_num):
        '''evaluator'''
        # 将evaluator放在for里并不是说有多个 evaluator 而是只在在添加最后一个learner时顺便加入一个evaluator
        if learner_id == learner_num - 1:
            evaluator_pipe = EvaluatorPipe()
            process.append(mp.Process(target=evaluator_pipe.run, args=(args, agent_id)))  # args中的那个环境是传给evaluator的
        else:
            evaluator_pipe = None

        '''explorer'''
        worker_pipe = WorkerPipe(args.env_num, args.worker_num)
        # 一个learner对应多个(args.worker_num个)worker帮助其探索

        for worker_id in range(args.worker_num):
            if args.env_num == 1:
                env_pipe = None
            else:
                env_pipe = VectorEnvPipe(args)  # 一个worker给分配args.env_num个环境
                process.extend(env_pipe.process)
            process.append(mp.Process(target=worker_pipe.run, args=(args, env_pipe, worker_id, learner_id)))
        # 只有最后一个learner会拥有evaluator_pipe 其他learner拥有的都是None
        process.append(mp.Process(target=learner_pipe.run, args=(args, evaluator_pipe, worker_pipe, learner_id)))
        # 两个learner 每个learner下2个worker 每个worker只在1个环境下工作
    
    [(p.start(), time.sleep(0.3)) for p in process]
    print("process start")
    process[-1].join()
    process_safely_terminate(process)


"""Utils"""


def get_num_learner(visible_gpu):  # 传入的visible_gpu 应该是个长字符串 比如 '0,1' 表示用两个gpu
    assert isinstance(visible_gpu, str)  # visible_gpu may in {'0', '1', '1,', '1,2', '1,2,'}
    visible_gpu = eval(visible_gpu)
    num_learner = 1 if isinstance(visible_gpu, int) else len(visible_gpu)
    return num_learner


def process_safely_terminate(process):
    for p in process:
        try:
            p.kill()
        except OSError as e:
            print(e)
            pass


def get_optim_parameters(optim):  # for avg_update_optim()
    params_list = list()
    for params_dict in optim.state_dict()['state'].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list


def avg_update_optim(dst_optim, src_optim_param, device):
    for dst, src in zip(get_optim_parameters(dst_optim), src_optim_param):
        dst.data.copy_((dst.data + src.data.to(device)) * 0.5)
        # dst.data.copy_(src.data * tau + dst.data * (1 - tau))


def avg_update_net(dst_net, src_net_param, device):
    for dst, src in zip(dst_net.parameters(), src_net_param):
        dst.data.copy_((dst.data + src.data.to(device)) * 0.5)
