import os
import time
import torch
import numpy as np

"""[ElegantRL.2021.09.01](https://github.com/AI4Finance-Foundation/ElegantRL)"""


class Evaluator:
    def __init__(self, cwd, agent_id, device, eval_env, eval_gap, eval_times1, eval_times2, ):
        self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
        self.recorder_path = f'{cwd}/recorder.npy'

        self.cwd = cwd
        self.device = device
        self.agent_id = agent_id
        self.eval_env = eval_env
        self.eval_gap = eval_gap
        self.eval_times1 = eval_times1
        self.eval_times2 = eval_times2
        self.target_return = eval_env.target_return

        self.r_max = -np.inf
        self.r_exp_max = -np.inf
        self.eval_time = 0
        self.used_time = 0
        self.total_step = 0
        self.start_time = time.time()
        print(f"{'#' * 80}\n"
              f"{'ID':<3}{'Step':>8}{'maxR':>8} |"
              f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
              f"{'expR':>8}{'objC':>7}{'etc.':>7}")

    def evaluate_and_save(self, act, steps, r_exp, log_tuple) -> (bool, bool):  # 2021-09-09
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time < self.eval_gap:

            if_reach_goal = False
            if_save = False
        else:
            self.eval_time = time.time()

            '''evaluate first time'''

            rewards_steps_list = [get_episode_return_and_step(self.eval_env, act, self.device) for _ in
                                  range(self.eval_times1)]

            r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)

            '''evaluate second time'''
            if r_avg > self.r_max:  # evaluate actor twice to save CPU Usage and keep precision

                rewards_steps_list += [get_episode_return_and_step(self.eval_env, act, self.device)
                                       for _ in range(self.eval_times2 - self.eval_times1)]
                r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)


            '''save the policy network'''
            if_save = r_avg > self.r_max
            if if_save:  # save checkpoint with highest episode return
                self.r_max = r_avg  # update max reward (episode return)

                act_save_path = f'{self.cwd}/actor.pth'
                torch.save(act.state_dict(), act_save_path)  # save policy network in *.pth

                print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |")  # save policy and print

            if r_exp > self.r_exp_max:
                self.r_exp_max = r_exp
                act_save_path = f'{self.cwd}/actor_exp_best.pth'
                torch.save(act.state_dict(), act_save_path)  # save policy network in *.pth


            self.recorder.append((self.total_step, r_avg, r_std, r_exp, *log_tuple))  # update recorder

            '''print some information to Terminal'''
            if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
            if if_reach_goal and self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'ID':<3}{'Step':>8}{'TargetR':>8} |"
                      f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
                      f"{'UsedTime':>8}  ########\n"
                      f"{self.agent_id:<3}{self.total_step:8.2e}{self.target_return:8.2f} |"
                      f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                      f"{self.used_time:>8}  ########")
            

            print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |"
                  f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                  f"{r_exp:8.2f}{''.join(f'{n:7.2f}' for n in log_tuple)}")
            self.draw_plot()
        return if_reach_goal, if_save

    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)
        r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
        r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std

    def save_or_load_recoder(self, if_save):
        if if_save:
            np.save(self.recorder_path, self.recorder)
        elif os.path.exists(self.recorder_path):
            recorder = np.load(self.recorder_path)
            self.recorder = [tuple(i) for i in recorder]  # convert numpy to list
            self.total_step = self.recorder[-1][0]

    def draw_plot(self):
        if len(self.recorder) == 0:
            print("| save_npy_draw_plot() WARNNING: len(self.recorder)==0")
            return None

        np.save(self.recorder_path, self.recorder)

        '''draw plot and save as png'''
        train_time = int(time.time() - self.start_time)
        total_step = int(self.recorder[-1][0])
        save_title = f"step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"

        save_learning_curve(self.recorder, self.cwd, save_title)

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

    for episode_step in range(max_step):
            
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)  # 直接用actor的forward得到tanh后的mean(-1,1)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        # action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        action = torch.sigmoid(a_tensor).detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        # state, reward, done, _ = env.step(np.tanh(action))  # 在step中将(-1,1)乘上action max
        # state, reward, done, _ = env.step(sigmoid(action))
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
        
    episode_return = getattr(env, 'episode_return', episode_return)
    return episode_return, episode_step


def save_learning_curve(recorder=None, cwd='.', save_title='learning curve', fig_name='plot_learning_curve.jpg'):
    if recorder is None:
        recorder = np.load(f"{cwd}/recorder.npy")

    recorder = np.array(recorder)
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    r_exp = recorder[:, 3]
    obj_c = recorder[:, 4]
    obj_a = recorder[:, 5]
    # 6是a_std 定值
    actor_entropy = recorder[:, 7] # 策略熵 随训练逐渐减小合理
    actor_advsurr = recorder[:, 8]
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
    color01 = 'darkcyan'
    ax01.set_ylabel('Explore AvgReward', color=color01)
    ax01.plot(steps, r_exp, color=color01, alpha=0.5, )
    ax01.tick_params(axis='y', labelcolor=color01)

    color0 = 'lightcoral'
    ax00.set_ylabel('Episode Return')
    ax00.plot(steps, r_avg, label='Episode Return', color=color0)
    ax00.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)
    ax00.grid()

    '''axs[1]'''
    ax10 = axs[1]
    ax10.cla()

    ax11 = axs[1].twinx()
    color11 = 'darkcyan'
    ax11.set_ylabel('objC', color=color11)
    ax11.fill_between(steps, obj_c, facecolor=color11, alpha=0.2, )
    ax11.tick_params(axis='y', labelcolor=color11)

    color10 = 'royalblue'
    ax10.set_xlabel('Total Steps')
    ax10.set_ylabel('objA', color=color10)
    ax10.plot(steps, obj_a, color=color10)  # label='objA' 就是那个legend 暂时不画
    ax10.tick_params(axis='y', labelcolor=color10)
    # 6为其他要画图的东西 但是我们没有 所以其实可以不用画
    # for plot_i in range(6, recorder.shape[1]):
    #     other = recorder[:, plot_i]
    #     ax10.plot(steps, other, label=f'{plot_i}', color='grey', alpha=0.5)
    for plot_i in range(7, recorder.shape[1]):
        other = recorder[:, plot_i]
        ax10.plot(steps, other, label=f'{plot_i}', color='grey', alpha=0.5)
    ax10.legend()
    ax10.grid()

    '''plot save'''
    plt.title(save_title, y=2.3)  # save_title = f"step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"
    plt.savefig(f"{cwd}/{fig_name}")
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()
