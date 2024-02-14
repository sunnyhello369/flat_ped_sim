import numpy as np


# def save_learning_curve(recorder=None, cwd='.', save_title='learning curve', fig_name='plot_learning_curve.jpg'):
#     # if recorder is None:
#     recorder = np.load(f"{cwd}/recorder.npy")

#     recorder = np.array(recorder)
#     steps = recorder[:, 0]  # x-axis is training steps
#     r_avg = recorder[:, 1]
#     r_std = recorder[:, 2]
#     r_exp = recorder[:, 3]
#     obj_c = recorder[:, 4]
#     obj_a = recorder[:, 5]
#     # 6是a_std 定值
#     actor_entropy = recorder[:, 7] # 策略熵 随训练逐渐减小合理
#     print(actor_entropy)
#     actor_advsurr = recorder[:, 8]
#     # self.total_step--[0], r_avg--[1], r_std--[2], r_exp--[3], *log_tuple(critic_LOSS--[4], actor_LOSS--[5])
#     # 希望critic_LOSS接近于0 此loss是critic网络对gae或raw_reward的拟合 希望拟合的越准越好
#     # 希望actor_LOSS越小越好 此loss是-advantage * clamp ratio + policy entropy(因为熵是负的 所以lambda_entropy越大则优化器偏向增大熵)

#     '''plot subplots'''
#     import matplotlib as mpl
#     mpl.use('Agg')
#     """Generating matplotlib graphs without a running X server [duplicate]
#     write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
#     https://stackoverflow.com/a/4935945/9293137
#     """
#     import matplotlib.pyplot as plt
#     fig, axs = plt.subplots(2)

#     '''axs[0]'''
#     ax00 = axs[0]
#     ax00.cla()

#     ax01 = axs[0].twinx()
#     color01 = 'darkcyan'
#     ax01.set_ylabel('Explore AvgReward', color=color01)
#     ax01.plot(steps, r_exp, color=color01, alpha=0.5, )
#     ax01.tick_params(axis='y', labelcolor=color01)

#     color0 = 'lightcoral'
#     ax00.set_ylabel('Episode Return')
#     ax00.plot(steps, r_avg, label='Episode Return', color=color0)
#     ax00.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)
#     ax00.grid()

#     '''axs[1]'''
#     ax10 = axs[1]
#     ax10.cla()

#     ax11 = axs[1].twinx()
#     color11 = 'darkcyan'
#     ax11.set_ylabel('objC', color=color11)
#     ax11.fill_between(steps, obj_c, facecolor=color11, alpha=0.2, )
#     ax11.tick_params(axis='y', labelcolor=color11)

#     color10 = 'royalblue'
#     ax10.set_xlabel('Total Steps')
#     ax10.set_ylabel('objA', color=color10)
#     ax10.plot(steps, obj_a, color=color10)  # label='objA' 就是那个legend 暂时不画
#     ax10.tick_params(axis='y', labelcolor=color10)
#     # 6为其他要画图的东西 但是我们没有 所以其实可以不用画
#     # for plot_i in range(6, recorder.shape[1]):
#     #     other = recorder[:, plot_i]
#     #     ax10.plot(steps, other, label=f'{plot_i}', color='grey', alpha=0.5)
#     for plot_i in range(7, recorder.shape[1]):
#         other = recorder[:, plot_i]
#         ax10.plot(steps, other, label=f'{plot_i}', color='grey', alpha=0.5)
#     ax10.legend()
#     ax10.grid()

#     '''plot save'''
#     plt.title(save_title, y=2.3)  # save_title = f"step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"
#     # plt.savefig(f"{cwd}/{fig_name}")
#     # plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
#     # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()



# # save_learning_curve(recorder=None, cwd='/home/dmz/flat_ped_sim/src/d86env/ElegantRL/08_ok', save_title='learning curve', fig_name='plot_learning_curve.jpg')
# save_learning_curve(recorder=None, cwd='/home/dmz/flat_ped_sim/src/d86env/my_env_AgentPPO', save_title='learning curve', fig_name='plot_learning_curve.jpg')

last_world_action_info = [(6.53236959612089, 6.112976574237773),(6.283226550904352, 6.144816073308753)]

robot = (6.531904697418213, 6.113147735595703) 
goal = (-2.0999999418854713, 10.650000248104334)

_2andgoal = np.arctan2(goal[1] - last_world_action_info[-2][1], goal[0] - last_world_action_info[-2][0])
v_heading = np.arctan2(last_world_action_info[-1][1] - last_world_action_info[-2][1], last_world_action_info[-1][0] - last_world_action_info[-2][0])
print(_2andgoal,v_heading)
# last_world_action_info   [-2] and goal    d 
# 3.0144855712909444      2.657689003585359 0.3567965677055853

