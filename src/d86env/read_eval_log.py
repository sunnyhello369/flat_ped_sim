import numpy as np


data = np.load("/home/chen/desire_10086/flat_ped_sim/src/arena-rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/training_logs/train_eval_log/MLP_B__P__V__relu_2023_01_03__00_43_05/evaluations.npz")


for res in zip(data[ 'timesteps'],data[ 'results'], data['ep_lengths'], data['successes']):
    for r in zip(res[1],res[2],res[3]):
        print("reward:"+str(r[0])+" eps_steps:"+str(r[1])+" is_success:"+str(r[2]))
    # print("timesteps: "+str(res[0])) # 当前总采集步数
    # print("results: "+str(res[1])) # 本次评估得到的奖励值
    # print(" ep_lengths: "+str(res[2])) # 本次评估所走的步长
    # print(" successes: "+str(res[3])) # 成功率
    print("************************************************")

