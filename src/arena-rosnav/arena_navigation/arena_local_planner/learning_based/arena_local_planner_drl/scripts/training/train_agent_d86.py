#!/usr/bin/env python
from typing import Type, Union

import os, sys, rospy, time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy

from rl_agent.model.agent_factory import AgentFactory
from rl_agent.model.base_agent import BaseAgent
from scripts.custom_policy import AGENT_d86

from tools.argsparser import parse_training_args
from tools.custom_mlp_utils import *
from tools.train_agent_utils_d86 import *
from tools.staged_train_callback import InitiateNewTrainStage
# import faulthandler
# faulthandler.enable()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    args, _ = parse_training_args()

    # in debug mode, we emulate multiprocessing on only one process
    # in order to be better able to locate bugs
    if args.debug:
        rospy.init_node("debug_node", disable_signals=False)

    # generate agent name and model specific paths
    AGENT_NAME = get_agent_name(args)
    PATHS = get_paths(AGENT_NAME, args)
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
    wait_for_nodes(with_ns=ns_for_nodes, n_envs=args.n_envs, timeout=5)

    # initialize hyperparameters (save to/ load from json)
    params = initialize_hyperparameters(
        PATHS=PATHS,
        load_target=args.load,
        config_name=args.config,
        n_envs=args.n_envs,
    )

    # instantiate train environment
    # when debug run on one process only
    if not args.debug and ns_for_nodes:
        env = SubprocVecEnv(
            [make_envs(args, ns_for_nodes, i, params=params, PATHS=PATHS) for i in range(args.n_envs)],
            start_method="fork",
        )
    else:
        env = DummyVecEnv([make_envs(args, ns_for_nodes, i, params=params, PATHS=PATHS) for i in range(args.n_envs)])

    # threshold settings for training curriculum
    # type can be either 'succ' or 'rew'
    trainstage_cb = InitiateNewTrainStage(
        n_envs=args.n_envs,
        treshhold_type="succ",
        upper_threshold=0.9,
        lower_threshold=0.7,
        task_mode=params["task_mode"],
        verbose=1,
    )

    # stop training on reward threshold callback
    stoptraining_cb = StopTrainingOnRewardThreshold(
        treshhold_type="succ", threshold=0.95, verbose=1
    )


    # instantiate eval environment
    # take task_manager from first sim (currently evaluation only provided for single process)
    if ns_for_nodes:
        eval_env = DummyVecEnv(
            [
                make_envs(
                    args,
                    ns_for_nodes,
                    0,
                    params=params,
                    PATHS=PATHS,
                    train=False,
                )
            ]
        )
    else:
        eval_env = env

    # try to load most recent vec_normalize obj (contains statistics like moving avg)
    env, eval_env = load_vec_normalize(params, PATHS, env, eval_env)

    # evaluation settings
    # n_eval_episodes: number of episodes to evaluate agent on
    # eval_freq: evaluate the agent every eval_freq train timesteps

    # eval_cb = EvalCallback(
    #     eval_env=eval_env,
    #     train_env=env,
    #     n_eval_episodes=100,
    #     eval_freq=15000,
    #     log_path=PATHS["eval"],
    #     best_model_save_path=PATHS["model"],
    #     deterministic=True,
    #     callback_on_eval_end=trainstage_cb,
    #     callback_on_new_best=stoptraining_cb,
    # )

    eval_cb = EvalCallback(
        eval_env=eval_env,
        train_env=env,
        n_eval_episodes=100,
        eval_freq=200,
        log_path=PATHS["eval"],
        best_model_save_path=PATHS["model"],
        deterministic=True,
        callback_on_eval_end=trainstage_cb,
        callback_on_new_best=stoptraining_cb,
    )

    policy_kwargs_agent_d86 = dict(
        features_extractor_class=AGENT_d86,
        features_extractor_kwargs=dict(features_dim=64,stack_time=3,lidar_num=360,rs_num=4),
        net_arch=[dict(vf=[16], pi=[16])],
        activation_fn=th.nn.ReLU,
    )

    # determine mode
    if args.custom_mlp:
        # custom mlp flag
        # model = PPO(
        #     "MlpPolicy",
        #     env,
        #     policy_kwargs=dict(net_arch=args.net_arch, activation_fn=get_act_fn(args.act_fn)),
        #     gamma=params["gamma"],
        #     n_steps=params["n_steps"],
        #     ent_coef=params["ent_coef"],
        #     learning_rate=params["learning_rate"],
        #     vf_coef=params["vf_coef"],
        #     max_grad_norm=params["max_grad_norm"],
        #     gae_lambda=params["gae_lambda"],
        #     batch_size=params["m_batch_size"],
        #     n_epochs=params["n_epochs"],
        #     clip_range=params["clip_range"],
        #     tensorboard_log=PATHS["tb"],
        #     verbose=1,
        # )
        model = PPO(
            "MlpPolicy", env, 
            policy_kwargs = policy_kwargs_agent_d86, 
            gamma = params['gamma'],            n_steps = params['n_steps'], 
            ent_coef = params['ent_coef'],      learning_rate = params['learning_rate'], 
            vf_coef = params['vf_coef'],        max_grad_norm = params['max_grad_norm'], 
            gae_lambda = params['gae_lambda'],  batch_size = params['m_batch_size'], 
            n_epochs = params['n_epochs'],      clip_range = params['clip_range'], 
            tensorboard_log = PATHS['tb'],      verbose = 1
        )
        print("mlp load ok !!!!!!!!!!!!!!")
    else:
        # load flag
        if os.path.isfile(os.path.join(PATHS["model"], AGENT_NAME + ".zip")):
            model = PPO.load(os.path.join(PATHS["model"], AGENT_NAME), env)
        elif os.path.isfile(os.path.join(PATHS["model"], "best_model.zip")):
            model = PPO.load(os.path.join(PATHS["model"], "best_model"), env)
        update_hyperparam_model(model, PATHS, params, args.n_envs)

    print("load ok !!!!!!!!!!!!!!")
    # set num of timesteps to be generated
    n_timesteps = 20000000 if args.n is None else args.n
    # start training
    start = time.time()
    try:
        model.learn(
            total_timesteps=n_timesteps,
            # callback=eval_cb,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt..")
    # finally:
    # update the timesteps the model has trained in total
    # update_total_timesteps_json(n_timesteps, PATHS)

    model.env.close()
    print(f"Time passed: {time.time()-start}s")
    print("Training script will be terminated")
    sys.exit()


if __name__ == "__main__":
    main()
