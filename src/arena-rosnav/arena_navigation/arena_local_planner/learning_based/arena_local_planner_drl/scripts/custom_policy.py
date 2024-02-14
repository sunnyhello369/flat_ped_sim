import imp
import os
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym, rospy
import rospkg
import torch as th
import yaml
import numpy as np
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

""" 
_RS: Robot state size - placeholder for robot related inputs to the NN
_L: Number of laser beams - placeholder for the laser beam data 
"""
_RS = 2  # robot state size
robot_model = rospy.get_param("model")
ROBOT_SETTING_PATH = rospkg.RosPack().get_path("simulator_setup")
yaml_ROBOT_SETTING_PATH = os.path.join(
    ROBOT_SETTING_PATH, "robot", f"{robot_model}.model.yaml"
)

with open(yaml_ROBOT_SETTING_PATH, "r") as fd:
    robot_data = yaml.safe_load(fd)
    for plugin in robot_data["plugins"]:
        if plugin["type"] == "Laser":
            laser_angle_min = plugin["angle"]["min"]
            laser_angle_max = plugin["angle"]["max"]
            laser_angle_increment = plugin["angle"]["increment"]
            _L = int(
                round(
                    (laser_angle_max - laser_angle_min) / laser_angle_increment
                )
                + 1
            )  # num of laser beams
            break


class MLP_ARENA2D(nn.Module):
    """
    Custom Multilayer Perceptron for policy and value function.
    Architecture was taken as reference from: https://github.com/ignc-research/arena2D/tree/master/arena2d-agents.
    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 32,
        last_layer_dim_vf: int = 32,
    ):
        super(MLP_ARENA2D, self).__init__()

        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Body network
        self.body_net = nn.Sequential(
            nn.Linear(_L + _RS, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU(),
        )

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        body_x = self.body_net(features)
        return self.policy_net(body_x), self.value_net(body_x)


class MLP_ARENA2D_POLICY(ActorCriticPolicy):
    """
    Policy using the custom Multilayer Perceptron.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs,
    ):
        super(MLP_ARENA2D_POLICY, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
        # Enable orthogonal initialization
        self.ortho_init = True

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MLP_ARENA2D(64)


class AGENT_1(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network to serve as feature extractor ahead of the policy and value network.
    Architecture was taken as reference from: https://arxiv.org/abs/1808.03841
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, observation_space: gym.spaces.Box, features_dim: int = 128
    ):
        super(AGENT_1, self).__init__(observation_space, features_dim + _RS)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # tensor_forward = th.as_tensor(observation_space.sample()[None]).float()
            tensor_forward = th.randn(1, 1, _L)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc_1 = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor),
            extracted features by the network
        """
        laser_scan = th.unsqueeze(observations[:, :-_RS], 1)
        robot_state = observations[:, -_RS:]

        extracted_features = self.fc_1(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_drl_local_planner: (dict)
"""
policy_kwargs_agent_1 = dict(
    features_extractor_class=AGENT_1,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[dict(vf=[64], pi=[64])],
    activation_fn=th.nn.ReLU,
)


class AGENT_2(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network to serve as feature extractor ahead of the policy and value network.
    Architecture was taken as reference from: https://arxiv.org/abs/1808.03841
    (DRL_LOCAL_PLANNER)
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, observation_space: gym.spaces.Box, features_dim: int = 128
    ):
        super(AGENT_2, self).__init__(observation_space, features_dim + _RS)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # tensor_forward = th.as_tensor(observation_space.sample()[None]).float()
            tensor_forward = th.randn(1, 1, _L)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc_1 = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor),
            extracted features by the network
        """
        laser_scan = th.unsqueeze(observations[:, :-_RS], 1)
        robot_state = observations[:, -_RS:]

        extracted_features = self.fc_1(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_drl_local_planner: (dict)
"""
policy_kwargs_agent_2 = dict(
    features_extractor_class=AGENT_2,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[dict(vf=[128], pi=[128])],
    activation_fn=th.nn.ReLU,
)


class AGENT_3(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network to serve as feature extractor ahead of the policy and value network.
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, observation_space: gym.spaces.Box, features_dim: int = 128
    ):
        super(AGENT_3, self).__init__(observation_space, features_dim + _RS)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # tensor_forward = th.as_tensor(observation_space.sample()[None]).float()
            tensor_forward = th.randn(1, 1, _L)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc_1 = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
        )

        self.fc_2 = nn.Sequential(nn.Linear(256, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor),
            extracted features by the network
        """
        laser_scan = th.unsqueeze(observations[:, :-_RS], 1)
        robot_state = observations[:, -_RS:]

        extracted_features = self.fc_2(self.fc_1(self.cnn(laser_scan)))
        # return self.fc_2(features)
        return th.cat((extracted_features, robot_state), 1)


"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_drl_local_planner: (dict)
"""
policy_kwargs_agent_3 = dict(
    features_extractor_class=AGENT_3,
    features_extractor_kwargs=dict(features_dim=(128)),
    net_arch=[dict(vf=[64, 64], pi=[64, 64])],
    activation_fn=th.nn.ReLU,
)


class AGENT_4(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.
    Architecture was taken as reference from: https://github.com/ethz-asl/navrep
    (CNN_NAVREP)
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, observation_space: gym.spaces.Box, features_dim: int = 32
    ):
        super(AGENT_4, self).__init__(observation_space, features_dim + _RS)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 9, 4),
            nn.ReLU(),
            nn.Conv1d(64, 128, 6, 4),
            nn.ReLU(),
            nn.Conv1d(128, 256, 4, 4),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, _L)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        laser_scan = th.unsqueeze(observations[:, :-_RS], 1)
        robot_state = observations[:, -_RS:]

        extracted_features = self.fc(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_kwargs_navrep: (dict)
"""
policy_kwargs_agent_4 = dict(
    features_extractor_class=AGENT_4,
    features_extractor_kwargs=dict(features_dim=32),
    net_arch=[dict(vf=[64, 64], pi=[64, 64])],
    activation_fn=th.nn.ReLU,
)


"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_kwargs_navrep: (dict)
"""
policy_kwargs_agent_5 = dict(
    net_arch=[128, dict(pi=[64], vf=[64])], activation_fn=th.nn.ReLU
)

"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_kwargs_navrep: (dict)
"""
policy_kwargs_agent_6 = dict(
    net_arch=[128, 64, dict(pi=[64, 64], vf=[64, 64])], activation_fn=th.nn.ReLU
)

"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_kwargs_navrep: (dict)
"""
policy_kwargs_agent_7 = dict(
    net_arch=[128, 64, 64, dict(pi=[64, 64], vf=[64, 64])],
    activation_fn=th.nn.ReLU,
)

"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_kwargs_navrep: (dict)
"""
policy_kwargs_agent_8 = dict(
    net_arch=[64, 64, 64, dict(pi=[64, 64], vf=[64, 64])],
    activation_fn=th.nn.ReLU,
)

"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_drl_local_planner: (dict)
"""
policy_kwargs_agent_9 = dict(
    features_extractor_class=AGENT_1,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[dict(vf=[64, 64], pi=[64, 64])],
    activation_fn=th.nn.ReLU,
)


class AGENT_10(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, observation_space: gym.spaces.Box, features_dim: int = 32
    ):
        super(AGENT_10, self).__init__(observation_space, features_dim + _RS)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, _L)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        laser_scan = th.unsqueeze(observations[:, :-_RS], 1)
        robot_state = observations[:, -_RS:]

        extracted_features = self.fc(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_kwargs_navrep: (dict)
"""
policy_kwargs_agent_10 = dict(
    features_extractor_class=AGENT_10,
    features_extractor_kwargs=dict(features_dim=512),
    net_arch=[dict(vf=[128], pi=[128])],
    activation_fn=th.nn.ReLU,
)

"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_kwargs_navrep: (dict)
"""
policy_kwargs_agent_11 = dict(
    features_extractor_class=AGENT_10,
    features_extractor_kwargs=dict(features_dim=512),
    net_arch=[dict(vf=[64, 64], pi=[64, 64])],
    activation_fn=th.nn.ReLU,
)

"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_kwargs_navrep: (dict)
"""
policy_kwargs_agent_12 = dict(
    features_extractor_class=AGENT_10,
    features_extractor_kwargs=dict(features_dim=64),
    net_arch=[dict(vf=[64, 64], pi=[64, 64])],
    activation_fn=th.nn.ReLU,
)

"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_kwargs_navrep: (dict)
"""
policy_kwargs_agent_13 = dict(
    net_arch=[64, 64, dict(pi=[64, 32], vf=[64, 32])], activation_fn=th.nn.ReLU
)

"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_kwargs_navrep: (dict)
"""
policy_kwargs_agent_14 = dict(
    net_arch=[64, 64, dict(pi=[64], vf=[64])], activation_fn=th.nn.ReLU
)

"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_kwargs_navrep: (dict)
"""
policy_kwargs_agent_15 = dict(
    net_arch=[64, 64, 64, 64, dict(pi=[64, 64], vf=[64, 64])],
    activation_fn=th.nn.ReLU,
)

"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_kwargs_navrep: (dict)
"""
policy_kwargs_agent_16 = dict(
    net_arch=[64, 64, 64, 64, dict(pi=[64, 64, 64], vf=[64, 64, 64])],
    activation_fn=th.nn.ReLU,
)

"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_kwargs_navrep: (dict)
"""
policy_kwargs_agent_17 = dict(
    features_extractor_class=AGENT_10,
    features_extractor_kwargs=dict(features_dim=64),
    net_arch=[dict(vf=[64, 64, 64], pi=[64, 64, 64])],
    activation_fn=th.nn.ReLU,
)

"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_kwargs_navrep: (dict)
"""
policy_kwargs_agent_18 = dict(
    features_extractor_class=AGENT_4,
    features_extractor_kwargs=dict(features_dim=64),
    net_arch=[dict(vf=[64, 64, 64], pi=[64, 64, 64])],
    activation_fn=th.nn.ReLU,
)

"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_kwargs_navrep: (dict)
"""
policy_kwargs_agent_19 = dict(
    net_arch=[128, 128, 128, dict(pi=[64, 64], vf=[64, 64])],
    activation_fn=th.nn.ReLU,
)

"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_kwargs_navrep: (dict)
"""
policy_kwargs_agent_20 = dict(
    net_arch=[128, 128, 128, 128, dict(pi=[64, 64, 64], vf=[64, 64, 64])],
    activation_fn=th.nn.ReLU,
)


class AGENT_22(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.
    Architecture was taken as reference from: https://github.com/ethz-asl/navrep
    (CNN_NAVREP)
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, observation_space: gym.spaces.Box, features_dim: int = 32
    ):
        super(AGENT_22, self).__init__(observation_space, features_dim + _RS)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 9, 4),
            nn.ReLU(),
            nn.Conv1d(64, 128, 6, 4),
            nn.ReLU(),
            nn.Conv1d(128, 256, 4, 4),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, _L)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc_1 = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
        )

        self.fc_2 = nn.Sequential(nn.Linear(128, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        laser_scan = th.unsqueeze(observations[:, :-_RS], 1)
        robot_state = observations[:, -_RS:]

        extracted_features = self.fc_2(self.fc_1(self.cnn(laser_scan)))
        features = th.cat((extracted_features, robot_state), 1)

        return features


"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.
:constant policy_kwargs_navrep: (dict)
"""
policy_kwargs_agent_22 = dict(
    features_extractor_class=AGENT_22,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[dict(vf=[64, 64, 64], pi=[64, 64, 64])],
    activation_fn=th.nn.ReLU,
)



class AGENT_d86(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.
    Architecture was taken as reference from: https://github.com/ethz-asl/navrep
    (CNN_NAVREP)
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, observation_space: gym.spaces.Box, features_dim: int = 32,
        stack_time:int = 4,
        lidar_num:int = 360,
        rs_num:int = 4
    ):
        super(AGENT_d86, self).__init__(observation_space, features_dim)

        self.stack_time = stack_time
        self.lidar_num = lidar_num
        self.rs_num = rs_num
        self.fc1 = nn.Sequential(
            nn.Linear(stack_time*lidar_num, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim*2),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(lidar_num, features_dim*2),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(features_dim*4+stack_time*rs_num, features_dim*2),
            nn.ReLU(),
            nn.Linear(features_dim*2, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """
        
        # laser_scan = np.array([])
        # robot_state = np.array([])
        # laser_convex = np.array([])

        # for i in range(self.stack_time):
        #     laser_scan = np.append(laser_scan,observations[i*(self.lidar_num+self.rs_num):i*(self.lidar_num+self.rs_num)+self.lidar_num])
        #     robot_state = np.append(robot_state,observations[i*(self.lidar_num+self.rs_num)+self.lidar_num:i*(self.lidar_num+self.rs_num)+self.lidar_num+self.rs_num])
        # laser_convex = np.append(laser_convex,observations[-self.lidar_num:])

        # laser_scan = []
        # robot_state = []
        # laser_convex = []

        # for i in range(self.stack_time):
        #     laser_scan.extend(observations[i*(self.lidar_num+self.rs_num):i*(self.lidar_num+self.rs_num)+self.lidar_num])
        #     robot_state.extend(observations[i*(self.lidar_num+self.rs_num)+self.lidar_num:i*(self.lidar_num+self.rs_num)+self.lidar_num+self.rs_num])
        # laser_convex.extend(observations[-self.lidar_num:])

        # obs not one env obs,obs[b:obs shape]
        stack_laser_features = self.fc1(observations[:,:self.stack_time*self.lidar_num])
        convex_features = self.fc2(observations[:,-self.lidar_num:])
        extracted_features = th.cat((stack_laser_features, convex_features,observations[:,self.stack_time*self.lidar_num:self.stack_time*self.lidar_num + self.stack_time*self.rs_num]), 1)
        features = self.fc3(extracted_features)
        
        return features




import torch as th
from gym import spaces
import time

if __name__ == "__main__":


    obs_space = []
    # 观测空间一旦设置后返回给环境，环境会用这个空间初始化网络，且只初始化一次
    # 不能多次重置,所以直接将需要动态改变的观测值 距离和动作中的点云距离设为-1~1 和 0~1
    # 再在collector乘上对应要求的长度
    for i in range(4):
        obs_space.extend([
            spaces.Box(
                    low=0,
                    high=10.,
                    shape=(360,),
                    dtype=np.float32,
                ),
                spaces.Box(
                    low=-np.pi,
                    high=np.pi,
                    shape=(1,),
                    dtype=np.float32,  # dtheta 速度方向与终态直线角 之差
                ),                
                spaces.Box(
                    low=0, high=100., shape=(1,), dtype=np.float32
                ),  # dist2goal 为了更新距离上下限
                spaces.Box(
                    low=2.5,
                    high=2.5,
                    shape=(2,),
                    dtype=np.float32,  # linear vel
                )
        ])
    
    obs_space.append(spaces.Box(
                low=0,
                high=10.,
                shape=(360,),
                dtype=np.float32,
            )
        )

    low = []
    high = []
    for space in obs_space:
        low.extend(space.low.tolist())
        high.extend(space.high.tolist())
    obs_space = spaces.Box(np.array(low).flatten(), np.array(high).flatten())
    robot_state_num = 4*(2 + 2)
    obs = obs_space.sample()
    print(len(obs))
    coder = AGENT_d86(obs_space,features_dim=64)
    start_time = time.time()
    features = coder(obs)
    print(time.time()-start_time)
    print(features.shape)