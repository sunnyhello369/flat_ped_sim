#. /home/chen/anaconda3/etc/profile.d/conda.sh
conda activate wzl_d86
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_HOSTNAME=127.0.0.1

# if you 
source /opt/ros/noetic/setup.bash
#source /home/chen/catkin_workspace/install/setup.bash --extend
#source /home/chen/desire_10086/geometry2_ws/devel_isolated/setup.bash
source /home/dmz/flat_ped_sim/devel/setup.bash



# exprot path can cotrol which project you want to set
#:/opt/ros/kinetic/share/ros\

export ROS_PACKAGE_PATH\
=/opt/ros/noetic/share\
:/home/dmz/flat_ped_sim/src

#sudo ln -s /usr/local/cuda-11.0/ /usr/local/cuda
# ORIGINAL_CUDA_HOME=$CUDA_HOME
# ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
# export CUDA_HOME=/usr/local/cuda-11.0
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
