ps -ef|grep train_mp.py|grep -v grep|cut -c 9-16|xargs kill -9
# ps -ef|grep arena_intermediate_planner|grep -v grep|cut -c 9-16|xargs kill -9
ps -ef|grep forkserver|grep -v grep|cut -c 9-16|xargs kill -9
