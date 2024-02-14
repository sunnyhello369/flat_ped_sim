import json
import numpy as np
from math import sqrt
class RobotStatistics:
    def __init__(self,obs_num = 10):
        self.filepath = "D:/RL/_1repertory_log/230616_data/data_analyze_teb_"+str(obs_num)+"d_eval.json"
        self.start_pos_dis_threshold = 0.2
        self.is_eps_finish = {i: False for i in range(1, 1001)}
        self.data = self.load_data()
        self.eps_data = self.group_by_eps()
        self.finished_eps_data = self.get_finished_eps()
        print(len(self.finished_eps_data))
        

    def load_data(self):
        with open(self.filepath, 'r') as f:
            data = [json.loads(line) for line in f]
        return data

    def process_eps_data(self, eps_data):
        processed_data = []
        start_pos = eps_data[0]['start_pos']
        for entry in eps_data:
            distance = self.distance(entry['position'], start_pos)
            if distance < self.start_pos_dis_threshold:
                processed_data = eps_data[eps_data.index(entry):]
                break
        return processed_data

    def group_by_eps(self):
        eps_data = {}
        for entry in self.data:
            eps = entry['eps']
            if self.is_eps_finish[eps]:
                continue
            if eps not in eps_data:
                eps_data[eps] = []
            eps_data[eps].append(entry)
            
            if entry['success'] or entry['timeout'] or entry['collide']:
                self.is_eps_finish[eps] = True

        for eps,data in eps_data.items():
            eps_data[eps] = self.process_eps_data(data)
            # Check if the position difference is less than threshold, then append
            # if self.distance(entry['position'], entry['start_pos']) < self.start_pos_dis_threshold:
            #     eps_data[eps].append(entry)

        eps_data = {eps:data for eps,data in eps_data.items() if data}

        return eps_data

    def get_finished_eps(self):
        return {eps: data for eps, data in self.eps_data.items() if data[-1]['success'] or data[-1]['timeout'] or data[-1]['collide']}

    def success_rate(self):
        successful_eps = sum(1 for eps in self.finished_eps_data.values() if eps[-1]['success'] and len(eps) > 0) 
        return successful_eps / len(self.finished_eps_data)

    def collide_rate(self):
        collided_eps = sum(1 for eps in self.finished_eps_data.values() if eps[-1]['collide'] and len(eps) > 0)
        return collided_eps / len(self.finished_eps_data)

    def timeout_rate(self):
        timeout_eps = sum(1 for eps in self.finished_eps_data.values() if eps[-1]['timeout'] and len(eps) > 0)
        return timeout_eps / len(self.finished_eps_data)

    def mean_and_std(self, key):
        values = [(eps[-1][key]-eps[0][key]) for eps in self.finished_eps_data.values() if eps[-1]['success'] and len(eps) > 0]
        return np.mean(values), np.std(values)

    def distance(self, pos1, pos2):
        return sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)

    def mean_and_std_distance(self):
        distances = []
        for eps in self.finished_eps_data.values():
            if eps[-1]['success']  and len(eps) > 0:
                distance = sum(self.distance(pos1['position'], pos2['position']) 
                               for pos1, pos2 in zip(eps, eps[1:]))
                distances.append(distance)
        return np.mean(distances), np.std(distances)

    def mean_and_std_velocity(self):
        velocities = []
        for eps in self.finished_eps_data.values():
            if eps[-1]['success']  and len(eps) > 0:
                velocity = sum(sqrt(vel['velocity'][0]**2 + vel['velocity'][1]**2) for vel in eps) / len(eps)
                velocities.append(velocity)
        return np.mean(velocities), np.std(velocities)
    
    def get_eps_stats(self, eps):
        if eps not in self.eps_data:
            return None

        eps_data = self.eps_data[eps]
        success = eps_data[-1]['success']
        time = eps_data[-1]['time']
        
        distance = sum(self.distance(pos1['position'], pos2['position']) 
                       for pos1, pos2 in zip(eps_data, eps_data[1:]))
        
        velocity = sum(sqrt(vel['velocity'][0]**2 + vel['velocity'][1]**2) for vel in eps_data) / len(eps_data)
        
        return success,time, distance, velocity


if __name__ == '__main__':
    stats = RobotStatistics(obs_num = 30)
    print('Success rate:', stats.success_rate())
    print('Collide rate:', stats.collide_rate())
    print('Timeout rate:', stats.timeout_rate())
    print('Time mean and std:', stats.mean_and_std('time'))
    print('Distance mean and std:', stats.mean_and_std_distance())
    print('Velocity mean and std:', stats.mean_and_std_velocity())

# 230608
# import json
# import numpy as np
# from math import sqrt
# class RobotStatistics:
#     def __init__(self):
#         self.filepath = '/home/chen/desire_10086/flat_ped_sim/src/d86env/result_model/result_model_teb_10d/data_analyze_teb_10d_eval.json'
#         self.data = self.load_data()
#         self.eps_data = self.group_by_eps()
#         self.finished_eps_data = self.get_finished_eps()

#     def load_data(self):
#         with open(self.filepath, 'r') as f:
#             data = [json.loads(line) for line in f]
#         return data

#     def group_by_eps(self):
#         eps_data = {}
#         for entry in self.data:
#             eps = entry['eps']
#             if eps not in eps_data:
#                 eps_data[eps] = []
#             eps_data[eps].append(entry)
#         return eps_data

#     def get_finished_eps(self):
#         return {eps: data for eps, data in self.eps_data.items() if data[-1]['success'] or data[-1]['timeout'] or data[-1]['collide']}

#     def success_rate(self):
#         successful_eps = sum(1 for eps in self.finished_eps_data.values() if eps[-1]['success'])
#         return successful_eps / len(self.finished_eps_data)

#     def collide_rate(self):
#         collided_eps = sum(1 for eps in self.finished_eps_data.values() if eps[-1]['collide'])
#         return collided_eps / len(self.finished_eps_data)

#     def timeout_rate(self):
#         timeout_eps = sum(1 for eps in self.finished_eps_data.values() if eps[-1]['timeout'])
#         return timeout_eps / len(self.finished_eps_data)

#     def mean_and_std(self, key):
#         values = [eps[-1][key] for eps in self.finished_eps_data.values() if eps[-1]['success']]
#         return np.mean(values), np.std(values)

#     def distance(self, pos1, pos2):
#         return sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)

#     def mean_and_std_distance(self):
#         distances = []
#         for eps in self.finished_eps_data.values():
#             if eps[-1]['success']:
#                 distance = sum(self.distance(pos1['position'], pos2['position']) 
#                                for pos1, pos2 in zip(eps, eps[1:]))
#                 distances.append(distance)
#         return np.mean(distances), np.std(distances)

#     def mean_and_std_velocity(self):
#         velocities = []
#         for eps in self.finished_eps_data.values():
#             if eps[-1]['success']:
#                 velocity = sum(sqrt(vel['velocity'][0]**2 + vel['velocity'][1]**2) for vel in eps) / len(eps)
#                 velocities.append(velocity)
#         return np.mean(velocities), np.std(velocities)
    
#     def get_eps_stats(self, eps):
#         if eps not in self.eps_data:
#             return None

#         eps_data = self.eps_data[eps]
#         success = eps_data[-1]['success']
#         time = eps_data[-1]['time']
        
#         distance = sum(self.distance(pos1['position'], pos2['position']) 
#                        for pos1, pos2 in zip(eps_data, eps_data[1:]))
        
#         velocity = sum(sqrt(vel['velocity'][0]**2 + vel['velocity'][1]**2) for vel in eps_data) / len(eps_data)
        
#         return success,time, distance, velocity


# if __name__ == '__main__':
#     stats = RobotStatistics()
#     print('Success rate:', stats.success_rate())
#     print('Collide rate:', stats.collide_rate())
#     print('Timeout rate:', stats.timeout_rate())
#     print('Time mean and std:', stats.mean_and_std('time'))
#     print('Distance mean and std:', stats.mean_and_std_distance())
#     print('Velocity mean and std:', stats.mean_and_std_velocity())

# import json
# import numpy as np
# from math import sqrt
# class RobotStatistics:
#     def __init__(self, filepath):
#         self.filepath = filepath
#         self.data = self.load_data()
#         self.eps_data = self.group_by_eps()
#         self.finished_eps_data = self.get_finished_eps()

#     def load_data(self):
#         with open(self.filepath, 'r') as f:
#             data = [json.loads(line) for line in f]
#         return data

#     def group_by_eps(self):
#         eps_data = {}
#         for entry in self.data:
#             eps = entry['eps']
#             if eps not in eps_data:
#                 eps_data[eps] = []
#             eps_data[eps].append(entry)
#         return eps_data

#     def get_finished_eps(self):
#         return {eps: data for eps, data in self.eps_data.items() if data[-1]['success'] or data[-1]['timeout'] or data[-1]['collide']}

#     def success_rate(self):
#         successful_eps = sum(1 for eps in self.finished_eps_data.values() if eps[-1]['success'])
#         return successful_eps / len(self.finished_eps_data)

#     def collide_rate(self):
#         collided_eps = sum(1 for eps in self.finished_eps_data.values() if eps[-1]['collide'])
#         return collided_eps / len(self.finished_eps_data)

#     def timeout_rate(self):
#         timeout_eps = sum(1 for eps in self.finished_eps_data.values() if eps[-1]['timeout'])
#         return timeout_eps / len(self.finished_eps_data)

#     def mean_and_std(self, key):
#         values = [eps[-1][key] for eps in self.finished_eps_data.values() if eps[-1]['success']]
#         return np.mean(values), np.std(values)

#     def distance(self, pos1, pos2):
#         return sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)

#     def mean_and_std_distance(self):
#         distances = []
#         for eps in self.finished_eps_data.values():
#             if eps[-1]['success']:
#                 distance = sum(self.distance(pos1['position'], pos2['position']) 
#                                for pos1, pos2 in zip(eps, eps[1:]))
#                 distances.append(distance)
#         return np.mean(distances), np.std(distances)

#     def mean_and_std_velocity(self):
#         velocities = []
#         for eps in self.finished_eps_data.values():
#             if eps[-1]['success']:
#                 velocity = sum(sqrt(vel['velocity'][0]**2 + vel['velocity'][1]**2) for vel in eps) / len(eps)
#                 velocities.append(velocity)
#         return np.mean(velocities), np.std(velocities)

# if __name__ == '__main__':
#     stats = RobotStatistics('/home/chen/desire_10086/flat_ped_sim/src/data_collected_teb_scenarios_30d_eval.json')
#     print("data_collected_teb_scenarios_30d_eval")
#     print('Success rate:', stats.success_rate())
#     print('Collide rate:', stats.collide_rate())
#     print('Timeout rate:', stats.timeout_rate())
#     print('Time mean and std:', stats.mean_and_std('time'))
#     print('Distance mean and std:', stats.mean_and_std_distance())
#     print('Velocity mean and std:', stats.mean_and_std_velocity())