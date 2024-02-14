import json
import numpy as np
from math import sqrt
from itertools import islice
class RobotStatistics:
    def __init__(self):
        self.filepath = '/home/dmz/flat_ped_sim/src/d86env/result_model_senario_230612/data_collected_analyze_30d.json'
        self.data = self.load_data()
        self.eps_data = self.group_by_eps()
        self.finished_eps_data = self.get_finished_eps()

    def load_data(self):
        with open(self.filepath, 'r') as f:
            data = [json.loads(line) for line in f]
        return data

    def group_by_eps(self):
        eps_data = {}
        for entry in self.data:
            eps = entry['eps']
            if eps > 1000:
                break
            if eps not in eps_data:
                eps_data[eps] = []
            eps_data[eps].append(entry)
        return eps_data


    def get_finished_eps(self):
        return {eps: data for eps, data in self.eps_data.items() if data[-1]['success'] or data[-1]['timeout'] or data[-1]['collide']}

    def success_rate(self):
        successful_eps = sum(1 for eps in self.finished_eps_data.values() if eps[-1]['success'])
        return successful_eps / len(self.finished_eps_data)

    def collide_rate(self):
        collided_eps = sum(1 for eps in self.finished_eps_data.values() if eps[-1]['collide'])
        return collided_eps / len(self.finished_eps_data)

    def timeout_rate(self):
        timeout_eps = sum(1 for eps in self.finished_eps_data.values() if eps[-1]['timeout'])
        return timeout_eps / len(self.finished_eps_data)

    def mean_and_std(self, key):
        values = [eps[-1][key] for eps in self.finished_eps_data.values() if eps[-1]['success']]
        return np.mean(values), np.std(values)

    def distance(self, pos1, pos2):
        return sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)

    def mean_and_std_distance(self):
        distances = []
        for eps in self.finished_eps_data.values():
            if eps[-1]['success']:
                distance = sum(self.distance(pos1['position'], pos2['position']) 
                               for pos1, pos2 in zip(eps, eps[1:]))
                distances.append(distance)
        return np.mean(distances), np.std(distances)

    def mean_and_std_velocity(self):
        velocities = []
        for eps in self.finished_eps_data.values():
            if eps[-1]['success']:
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
    stats = RobotStatistics()
    print('Success rate:', stats.success_rate())
    print('Collide rate:', stats.collide_rate())
    print('Timeout rate:', stats.timeout_rate())
    print('Time mean and std:', stats.mean_and_std('time'))
    print('Distance mean and std:', stats.mean_and_std_distance())
    print('Velocity mean and std:', stats.mean_and_std_velocity())


# import json
# import numpy as np
# from math import sqrt

# class RobotStatistics:
#     def __init__(self, filepath):
#         self.filepath = filepath
#         self.data = self.load_data()
#         self.eps_data = self.group_by_eps()

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

#     def success_rate(self):
#         successful_eps = sum(1 for eps in self.eps_data.values() if eps[-1]['success'])
#         total_eps = len(self.eps_data)
#         return successful_eps / total_eps

#     def mean_and_std(self, key):
#         values = [eps[-1][key] for eps in self.eps_data.values()]
#         return np.mean(values), np.std(values)

#     def distance(self, pos1, pos2):
#         return sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)

#     def mean_and_std_distance(self):
#         distances = []
#         for eps in self.eps_data.values():
#             distance = sum(self.distance(pos1['position'], pos2['position']) 
#                            for pos1, pos2 in zip(eps, eps[1:]))
#             distances.append(distance)
#         return np.mean(distances), np.std(distances)

#     def mean_and_std_velocity(self):
#         velocities = []
#         for eps in self.eps_data.values():
#             velocity = sum(sqrt(vel['velocity'][0]**2 + vel['velocity'][1]**2) for vel in eps) / len(eps)
#             velocities.append(velocity)
#         return np.mean(velocities), np.std(velocities)


# if __name__ == '__main__':
#     stats = RobotStatistics('/home/dmz/flat_ped_sim/src/data_collected_analyze.json')
#     print('Success rate:', stats.success_rate())
#     print('Time mean and std:', stats.mean_and_std('time'))
#     print('Distance mean and std:', stats.mean_and_std_distance())
#     print('Velocity mean and std:', stats.mean_and_std_velocity())
