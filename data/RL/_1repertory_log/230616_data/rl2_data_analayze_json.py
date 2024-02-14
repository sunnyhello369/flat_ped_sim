import json
import numpy as np
from math import sqrt
from itertools import islice
class RobotStatistics:
    def __init__(self,path = "flat_ped_sim_0502_sparse_polar2p_limit/s10d10"):
        self.filepath = "D:/RL/_1repertory_log/230616_data/"+path+"_scence_analyze.json"
        print(self.filepath)
        self.is_eps_finish = {i: False for i in range(1, 1001)}
        self.data = self.load_data()
        self.eps_data = self.group_by_eps()
        self.finished_eps_data = self.get_finished_eps()
        self.time_interval = 0.1

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
            if self.is_eps_finish[eps]:
                continue

            if eps not in eps_data:
                eps_data[eps] = []

            eps_data[eps].append(entry)

            if entry['success'] or entry['timeout'] or entry['collide']:
                self.is_eps_finish[eps] = True

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

    def speed(self, velocity):
        return np.linalg.norm(velocity)

    def calculate_acceleration(self, velocities):
        speeds = [self.speed(velocity) for velocity in velocities]
        accelerations = [(speeds[i + 1] - speeds[i]) / self.time_interval for i in range(len(speeds) - 1)]
        return accelerations

    def calculate_jerk(self, accelerations):
        jerks = [(accelerations[i + 1] - accelerations[i]) / self.time_interval for i in range(len(accelerations) - 1)]
        return jerks

    def mean_and_std_acceleration(self):
        accelerations = []
        for eps in self.finished_eps_data.values():
            if eps[-1]['success']:
                velocities = [entry['velocity'] for entry in eps]
                accels = self.calculate_acceleration(velocities)
                accelerations.extend(accels)
        return np.mean(accelerations), np.std(accelerations)

    def mean_and_std_jerk(self):
        jerks = []
        for eps in self.finished_eps_data.values():
            if eps[-1]['success']:
                velocities = [entry['velocity'] for entry in eps]
                accels = self.calculate_acceleration(velocities)
                jerks_in_eps = self.calculate_jerk(accels)
                jerks.extend(jerks_in_eps)
        return np.mean(jerks), np.std(jerks)


# scence = "s10" 
# scence = "s10d5" 
scence = "s10d10" 
path = ["flat_ped_sim_0502_sparse_polar2p_limit/"+scence,"flat_ped_sim_0430_e2e/"+scence,
        "flat_ped_sim_0424_convex_vel/"+scence,"flat_ped_sim_0506polar1p/"+scence,
        "flat_ped_sim_0510_scan2p/"+scence]

if __name__ == '__main__':
    for i in range(len(path)):
        stats = RobotStatistics(path[i])
        # print(stats.filepath)
        print('Success rate:', stats.success_rate())
        print('Collide rate:', stats.collide_rate())
        print('Timeout rate:', stats.timeout_rate())
        print('Time mean and std:', stats.mean_and_std('time'))
        print('Distance mean and std:', stats.mean_and_std_distance())
        print('Velocity mean and std:', stats.mean_and_std_velocity())
        print('Acceleration mean and std:', stats.mean_and_std_acceleration())
        print('Jerk mean and std:', stats.mean_and_std_jerk())
        print("")
