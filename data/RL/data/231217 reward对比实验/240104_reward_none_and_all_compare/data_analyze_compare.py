import json
import numpy as np
from math import sqrt


class RobotStatistics:
    def __init__(self, filepath ,filepath2=None, compare_mode=False):
        self.filepath = filepath
        self.data = self.load_data(self.filepath)
        self.eps_data = self.group_by_eps(self.data)
        self.finished_eps_data = self.get_finished_eps(self.eps_data)
        
        self.time_interval = 0.1  # 每条记录的时间间隔

        self.filepath2 = filepath2
        self.compare_mode = compare_mode
        if self.filepath2:
            self.data2 = self.load_data(self.filepath2)
            self.eps_data2 = self.group_by_eps(self.data2)
            self.finished_eps_data2 = self.get_finished_eps(self.eps_data2)
        else:
            self.finished_eps_data2 = {}

    def load_data(self, filepath):
        with open(filepath, 'r') as f:
            data = [json.loads(line) for line in f]
        return data

    def group_by_eps(self, data):
        eps_data = {}
        for entry in data:
            eps = entry['eps']
            if eps > 1000:
                break
            if eps not in eps_data:
                eps_data[eps] = []
            eps_data[eps].append(entry)
        return eps_data

    def get_finished_eps(self, eps_data):
        return {eps: data for eps, data in eps_data.items() if
                data[-1]['success'] or data[-1]['timeout'] or data[-1]['collide']}

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
        return sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)

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
                velocity = sum(sqrt(vel['velocity'][0] ** 2 + vel['velocity'][1] ** 2) for vel in eps) / len(eps)
                velocities.append(velocity)
        return np.mean(velocities), np.std(velocities)

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
        for eps, eps_data in self.finished_eps_data.items():
            if eps_data[-1]['success'] and (not self.compare_mode or (eps in self.finished_eps_data2 and self.finished_eps_data2[eps][-1]['success'])):
                velocities = [entry['velocity'] for entry in eps_data]
                accels = self.calculate_acceleration(velocities)
                accelerations.extend(accels)
        return np.mean(accelerations), np.std(accelerations)

    def mean_and_std_jerk(self):
        jerks = []
        for eps, eps_data in self.finished_eps_data.items():
            if eps_data[-1]['success'] and (not self.compare_mode or (eps in self.finished_eps_data2 and self.finished_eps_data2[eps][-1]['success'])):
                velocities = [entry['velocity'] for entry in eps_data]
                accels = self.calculate_acceleration(velocities)
                jerks_in_eps = self.calculate_jerk(accels)
                jerks.extend(jerks_in_eps)
        return np.mean(jerks), np.std(jerks)

    def mean_and_std_abs_acceleration(self):
        accelerations = []
        for eps, eps_data in self.finished_eps_data.items():
            if eps_data[-1]['success'] and (not self.compare_mode or (eps in self.finished_eps_data2 and self.finished_eps_data2[eps][-1]['success'])):
                velocities = [entry['velocity'] for entry in eps_data]
                accels = self.calculate_acceleration(velocities)
                accelerations.extend([abs(a) for a in accels])  # 使用绝对值
        return np.mean(accelerations), np.std(accelerations)

    def mean_and_std_abs_jerk(self):
        jerks = []
        for eps, eps_data in self.finished_eps_data.items():
            if eps_data[-1]['success'] and (not self.compare_mode or (eps in self.finished_eps_data2 and self.finished_eps_data2[eps][-1]['success'])):
                velocities = [entry['velocity'] for entry in eps_data]
                accels = self.calculate_acceleration(velocities)
                jerks_in_eps = self.calculate_jerk(accels)
                jerks.extend([abs(j) for j in jerks_in_eps])  # 使用绝对值
        return np.mean(jerks), np.std(jerks)
    
    def total_absolute_acceleration_and_jerk(self):
        total_acceleration = 0
        total_jerk = 0
        for eps, eps_data in self.finished_eps_data.items():
            if eps_data[-1]['success'] and (not self.compare_mode or (eps in self.finished_eps_data2 and self.finished_eps_data2[eps][-1]['success'])):
                velocities = [entry['velocity'] for entry in eps_data]
                accels = self.calculate_acceleration(velocities)
                total_acceleration += sum([abs(a) for a in accels])

                jerks = self.calculate_jerk(accels)
                total_jerk += sum([abs(j) for j in jerks])

        return total_acceleration, total_jerk

    def mean_and_std_of_eps_acceleration_and_jerk(self):
        eps_acceleration_means = []
        eps_jerk_means = []

        for eps, eps_data in self.finished_eps_data.items():
            if eps_data[-1]['success'] and (not self.compare_mode or (eps in self.finished_eps_data2 and self.finished_eps_data2[eps][-1]['success'])):
                velocities = [entry['velocity'] for entry in eps_data]
                accels = self.calculate_acceleration(velocities)
                eps_acceleration_means.append(np.mean([abs(a) for a in accels]))

                jerks = self.calculate_jerk(accels)
                eps_jerk_means.append(np.mean([abs(j) for j in jerks]))

        eps_mean_acceleration,eps_std_acceleration = np.mean(eps_acceleration_means), np.std(eps_acceleration_means)
        eps_mean_jerk,eps_std_jerk = np.mean(eps_jerk_means), np.std(eps_jerk_means)

        return eps_mean_acceleration,eps_std_acceleration,eps_mean_jerk,eps_std_jerk




if __name__ == '__main__':
    
    filepath = r'D:\RL\data\231217 reward对比实验\240104_reward_none_and_all_compare\reward_no_track\reward_test_analyze_no_track_reward_d30.json'
    # filepath2 = r'D:\RL\data\231217 reward对比实验\240104_reward_none_and_all_compare\reward_none\data_collected_analyze_10d.json'
    # filepath2 = r'D:\RL\data\231217 reward对比实验\240104_reward_none_and_all_compare\reward_no_change\reward_test_analyze_no_change_reward_d10.json'
    filepath2 = r'D:\RL\data\231217 reward对比实验\240104_reward_none_and_all_compare\reward_all\reward_test_analyze_all_reward_d30.json'
    compare_mode = True
    
    stats = RobotStatistics(filepath, filepath2, compare_mode)

    print(f"Data Analysis from: {stats.filepath}\n")

    print("== Rates ==")
    print(f"Success Rate: {stats.success_rate() * 100:.1f}%")
    print(f"Collide Rate: {stats.collide_rate() * 100:.1f}%")
    print(f"Timeout Rate: {stats.timeout_rate() * 100:.1f}%\n")


    print("== Time and Distance ==")
    time_mean, time_std = stats.mean_and_std('time')
    print(f"Time - Mean: {time_mean:.1f}, Std Dev: {time_std:.1f}")

    distance_mean, distance_std = stats.mean_and_std_distance()
    print(f"Distance - Mean: {distance_mean:.2f}, Std Dev: {distance_std:.2f}\n")

    print("== Step Velocity, Acceleration, and Jerk ==")
    velocity_mean, velocity_std = stats.mean_and_std_velocity()
    print(f"Velocity - Mean: {velocity_mean:.2f}, Std Dev: {velocity_std:.2f}")

    acceleration_mean, acceleration_std = stats.mean_and_std_acceleration()
    print(f"Acceleration - Mean: {acceleration_mean:.2f}, Std Dev: {acceleration_std:.2f}")

    jerk_mean, jerk_std = stats.mean_and_std_jerk()
    print(f"Jerk - Mean: {jerk_mean:.2f}, Std Dev: {jerk_std:.2f}\n")

    print("== Absolute Values ==")
    abs_acceleration_mean, abs_acceleration_std = stats.mean_and_std_abs_acceleration()
    print(f"Abs Acceleration - Mean: {abs_acceleration_mean:.2f}, Std Dev: {abs_acceleration_std:.2f}")

    abs_jerk_mean, abs_jerk_std = stats.mean_and_std_abs_jerk()
    print(f"Abs Jerk - Mean: {abs_jerk_mean:.2f}, Std Dev: {abs_jerk_std:.2f}")

    total_abs_acceleration, total_abs_jerk = stats.total_absolute_acceleration_and_jerk()
    print(f"Total Abs Acceleration: {total_abs_acceleration:.2f}")
    print(f"Total Abs Jerk: {total_abs_jerk:.2f}\n")

    print("== Episode Acceleration, and Jerk ==")
    eps_acceleration_mean, eps_acceleration_std = stats.mean_and_std_of_eps_acceleration_and_jerk()[0:2]
    print(f"Eps Acceleration - Mean: {eps_acceleration_mean:.2f}, Std Dev: {eps_acceleration_std:.2f}")

    eps_jerk_mean, eps_jerk_std = stats.mean_and_std_of_eps_acceleration_and_jerk()[2:4]
    print(f"Eps Jerk - Mean: {eps_jerk_mean:.2f}, Std Dev: {eps_jerk_std:.2f}")


    