import numpy as np
# import matplotlib.pyplot as plt
#from ruamel.yaml.timestamp import TimeStamp

# from functions_D import *
from matplotlib.patches import Ellipse
# import scipy
# from scipy.linalg import block_diag
import sys


data = np.load('complete_test.npz', allow_pickle=True)
Uf = data['Uf']  # measured forward velocity (odometry)
Ua = data['Ua']  # measured angular velocity (odometry)
time_stamps = data['TimeStamp']
Pose = data['PoseOdom']  # data to be used only for comparison (x(t), y(t), theta(t) of the robot)
ranges = data['noisyRangeData']
angles = data['angles']


print(ranges.shape)
print(angles.shape)
print(np.sum(np.isinf(ranges)))

print(1080*4688 / np.sum(np.isinf(ranges)))
print('in percentage:', 100 / (1080*4688 / np.sum(np.isinf(ranges))))
is_inf_mask = np.isinf(ranges)
finite_ranges = ranges[~is_inf_mask]

maximum_non_inf = np.max(finite_ranges)
# print('shape of finite ranges:', finite_ranges.shape)
print('len of finite ranges:', len(finite_ranges))
print('max range not inf:', maximum_non_inf)

checker = 7
boolean_mask = (finite_ranges > checker)
count = np.sum(boolean_mask)
print(f"number of ranges greater than {checker}: {count}. In percentage: {100 * count / len(finite_ranges)}")


print("--------------------------------------------------")

# for i in ranges:
#     print(i)
print(f'len ranges: {len(ranges[:, 0])}')

ranges = data['noisyRangeData']
angles = data['angles']

ranges = ranges[::6, :]
angles = angles[::6]
print(ranges.shape)
print(angles.shape)
print(np.sum(np.isinf(ranges)))
print(f'len ranges: {ranges.size}')
print(np.sum(np.isinf(ranges[:, 0])))


print(180*4688 / np.sum(np.isinf(ranges)))
print('in percentage:', 100 / (180*4688 / np.sum(np.isinf(ranges))))
is_inf_mask = np.isinf(ranges)
finite_ranges = ranges[~is_inf_mask]

maximum_non_inf = np.max(finite_ranges)
# print('shape of finite ranges:', finite_ranges.shape)
print('len of finite ranges:', len(finite_ranges))
print('max range not inf:', maximum_non_inf)

checker = 7
boolean_mask = (finite_ranges > checker)
count = np.sum(boolean_mask)
print(f"number of ranges greater than {checker}: {count}. In percentage: {100 * count / len(finite_ranges)}")




