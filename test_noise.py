import numpy as np

lidar = np.random.uniform(0, 30, 50)
# Choose sensor quality
k = 0.02  # good LiDAR

# Add proportional Gaussian noise
noisy_lidar = lidar + np.random.normal(0, k*lidar)
noisy_lidar = np.clip(noisy_lidar, 0, 30)
# not_full = np.where(lidar < 20)[0]
# print(lidar)
# print(not_full)
# print(type(not_full))
# # print(len(not_full))
# # # print(lidar[not_full])
# indices = np.random.choice(len(not_full), size=5, replace=False)
#
# print(indices)
# print(not_full[indices])
#
# lidar[not_full[indices]] = 30
# print(lidar)

reading = np.full(108,1)
dis

