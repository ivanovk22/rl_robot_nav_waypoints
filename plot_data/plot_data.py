import numpy as np
import matplotlib.pyplot as plt

# plt.close('all')
data_angles = np.load('data_me_alone_testing.npz', allow_pickle=True)
data = np.load('data_people_moving_2.npz', allow_pickle=True)
Pose = data['Pose']
angles = data_angles['angles']
ranges = data['ranges']
time = data['time']
time_ns = data['time_ns']
target_cmd = data['target_cmd']
filtered_cmd = data['filtered_cmd']
odom_cmd = data['odom_cmd']

min_ran = np.array([])
for r in ranges.T:
    min_ran = np.append(min_ran, min(r))

count = np.sum(min_ran < 0.2)
print(count)





print('pose shape:', Pose.shape)
print('ranges shape:', ranges.shape)
print('angles shape', angles.shape)
print('time shape', time.shape)
print('time ns shape', time_ns.shape)
print('target_cmd shape', target_cmd.shape)
print('filtered_cmd shape', filtered_cmd.shape)
print('odom_cmd shape', odom_cmd.shape)
# print(time[0])
# print(time_ns[0])


N = len(Pose) - 2
print(N)
pose_true = Pose[1:N]
print(len(pose_true))



true_meas = []
for p in range(N - 1):
    ranges_t = ranges[:, p+1]
    for meas_idx in range(0, len(ranges_t)):
        if np.isinf(ranges_t[meas_idx]) == False:
            true_r_x = pose_true[p][0]
            true_r_y = pose_true[p][1]
            true_r_theta = pose_true[p][2]

            mp = ranges_t[meas_idx]
            ma = angles[meas_idx]

            true_l_x = true_r_x + mp * np.cos(true_r_theta + ma)
            true_l_y = true_r_y + mp * np.sin(true_r_theta + ma)
            true_meas.append([true_l_x, true_l_y])


true_meas = np.array(true_meas)
plt.figure()
plt.scatter(true_meas[:, 0], true_meas[:, 1], label='Lidar scans',
            color='blue', marker='o', s=0.3)
plt.plot(pose_true[:,0], pose_true[:,1], color='red', linestyle='--')

plt.legend()
plt.title(r'Lidar scans from the robot pose provided by the odometry')


#
T = time.copy()[3:]
print('T:', len(T))
""" Plots and comparisons """
fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(3, 1, 1)
ax1.plot(T, pose_true[:, 0], label=r'$x_1(t)$')

plt.xlabel("$t$")
plt.ylabel("$x_1(t)$")
plt.title("Robot's X position")
plt.legend(loc='upper right')

ax1 = plt.subplot(3, 1, 2)
ax1.plot(T, pose_true[:, 1], label=r'$x_2(t)$')
# ax1.plot(T, pose_pred[:, 1], label=r'$\hat{x}_2(t)$', color='red', linestyle='--')
# ax1.plot(T, pose_odom_pred[:, 1], label=r'$\hat{x}_{2,odom}(t)$', color='black', linestyle=':')
plt.xlabel("$t$")
plt.ylabel("$x_2(t)$")
plt.title("Robot's Y position$")
plt.legend(loc='upper right')

ax1 = plt.subplot(3, 1, 3)
ax1.plot(T, np.unwrap(pose_true[:, 2]), label=r'$\theta(t)$')

plt.xlabel("$t$ (s)")
plt.ylabel(r"$\theta(t)$ (rad)")
plt.title("Robot Orientation")
plt.legend(loc='upper right')
plt.subplots_adjust(hspace=0.6)



t = np.arange(target_cmd.shape[0])
t = np.divide(t, 4)



print('t:', len(t))
# print(t[-1])
fig,axs = plt.subplots(2,1,figsize = (12,6), sharex=True)

axs[0].plot(t,target_cmd[:, 0], linewidth=2)
axs[0].set_ylabel("Lin Velocity")
axs[0].set_xlabel("Time")
axs[0].set_title("Linear Velocity Target")
axs[0].grid(True)

axs[1].plot(t,target_cmd[:, 1], linewidth=2)
axs[1].set_ylabel("Ang Velocity")
axs[1].set_xlabel("Step")
axs[1].set_title("Angular Velocity Target")
axs[1].grid(True)

for ax in axs:
    ax.tick_params(labelbottom=True)

plt.tight_layout()




t = np.arange(filtered_cmd.shape[0])
t = np.divide(t, 4)
fig,axs = plt.subplots(2,1,figsize = (12,6), sharex=True)

axs[0].plot(t,filtered_cmd[:, 0], linewidth=2)
axs[0].set_ylabel("Lin Velocity")
axs[0].set_xlabel("Step")
axs[0].set_title("Linear Velocity Filtered")
axs[0].grid(True)

axs[1].plot(t,filtered_cmd[:, 1], linewidth=2)
axs[1].set_ylabel("Ang Velocity")
axs[1].set_xlabel("Step")
axs[1].set_title("Angular Velocity Filtered")
axs[1].grid(True)

for ax in axs:
    ax.tick_params(labelbottom=True)

plt.tight_layout()


t = np.arange(odom_cmd.shape[0])
t = np.divide(t, 4)
fig,axs = plt.subplots(2,1,figsize = (12,6), sharex=True)

axs[0].plot(t,odom_cmd[:, 0], linewidth=2)
axs[0].set_ylabel("Lin Velocity")
axs[0].set_xlabel("Step")
axs[0].set_title("Linear Velocity True")
axs[0].grid(True)

axs[1].plot(t,odom_cmd[:, 1], linewidth=2)
axs[1].set_ylabel("Ang Velocity")
axs[1].set_xlabel("Step")
axs[1].set_title("Angular Velocity True")
axs[1].grid(True)

for ax in axs:
    ax.tick_params(labelbottom=True)

plt.tight_layout()

plt.show()