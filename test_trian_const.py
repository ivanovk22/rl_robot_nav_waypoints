import numpy as np
import matplotlib.pyplot as plt

# Robot parameters
v_max = 0.25
w_max = 1.0

V, W = np.meshgrid(np.linspace(0, v_max, 100),
                   np.linspace(-w_max, w_max, 100))
# print(V.shape)
# print(V)
# Allowed region: both constraints
allowed = np.zeros_like(V, dtype=bool)
for i in range(V.shape[0]):
    for j in range(V.shape[1]):
        v = V[i, j]
        w = W[i, j]

        # Step 1: clip angular velocity
        allowed_ang = w_max * (1 - v / v_max)
        w_clipped = np.clip(w, -allowed_ang, allowed_ang)

        # Step 2: clip linear velocity
        allowed_lin = v_max * (1 - abs(w_clipped) / w_max)
        v_clipped = np.clip(v, 0.0, allowed_lin)





        # If v and w are still unchanged after clipping, mark as allowed
        if np.isclose(v, v_clipped) and np.isclose(w, w_clipped):
            allowed[i, j] = True
# Convert boolean to int for plotting
allowed_numeric = allowed.astype(int)

plt.figure(figsize=(6,6))
plt.contourf(V, W, allowed_numeric, levels=[-0.1, 0.5, 1.1], colors=['white', 'lightblue'])
plt.xlabel('Linear velocity v [m/s]')
plt.ylabel('Angular velocity ω [rad/s]')
plt.title('Allowed region for symmetric triangular constraint')
plt.grid(True)
plt.show()
