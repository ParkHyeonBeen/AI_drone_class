import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *
from drawnow import *
import scipy.integrate as integrate
import matplotlib.animation as animation
from time import *

L1 = 1
L2 = 1
M1 = 1
M2 = 1
g = 9.8

n = 200  # 시나리오 스텝 수
k = 0 # 넘어가는 스텝
dt = 0.1
run_time = 20 # seconds

h_time = run_time/n
h_angle = 2 * pi / n

x_desire = np.zeros(n)
y_desire = np.zeros(n)
rrx_desire = np.zeros(n)
rry_desire = np.zeros(n)
Theta1_desire = np.zeros(n)
Theta2_desire = np.zeros(n)
Theta1_d_desire = np.zeros(n)
Theta2_d_desire = np.zeros(n)
Theta1_dd_desire = np.zeros(n)
Theta2_dd_desire = np.zeros(n)

state = np.zeros((2, 1, n))
state_d = np.zeros((2, 1, n))
state_dd = np.zeros((2, 1, n))

mat_inertia = np.zeros((2, 2, n))
mat_cori = np.zeros((2, 1, n))
mat_grav = np.zeros((2, 1, n))
mat_torque = np.zeros((2, 1, n))


def range_float(start, end, step):
    raw = start
    while raw < end:
        yield raw
        raw += step


for i in range_float(h_angle, 2 * pi, h_angle):

    r = 1 - cos(i)

    xd = r*sin(i)
    yd = r*cos(i)

    C = (r**2 - L1**2 - L2**2)/(2*L1*L2)
    D = sqrt(1 - C**2)

    theta2_tmp = atan2(D, C)
    theta1_tmp = atan2(yd, xd) - atan2(L2 * sin(theta2_tmp), L1 + L2 * cos(theta2_tmp))

    if i > 3*pi/2:

        theta1_tmp = theta1_tmp - 2 * pi

    Theta1_desire[k] = theta1_tmp
    Theta2_desire[k] = theta2_tmp

    rrx_desire[k] = L1 * cos(theta1_tmp)
    rry_desire[k] = L1 * sin(theta1_tmp)

    x_desire[k] = L1 * cos(theta1_tmp) + L2 * cos(theta1_tmp + theta2_tmp)
    y_desire[k] = L1 * sin(theta1_tmp) + L2 * sin(theta1_tmp + theta2_tmp)

    k += 1

    if k >= n:
        break

for i in range(1, n):

    Theta1_d_desire[i] = (Theta1_desire[i + 1] - Theta1_desire[i]) / h_time
    Theta2_d_desire[i] = (Theta2_desire[i + 1] - Theta2_desire[i]) / h_time

    Theta1_dd_desire[i] = (Theta1_d_desire[i + 1] - Theta1_d_desire[i]) / h_time
    Theta2_dd_desire[i] = (Theta2_d_desire[i + 1] - Theta2_d_desire[i]) / h_time

    state_dd[i] = [Theta1_dd_desire[i], Theta2_dd_desire[i]]


fig_desire = plt.figure()
ax = fig_desire.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], '-o', lw=2) # lw =Line Width
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate_desire(i):
    thisx = [0, rrx_desire[i], x_desire[i]]
    thisy = [0, rry_desire[i], y_desire[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text

print(len(y_desire))

ani = animation.FuncAnimation(fig_desire, animate_desire, np.arange(1, len(y_desire)),
                              interval=25, blit=True, init_func=init) # 반복적으로 함수를 호출하여 애니메이션을 만듦.

# ani.save('double_pendulum.mp4', fps=15)
plt.show()