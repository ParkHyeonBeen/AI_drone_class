import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *
from drawnow import *
import scipy.integrate as integrate
import matplotlib.animation as animation
from time import *
import random

# Parameter
L1 = 1
L2 = 1
M1 = 1
M2 = 1
g = 9.8

n = 500  # 시나리오 스텝 수
k = 0 # 넘어가는 스텝
dt = 0.1
run_time = 5 # seconds

# 테일러급수 단점 : h_time 에 따라 값들이 크게 바뀌기 때문에
#                 값이 너무 커질 경우(0.01보다 크면) 안정성을 읽을 가능성이 크다.

# Variable

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

x_current = np.zeros(n)
y_current = np.zeros(n)
rrx_current = np.zeros(n)
rry_current = np.zeros(n)
Theta1_current = np.zeros(n)
Theta2_current = np.zeros(n)
Theta1_d_current = np.zeros(n)
Theta2_d_current = np.zeros(n)
Theta1_dd_current = np.zeros(n)
Theta2_dd_current = np.zeros(n)

state_desire = np.zeros((n, 2, 1))
state_d_desire = np.zeros((n, 2, 1))
state_dd_desire = np.zeros((n, 2, 1))

state_current = np.zeros((n, 2, 1))
state_d_current = np.zeros((n, 2, 1))
state_dd_current = np.zeros((n, 2, 1))

error_state = np.zeros((n, 2, 1))
error_state_d = np.zeros((n, 2, 1))
error_state_dd = np.zeros((n, 2, 1))

sliding_variable = np.zeros((n, 2, 1))
sliding_variable_d = np.zeros((n, 2, 1))

mat_inertia = np.zeros((n, 2, 2))
mat_cori = np.zeros((n, 2, 1))
mat_grav = np.zeros((n, 2, 1))
Torque_eq = np.zeros((n, 2, 1))
state_CMD = np.zeros((n, 2, 1))
Torque_CMD = np.zeros((n, 2, 1))

# Kinematic


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

for i in range(0, n-1):

    Theta1_d_desire[i+1] = (Theta1_desire[i + 1] - Theta1_desire[i]) / h_time
    Theta2_d_desire[i+1] = (Theta2_desire[i + 1] - Theta2_desire[i]) / h_time

    Theta1_dd_desire[i+1] = (Theta1_d_desire[i + 1] - Theta1_d_desire[i]) / h_time
    Theta2_dd_desire[i+1] = (Theta2_d_desire[i + 1] - Theta2_d_desire[i]) / h_time

    state_desire[i + 1, :, :] = np.reshape([Theta1_desire[i], Theta2_desire[i]], (2, 1))
    state_d_desire[i + 1, :, :] = np.reshape([Theta1_d_desire[i], Theta2_d_desire[i]], (2, 1))
    state_dd_desire[i + 1, :, :] = np.reshape([Theta1_dd_desire[i], Theta2_dd_desire[i]], (2, 1))

# Dynamics and Sliding Mode Control

Theta1_current[0] = Theta1_desire[0]
Theta2_current[0] = Theta2_desire[0]

'''
Theta1_d_current[0] = Theta1_d_desire[1]
Theta2_d_current[0] = Theta2_d_desire[1]

Theta1_dd_current[0] = Theta1_dd_desire[1]
Theta2_dd_current[0] = Theta2_dd_desire[1]
'''

#state_dd_current[0, :, :] = np.array([Theta1_dd_current[0], Theta2_dd_current[0]]).reshape(2, 1)

Ks = np.matrix('100 0; 0 100')
Kt = np.matrix('10 0; 0 10')

for i in range(0, n):

    state_current[i, :, :] = np.array([Theta1_current[i], Theta2_current[i]]).reshape(2, 1)
    state_d_current[i, :, :] = np.array([Theta1_d_current[i], Theta2_d_current[i]]).reshape(2, 1)
    state_dd_current[i, :, :] = np.array([Theta1_dd_current[i], Theta2_dd_current[i]]).reshape(2, 1)

    mat_inertia[i, 0, 0] = (M1 + M2)*L1**2 + M2*L2**2 + 2*M2*L1*L2*cos(Theta2_current[i])
    mat_inertia[i, 1, 0] = M2*L2**2 + M2*L1*L2*cos(Theta2_current[i])
    mat_inertia[i, 0, 1] = M2*L2**2 + M2*L1*L2*cos(Theta2_current[i])
    mat_inertia[i, 1, 1] = M2*L2**2

    mat_cori[i, 0, 0] = -M2*L1*L2*(2*Theta1_d_current[i]*Theta2_d_current[i] + Theta2_d_current[i]**2)*sin(
        Theta2_desire[i])
    mat_cori[i, 1, 0] = M2*L1*L2*(Theta1_d_current[i]**2)*sin(Theta2_current[i])

    mat_grav[i, 0, 0] = (M1 + M2)*g*L1*cos(Theta1_current[i]) + M2*g*L2*cos(Theta1_current[i] + Theta2_current[i])
    mat_grav[i, 1, 0] = M2*g*L2*cos(Theta1_current[i] + Theta2_current[i])

    Torque_eq[i, :, :] = np.dot(mat_inertia[i, :, :], state_dd_current[i, :, :]) + mat_cori[i, :, :] + mat_grav[i, :, :]

    error_state[i, :, :] = state_desire[i, :, :] - state_current[i, :, :]
    error_state_d[i, :, :] = state_d_desire[i, :, :] - state_d_current[i, :, :]

    sliding_variable[i, :, :] = np.dot(Ks, error_state[i, :, :]) + error_state_d[i, :, :]
    state_CMD[i, :, :] = state_desire[i, :, :] + np.dot(Ks, error_state_d[i, :, :]) + np.dot(Kt, np.sign(sliding_variable[i, :, :]))

    Torque_CMD[i, :, :] = np.dot(mat_inertia[i, :, :], state_CMD[i, :, :]) + mat_cori[i, :, :] + mat_grav[i, :, :]

    rrx_current[i] = L1 * cos(Theta1_current[i])
    rry_current[i] = L1 * sin(Theta1_current[i])

    x_current[i] = L1*cos(Theta1_current[i]) + L2*cos(Theta1_current[i] + Theta2_current[i])
    y_current[i] = L1*sin(Theta1_current[i]) + L2*sin(Theta1_current[i] + Theta2_current[i])

    if i == n-1:
        break

    mat_inertia_inv = np.linalg.inv(mat_inertia[i, :, :])
    dot_tmp = Torque_CMD[i, :, :] - mat_cori[i, :, :] - mat_grav[i, :, :]
    state_dd_current[i + 1, :, :] = np.dot(mat_inertia_inv, dot_tmp)

    Theta1_dd_current[i + 1] = state_dd_current[i + 1, 0, 0]
    Theta2_dd_current[i + 1] = state_dd_current[i + 1, 1, 0]

    Theta1_d_current[i + 1] = Theta1_d_current[i] + Theta1_dd_current[i + 1]*h_time
    Theta2_d_current[i + 1] = Theta2_d_current[i] + Theta2_dd_current[i + 1]*h_time

    Theta1_current[i + 1] = Theta1_current[i] + Theta1_d_current[i + 1]*h_time
    Theta2_current[i + 1] = Theta2_current[i] + Theta2_d_current[i + 1]*h_time

'''
A = plt.plot(Theta1_current)
B = plt.plot(Theta2_current)
plt.plot(error_state[i, 0, 0],error_state[i, 1, 0])
plt.show()
print(error_state[2,:,:])
print(error_state_d[2,:,:])
print(sliding_variable[2,:,:])
print(state_dd_current[2,:,:])
'''

for i in range(n):

    print("%d th index" % i)
    print('Desire Value')
    print(Theta1_desire[i], Theta2_desire[i])
    print(Theta1_d_desire[i], Theta2_d_desire[i])
    print(Theta1_dd_desire[i], Theta2_dd_desire[i])
    print('Real Value')
    print(Theta1_current[i], Theta2_current[i])
    print(Theta1_d_current[i], Theta2_d_current[i])
    print(Theta1_dd_current[i], Theta2_dd_current[i])
    print('state')
    print(state_desire[i, :, :])
    print(state_current[i, :, :])
    print('Error state')
    print(error_state[i, :, :])
    print('Error state dif')
    print(error_state_d[i, :, :])
    print('Sliding Variable')
    print(sliding_variable[i, :, :])

phase_portrait = plt.plot(error_state[:, 0, 0], error_state_d[:, 1, 0])
plt.grid()

fig_desire = plt.figure()
ax_desire = fig_desire.add_subplot(111, projection='3d', autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax_desire.grid()

line_desire, = ax_desire.plot([], [], [], '-o', lw=2) # lw =Line Width
time_template_desire = 'time = %.1fs'
time_text_desire = ax_desire.text(0.05, 0.9, '', transform=ax_desire.transAxes)


fig_current = plt.figure()
ax_current = fig_current.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax_current.grid()

line_current, = ax_current.plot([], [], [], '-o', lw=2) # lw =Line Width
time_template_current = 'time = %.1fs'
time_text_current = ax_current.text(0.05, 0.9, '', transform=ax_current.transAxes)


def init_desire():
    line_desire.set_data([], [])
    time_text_desire.set_text('')
    return line_desire, time_text_desire


def init_current():
    line_current.set_data([], [])
    time_text_current.set_text('')
    return line_current, time_text_current


def animate_desire(i):
    thisx = [0, rrx_desire[i], x_desire[i]]
    thisy = [0, rry_desire[i], y_desire[i]]

    line_desire.set_data(thisx, thisy)
    time_text_desire.set_text(time_template_desire % (i*dt))
    return line_desire, time_text_desire


def animate_current(i):

    thisx = [0, rrx_current[i], x_current[i]]
    thisy = [0, rry_current[i], y_current[i]]

    line_current.set_data(thisx, thisy)
    time_text_current.set_text(time_template_current % (i*dt))
    return line_current, time_text_current


ani_desire = animation.FuncAnimation(fig_desire, animate_desire, np.arange(1, len(y_desire)),
                              interval=25, blit=True, init_func=init_desire) # 반복적으로 함수를 호출하여 애니메이션을 만듦.


ani_current = animation.FuncAnimation(fig_current, animate_current, np.arange(1, len(y_current)),
                                      interval=25, blit=True, init_func=init_current)


# ani.save('double_pendulum.mp4', fps=15)
plt.show()
