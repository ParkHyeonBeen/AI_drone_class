import time
from math import *
import keyboard
import numpy as np
from envs.quad_env import quadrotor
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import Axes3D


class visualize1():
    def __init__(self, quadrotor):

        self.qd = quadrotor

        self.fig = plt.figure()
        self.ax = p3.Axes3D(self.fig)

        self.x_axis = np.arange(-2, 3)
        self.y_axis = np.arange(-2, 3)
        self.z_axis = np.arange(-2, 3)
        self.pointCM, = self.ax.plot([0], [0], [0], 'b.')
        self.pointBLDC1, = self.ax.plot([0], [0], [0], 'b.')
        self.pointBLDC2, = self.ax.plot([0], [0], [0], 'b.')
        self.pointBLDC3, = self.ax.plot([0], [0], [0], 'b.')
        self.pointBLDC4, = self.ax.plot([0], [0], [0], 'b.')
        self.line1, = self.ax.plot([0, 0], [0, 0], [0, 0], 'b.')
        self.line2, = self.ax.plot([0, 0], [0, 0], [0, 0], 'b.')

        self.ax.plot([0, 0], [0, 0], [0, 0], 'k+')
        self.ax.plot(self.x_axis, np.zeros(5), np.zeros(5), 'r--', linewidth=0.5)
        self.ax.plot(np.zeros(5), self.y_axis, np.zeros(5), 'g--', linewidth=0.5)
        self.ax.plot(np.zeros(5), np.zeros(5), self.z_axis, 'b--', linewidth=0.5)

        self.ax.set_xlim([-3, 3])
        self.ax.set_ylim([-3, 3])
        self.ax.set_zlim([-3, 3])

        self.ax.set_xlabel('X-self.axis (in meters)')
        self.ax.set_ylabel('Y-self.axis (in meters)')
        self.ax.set_zlabel('Z-self.axis (in meters)')

        self.time_display = self.ax.text(22.0, 1.0, 39.0, "red", color='red', transform=self.ax.transAxes)
        self.state_display = self.ax.text(1.0, 1.0, 41.0, "green", color='green', transform=self.ax.transAxes)


    def update_point(self, n):

        self.line1.remove()
        self.line2.remove()
        self.qd.PID_position()
        self.qd.PID_attitude()
        self.qd.PID_rate()
        self.qd.quad_motor_speed()
        state = self.qd.step(self.qd.state, self.qd.input_vector)

        # length of the arm of the quadrotor is 0.5m
        (x_bf, y_bf, z_bf) = self.qd.rotateGFtoBF(state[0], state[1], state[2], state[9], state[10], state[11])
        (x_bl1, y_bl1, z_bl1) = self.qd.rotateBFtoGF(state[0], state[1], state[2], x_bf, y_bf + 0.25, z_bf)
        (x_bl2, y_bl2, z_bl2) = self.qd.rotateBFtoGF(state[0], state[1], state[2], x_bf + 0.25, y_bf, z_bf)
        (x_bl3, y_bl3, z_bl3) = self.qd.rotateBFtoGF(state[0], state[1], state[2], x_bf, y_bf - 0.25, z_bf)
        (x_bl4, y_bl4, z_bl4) = self.qd.rotateBFtoGF(state[0], state[1], state[2], x_bf - 0.25, y_bf, z_bf)

        self.line1, = self.ax.plot([x_bl4, x_bl2], [y_bl4, y_bl2], [z_bl4, z_bl2], 'ko-', lw=1.5, markersize=3)
        self.line2, = self.ax.plot([x_bl3, x_bl1], [y_bl3, y_bl1], [z_bl3, z_bl1], 'ko-', lw=1.5, markersize=3)

        # Comment this line if you don't require the trail that is left behined the quadrotor
        self.ax.plot([state[9]], state[10], state[11], "g.", markersize=1)
        # print self.qd.input_vector

        self.time_display.set_text('Simulation time = %.1fs' % (self.qd.time_elapsed()))
        self.state_display.set_text(
            'Position of the quad: \n x = %.1fm y = %.1fm z = %.1fm' % (self.qd.state[9], self.qd.state[10], self.qd.state[11]))

        if keyboard.is_pressed('p'):
            self.qd.pauseEnv()
        if keyboard.is_pressed('u'):
            self.qd.unpauseEnv()
        if keyboard.is_pressed('r'):
            self.qd.rstEnv()

        return self.pointCM, self.pointBLDC1, self.pointBLDC2, self.pointBLDC3, self.pointBLDC4, self.time_display, self.state_display

    def render(self):

        ani = animation.FuncAnimation(self.fig, self.update_point, interval=63)
        plt.show()


class visualize2():
    def __init__(self, states):
        self.states = states
        self.limit0 = -1.0
        self.limit1 = 1.0

    def func_line(self, num, dataSet, line):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(dataSet[0:2, :num])
        line.set_3d_properties(dataSet[2, :num])
        return line

    def func_dot(self, num, dataSet, line, redDots):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(dataSet[0:2, :num])
        line.set_3d_properties(dataSet[2, :num])
        redDots.set_data(dataSet[0:2, :num])
        redDots.set_3d_properties(dataSet[2, :num])
        return line, redDots

    def render(self):
        dataSet = np.array(self.states[:, :3]).T
        numDataPoints = len(self.states[:, :3])

        # GET SOME MATPLOTLIB OBJECTS
        fig = plt.figure()
        ax = Axes3D(fig)

        # NOTE: Can't pass empty arrays into 3d version of plot()
        redDots = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='r', marker='o')[0]  # For scatter plot
        line = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='g')[0]  # For line plot

        # AXES PROPERTIES]
        ax.set_xlim3d([self.limit0, self.limit1])
        ax.set_ylim3d([self.limit0, self.limit1])
        ax.set_zlim3d([self.limit0, self.limit1])
        ax.set_xlabel('X(t)')
        ax.set_ylabel('Y(t)')
        ax.set_zlabel('Z(t)')
        ax.set_title('Trajectory')

        # Creating the Animation object
        line_ani = animation.FuncAnimation(fig, self.func_dot, frames=numDataPoints, fargs=(dataSet, line, redDots), interval=5,
                                           blit=False)
        # line_ani.save(r'AnimationNew.mp4')
        plt.show()

