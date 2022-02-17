import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# # Make data
# u = np.linspace(0, 2*np.pi, 100)
# v = np.linspace(0, np.pi, 100)
# x = 10 * np.outer(np.cos(u), np.sin(v)) + 20 * np.ones((100, 100))
# y = 10 * np.outer(np.sin(u), np.sin(v))
# z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

x = np.ones((10, 10))*20
y = np.ones((10, 10))*20
z = np.ones((10, 10))

print(x.shape)

# Plot the surface
ax.plot_surface(x, y, z)

plt.show()


# ######################################################################################
#
# # ANIMATION FUNCTION
# def func(num, dataSet, line):
#     # NOTE: there is no .set_data() for 3 dim data...
#     line.set_data(dataSet[0:2, :num])
#     line.set_3d_properties(dataSet[2, :num])
#     return line
#
#
# # THE DATA POINTS
# t = np.arange(0, 20, 0.2)  # This would be the z-axis ('t' means time here)
# x = np.cos(t) - 1
# y = 1 / 2 * (np.cos(2 * t) - 1)
# dataSet = np.array([x, y, t])
# numDataPoints = len(t)
#
# # GET SOME MATPLOTLIB OBJECTS
# fig = plt.figure()
# ax = Axes3D(fig)
#
# # NOTE: Can't pass empty arrays into 3d version of plot()
# line = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='g')[0]  # For line plot
#
# # AXES PROPERTIES]
# # ax.set_xlim3d([limit0, limit1])
# ax.set_xlabel('X(t)')
# ax.set_ylabel('Y(t)')
# ax.set_zlabel('time')
# ax.set_title('Trajectory of electron for E vector along [120]')
#
# # Creating the Animation object
# line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataSet, line), interval=500, blit=False)
# # line_ani.save(r'AnimationNew.mp4')
#
# plt.show()

# class render():
#     def __init__(self, states):
#
#         self.x_data, self.y_data, self.z_data = [], [], []
#
#         self.x_data.append(states[0])
#         self.y_data.append(states[1])
#         self.z_data.append(states[2])
#
#         self.n = 500  # 시나리오 스텝 수
#         self.k = 0  # 넘어가는 스텝
#         self.dt = 0.1
#         self.run_time = 5  # seconds
#
#         fig_desire = plt.figure()
#         ax_desire = fig_desire.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
#         ax_desire.grid()
#
#         self.line_desire, = ax_desire.plot([], [], '-o', lw=2) # lw =Line Width
#         self.time_template_desire = 'time = %.1fs'
#         self.time_text_desire = ax_desire.text(0.05, 0.9, '', transform=ax_desire.transAxes)
#
#     def init_desire(self):
#         self.line_desire.set_data([], [])
#         self.time_text_desire.set_text('')
#         return self.line_desire, self.time_text_desire
#
#     def animate_desire(self, i):
#         thisx = x_desire[i]
#         thisy = y_desire[i]
#
#         # thisx = [0, rrx_desire[i], x_desire[i]]
#         # thisy = [0, rry_desire[i], y_desire[i]]
#
#         self.line_desire.set_data(thisx, thisy)
#         self.time_text_desire.set_text(self.time_template_desire % (i*self.dt))
#         return self.line_desire, self.time_text_desire
#
#     ani_desire = animation.FuncAnimation(self.fig_desire, animate_desire, np.arange(1, len(y_desire)),
#                                   interval=25, blit=True, init_func=init_desire) # 반복적으로 함수를 호출하여 애니메이션을 만듦.

#######################################################################################

# sibal = 500
#
# plt.figure(1)
# plt.plot(x_data)
# plt.get_current_fig_manager().window.setGeometry(0,sibal,sibal,sibal)
# plt.show(block=False)
# plt.pause(0.00001)
# plt.cla()
#
# plt.figure(2)
# plt.plot(y_data)
# plt.get_current_fig_manager().window.setGeometry(sibal, sibal, sibal, sibal)
# plt.show(block=False)
# plt.pause(0.00001)
# plt.cla()
#
# plt.figure(3)
# plt.plot(z_data)
# plt.get_current_fig_manager().window.setGeometry(sibal*2, sibal, sibal, sibal)
# plt.show(block=False)
# plt.pause(0.00001)
# plt.cla()

# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_data, y_data, z_data, marker='o', s=15, cmap='Greens')
# plt.cla()
