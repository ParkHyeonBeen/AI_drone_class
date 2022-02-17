import numpy as np
import vtk
from scipy.interpolate import interp1d
import math, os

class VTK_Visualization:

    def __init__(self, log_dir, angle, position, goal, obstacle_position=None, obstacle_position2=None):

        self.imdir = log_dir+"/vtk"
        os.mkdir(self.imdir)
        self.goal = goal
        self.position = position
        self.angle = angle
        self.obstacle_position = obstacle_position
        self.obstacle_position2 = obstacle_position2

        # Setup a renderer, render window, and interactor
        self.renderer = vtk.vtkRenderer()
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.SetWindowName("AirSimPX4")
        self.renderWindow.SetSize(720, 480)
        self.renderWindow.AddRenderer(self.renderer)
        self.renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        self.renderWindowInteractor.SetRenderWindow(self.renderWindow)
        # Add light
        self.add_light()
        # Set camera
        self.camera_position = np.array([0, -8., 5.])
        self.set_camera(position=self.camera_position, zoom=0.5)
        # Draw goal
        self.draw_goal(radius=3.0)
        # Draw floor
        self.draw_floor()
        # Draw Drone
        self.draw_drone()
        # Draw Drone
        if self.obstacle_position is not None:
            self.draw_flag()
        if self.obstacle_position2 is not None:
            self.draw_pillar()
        # Set background
        self.renderer.GradientBackgroundOn()
        self.renderer.SetBackground(0, 0, 0)
        self.renderer.SetBackground2(0.8, 0.8, 0.89)
        self.renderWindowInteractor.Initialize()

    def render(self,imsave=False):
        position = self.position + np.array([0.,0.,1.05])
        goal = self.goal + np.array([0.,0.,1.05])
        self.update_camera(np.zeros(3))
        self.update_goal(goal)
        if self.obstacle_position is not None:
            self.update_flag(self.obstacle_position)
        if self.obstacle_position2 is not None:
            self.update_pillar(self.obstacle_position2)
        for i in range(len(position)):
            self.update_drone(position=position[i],angle=self.angle[i])
            self.update_camera(position[i])
            self.renderWindow.Render()
            if imsave:
                self.save_image((i+1))

    def add_light(self):
        light = vtk.vtkLight()
        light.SetPosition(0, 0, 10)
        light.SetIntensity(0.8)
        self.renderer.AddLight(light)

    def set_camera(self, position, focalpoint=np.array([0,0,0]), zoom=1.0):
        self.camera = vtk.vtkCamera()
        self.camera.SetPosition(position[0],position[1],position[2])
        self.camera.SetFocalPoint(focalpoint[0],focalpoint[1],focalpoint[2])
        self.camera.Zoom(zoom)
        self.renderer.SetActiveCamera(self.camera)

    def draw_goal(self, radius=1.5):

        self.goal_actor = [vtk.vtkActor(),vtk.vtkActor()]
        for i in range(len(self.goal_actor)):
            # Create a sphere
            sphereSource = vtk.vtkSphereSource()
            sphereSource.SetCenter(0, 0, 0)
            if i == 0:
                r = radius
                opacity = 0.3
            else:
                r = 0.5
                opacity = 1.0
            sphereSource.SetRadius(r)
            # Create a mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphereSource.GetOutputPort())
            self.goal_actor[i].SetMapper(mapper)
            prop = self.goal_actor[i].GetProperty()
            prop.SetColor(0, 1, 0)
            prop.SetOpacity(opacity)
            self.renderer.AddActor(self.goal_actor[i])

    def draw_floor(self):

        L = 100
        lineN = 50

        # Create a floor
        cube = vtk.vtkCubeSource()
        cube.SetCenter(0.0, 0.0, 0.0)
        cube.SetXLength(L)
        cube.SetYLength(L)
        cubeMapper = vtk.vtkPolyDataMapper()
        cubeMapper.SetInputConnection(cube.GetOutputPort())
        actor = vtk.vtkActor()
        prop = actor.GetProperty()
        prop.SetColor(0.6, 0.6, 0.6)
        actor.SetMapper(cubeMapper)
        # Add the actor to the scene
        self.renderer.AddActor(actor)

        for i in range(lineN+1):
            # Create a grid
            cube = vtk.vtkCubeSource()
            cube.SetCenter((i-lineN/2)*L/lineN, 0.0, 0.5)
            cube.SetXLength(0.03)
            cube.SetYLength(L)
            cube.SetZLength(0.03)
            cubeMapper = vtk.vtkPolyDataMapper()
            cubeMapper.SetInputConnection(cube.GetOutputPort())
            actor = vtk.vtkActor()
            prop = actor.GetProperty()
            prop.SetColor(0.9,0.9,0.9)
            actor.SetMapper(cubeMapper)
            # Add the actor to the scene
            self.renderer.AddActor(actor)

        for i in range(lineN+1):
            # Create a grid
            cube = vtk.vtkCubeSource()
            cube.SetCenter(0.0, (i-lineN/2)*L/lineN, 0.5)
            cube.SetXLength(L)
            cube.SetYLength(0.03)
            cube.SetZLength(0.03)
            cubeMapper = vtk.vtkPolyDataMapper()
            cubeMapper.SetInputConnection(cube.GetOutputPort())
            actor = vtk.vtkActor()
            prop = actor.GetProperty()
            prop.SetColor(0.9,0.9,0.9)
            actor.SetMapper(cubeMapper)
            # Add the actor to the scene
            self.renderer.AddActor(actor)

        cubeAxesActor = vtk.vtkCubeAxesActor()
        cubeAxesActor.SetBounds(-L/2,L/2,-L/2,L/2,-L/2,L/2)
        cubeAxesActor.SetCamera(self.renderer.GetActiveCamera())

        cubeAxesActor.GetXAxesGridlinesProperty().SetColor(0.98, 0.98, 0.98)
        cubeAxesActor.GetYAxesGridlinesProperty().SetColor(0.98, 0.98, 0.98)
        cubeAxesActor.GetZAxesGridlinesProperty().SetColor(0.98, 0.98, 0.98)

        cubeAxesActor.GetXAxesInnerGridlinesProperty().SetColor(0.98, 0.98, 0.98)
        cubeAxesActor.GetYAxesInnerGridlinesProperty().SetColor(0.98, 0.98, 0.98)
        cubeAxesActor.DrawXInnerGridlinesOn()
        cubeAxesActor.DrawYInnerGridlinesOn()

        cubeAxesActor.DrawXGridlinesOn()
        cubeAxesActor.DrawYGridlinesOn()
        cubeAxesActor.DrawZGridlinesOn()

        cubeAxesActor.XAxisMinorTickVisibilityOn()
        cubeAxesActor.YAxisMinorTickVisibilityOn()
        cubeAxesActor.ZAxisMinorTickVisibilityOn()

        self.renderer.AddActor(cubeAxesActor)

    def draw_drone(self):
        drone_stl = "./STL/Delta.stl"

        reader = vtk.vtkSTLReader()
        reader.SetFileName(drone_stl)
        reader.Update()

        transform = vtk.vtkTransform()
        transform.Scale(0.0032, 0.0032, 0.0032)
        transform.Translate(-150,150,100)
        transform.RotateWXYZ(90, 1, 0, 0)
        transform.RotateWXYZ(45, 0, 1, 0)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputConnection(reader.GetOutputPort())
        transformFilter.SetTransform(transform)
        transformFilter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transformFilter.GetOutputPort())

        self.drone_actor = vtk.vtkActor()
        self.drone_actor.SetMapper(mapper)
        prop = self.drone_actor.GetProperty()
        prop.SetColor(np.array([0, 200, 255])/255)
        self.renderer.AddActor(self.drone_actor)

    def draw_pillar(self):

        self.pillar_actors = []

        for i in range(len(np.sum(self.obstacle_position2,axis=1))):
            stl = "./STL/Pillar.stl"

            reader = vtk.vtkSTLReader()
            reader.SetFileName(stl)
            reader.Update()

            transform = vtk.vtkTransform()
            transform.Scale(0.15, 0.15, 0.18)
            #transform.Translate(11, 0, 65)
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetInputConnection(reader.GetOutputPort())
            transformFilter.SetTransform(transform)
            transformFilter.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(transformFilter.GetOutputPort())

            pillar_actor = vtk.vtkActor()
            pillar_actor.SetMapper(mapper)
            prop = pillar_actor.GetProperty()
            prop.SetColor(np.array([0, 200, 255])/255)
            self.renderer.AddActor(pillar_actor)
            self.pillar_actors.append(pillar_actor)

    def draw_flag(self):

        self.flag_actors = []

        for i in range(len(np.sum(self.obstacle_position,axis=1))):
            stl = "./STL/Flag.stl"

            reader = vtk.vtkSTLReader()
            reader.SetFileName(stl)
            reader.Update()

            transform = vtk.vtkTransform()
            transform.Scale(0.1, 0.1, 0.15*0.75)
            transform.Translate(11, 0, 65)
            transform.RotateWXYZ(90, 0, 0, 1)
            transform.RotateWXYZ(180, 1, 0, 0)
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetInputConnection(reader.GetOutputPort())
            transformFilter.SetTransform(transform)
            transformFilter.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(transformFilter.GetOutputPort())

            flag_actor = vtk.vtkActor()
            flag_actor.SetMapper(mapper)
            prop = flag_actor.GetProperty()
            prop.SetColor(np.array([0, 200, 255])/255)
            self.renderer.AddActor(flag_actor)
            self.flag_actors.append(flag_actor)

    def update_camera(self, position):
        self.camera.SetPosition(position + self.camera_position)
        self.camera.SetFocalPoint(position)

    def update_flag(self, position):
        for i in range(len(self.flag_actors)):
            self.flag_actors[i].SetPosition(*position[i])

    def update_pillar(self, position):
        for i in range(len(self.pillar_actors)):
            self.pillar_actors[i].SetPosition(*position[i])

    def update_drone(self, position=None, angle=None):
        if position is not None:
            self.drone_actor.SetPosition(position[0], position[1], position[2])
        if angle is not None:
            self.drone_actor.SetOrientation(angle[0],angle[1],angle[2])
        pass

    def update_goal(self, goal):
        [goal_actor.SetPosition(goal[0],goal[1],goal[2]) for goal_actor in self.goal_actor]

    def save_image(self,idx):
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(self.renderWindow)
        w2if.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(self.imdir+"/%d.jpg"%idx)
        if vtk.VTK_MAJOR_VERSION == 5:
            writer.SetInput(w2if.GetOutput())
        else:
            writer.SetInputData(w2if.GetOutput())
        writer.Write()

def interp(data,multi,kind='linear'):
    x = np.linspace(0, len(data), num=len(data), endpoint=True)
    y = data.reshape([-1])
    f = interp1d(x, y, kind=kind)
    xnew = np.linspace(0, len(data), num=len(data)*multi, endpoint=True)
    return f(xnew)

def log_processing(raw_log,raw_goal,init_yaw):
    pose = raw_log[:,:3]
    pose[:,2] -= init_yaw
    pose[:,2] *= -1
    rot_matrix = np.array([[math.cos(-init_yaw*np.pi/180), math.sin(-init_yaw*np.pi/180)],
                           [-math.sin(-init_yaw*np.pi/180),math.cos(-init_yaw*np.pi/180)]])
    proc_goal = raw_goal.copy()
    proc_goal[:2] = np.matmul(raw_goal[:2], rot_matrix)

    position = raw_log[:,3:]
    position[:,:2] = np.matmul(position[:,:2], rot_matrix)

    position[:,:2] = position[:,:2][:,::-1]
    proc_goal[:2] = proc_goal[:2][::-1]

    proc_log = np.hstack([pose,position])

    init_log = []
    temp = np.array([np.zeros(proc_log.shape[-1]),proc_log[0]])
    for i in range(temp.shape[-1]):
        init_log.append(interp(temp.T[i], multi=10, kind='linear'))
    init_log = np.array(init_log).T
    init_log += np.random.normal(0.0, 0.01, init_log.shape)
    init_log[0] = np.zeros(proc_log.shape[-1])

    final_log = []
    final = proc_log[-1].copy()
    final[:2] = 0.
    final[-1] = 0.
    temp = np.array([proc_log[-1],final])
    for i in range(temp.shape[-1]):
        final_log.append(interp(temp.T[i], multi=10, kind='linear'))
    final_log = np.array(final_log).T
    final_log += np.random.normal(0.0, 0.01, final_log.shape)
    final_log[-1] = final
    proc_log = np.vstack([init_log,proc_log,final_log])

    simlog = []
    for i in range(proc_log.shape[-1]):
        simlog.append(interp(proc_log.T[i],multi=4,kind='cubic'))
    simlog = np.array(simlog).T

    pose = simlog[:, :3]
    position = simlog[:, 3:]

    return pose, position, proc_goal