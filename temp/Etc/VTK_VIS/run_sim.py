from vtkvis import VTK_Visualization, log_processing
import numpy as np
import pandas as pd
import imageio, cv2, glob

VTK = True
VIDEO = True
GIF = True

print("## Start")
print("    -- READING LOG FILES")

# Reading the data from a CSV file using pandas
log_dir = "log/ys720_11_TEST1_SONAR_MB1242"
csvfile = log_dir+"/trajectory.csv"
log = pd.read_csv(csvfile,sep=',',header=0).values

init_yaw = 50.08
goal = np.array([37.66,32.89, 3.])
obj = np.array([[-5,12,0],[5,12,0]])
obj2 = np.array([[0,25,0]])

log = log[:,:6] # [Roll, Pitch, Yaw, X, Y, Z]
pose, position, goal = log_processing(log,goal,init_yaw)

if VTK:
    print("    -- VTK VISUALIZATION")
    # VTK Visualization
    vis = VTK_Visualization(log_dir, pose, position, goal, obj, obj2)
    vis.render(imsave=True)

if VIDEO or GIF:

    print("    -- IMAGE PROCESSING")
    vtkimages = []
    image_log = glob.glob("%s/*.jpg" % (log_dir + '/vtk'))
    n = len(image_log)
    for i in range(n):
        image = cv2.imread(log_dir + '/vtk/%d.jpg' % (i + 1))
        vtkimages.append(image)

    position[:,:2] = position[:,:2][:,::-1]
    goal[:2] = goal[:2][::-1]
    pose[:,-1] = -pose[:,-1]
    diff = position - goal
    distance = np.linalg.norm(diff,axis=1)
    np.set_printoptions(suppress=True)
    goal_ = np.array2string(goal, precision=2, separator=',')
    for i in range(n):
        position_ = np.array2string(position[i], precision=2, separator=',')
        pose_ = np.array2string(pose[i], precision=2, separator=',')
        cv2.putText(vtkimages[i], "[GoalXYZ]    : %s m" % goal_, (25, 25), 16, 0.35, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(vtkimages[i], "[PositionXYZ] : %s m" % position_, (25, 45), 16, 0.35, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(vtkimages[i], "[AttitudeRPY] : %s deg" % pose_, (25, 65), 16, 0.35, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        if (i < 80) or (i > n-80):
            cv2.putText(vtkimages[i], "[Distance]    : ", (25, 85), 16, 0.35, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        else:
            cv2.putText(vtkimages[i], "[Distance]    : %.2f m" % distance[i], (25, 85), 16, 0.35, (0, 0, 0), 1, lineType=cv2.LINE_AA)


    H = 124
    logimages = []
    image_log = glob.glob("%s/*.jpg" % (log_dir + '/depth'))
    n = len(image_log)
    for i in range(n):
        image = cv2.imread(log_dir + '/depth/%d.jpg' % (i + 1))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (int(960 * H / 270), H), interpolation=cv2.INTER_NEAREST)
        logimages.append(image)

    from copy import deepcopy
    nvtk = len(vtkimages)
    ndpt = len(logimages)
    videoimages = deepcopy(vtkimages)
    gifimages = deepcopy(vtkimages)
    for i in range(nvtk):
        gifimages[i] = cv2.cvtColor(gifimages[i], cv2.COLOR_RGB2BGR)
        if i < 80 or i >= nvtk-80:
            videoimages[i][-H-10:-10,10:10+int(960*H/540)] = 0
            videoimages[i][-H-10:-10,-10-int(960*H/540):-10] = 0
            gifimages[i][-H - 10:-10, 10:10 + int(960 * H / 540)] = 0
            gifimages[i][-H - 10:-10, -10 - int(960 * H / 540):-10] = 0
        else:
            lidx = min((i-80)//4,len(logimages)-1)
            rgb = cv2.cvtColor(logimages[lidx][:,:int(960*H/540)], cv2.COLOR_BGR2RGB)
            dpt = cv2.cvtColor(logimages[lidx][:,int(960*H/540):], cv2.COLOR_BGR2RGB)
            videoimages[i][-H-10:-10,10:10+int(960*H/540)] = rgb
            videoimages[i][-H-10:-10,-10-int(960*H/540):-10] = dpt
            gifimages[i][-H - 10:-10, 10:10 + int(960 * H / 540)] = logimages[lidx][:, :int(960 * H / 540)]
            gifimages[i][-H - 10:-10, -10 - int(960 * H / 540):-10] = logimages[lidx][:, int(960 * H / 540):]

    if VIDEO:
        import time
        print("    -- MAKING VIDEO")
        size = videoimages[0].shape[:2][::-1]
        out = cv2.VideoWriter(log_dir+'/simlog.avi',cv2.VideoWriter_fourcc(*'MJPG'), 40, size)
        for i in range(len(videoimages)):
            time.sleep(0.02)
            out.write(videoimages[i])

    if GIF:
        print("    -- MAKING GIF")
        kargs={'duration':1/40}
        imageio.mimsave(log_dir+'/simlog.gif', gifimages, **kargs)

print("## Finish")