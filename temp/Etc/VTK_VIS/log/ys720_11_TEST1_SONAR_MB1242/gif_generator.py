import numpy as np
import glob, argparse
import imageio, cv2
import os

parser = argparse.ArgumentParser("MakeGIF_Log")
parser.add_argument('-fps', default=3, help='frames per second')
args = parser.parse_args()

if __name__=='__main__':

    print("## Start making gif file...")
    log_dir = os.path.dirname(os.path.abspath(__file__))
    sidx = log_dir.find("log/")
    filename = log_dir[sidx:] + "/log.gif"
    print("  [File] %s"%filename)
    images = []
    image_log = glob.glob("%s/*.jpg"%(log_dir+'/depth'))
    n = len(image_log)
    for i in range(n):
        image = cv2.imread(log_dir+'/depth/%d.jpg'%(i+1))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    kargs={'duration':1/args.fps}
    imageio.mimsave(log_dir+'/log.gif', images, **kargs)
    print("## Finish")
