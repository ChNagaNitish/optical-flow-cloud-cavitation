import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

def plotVelAtPoint(path):
    inputVelData = np.load(path)
    mm_per_px = 0.02175955
    fps_capture = 130000
    factor = mm_per_px*1e-3*fps_capture
    u1 = inputVelData[:,:,:,0]*factor
    #velMag = np.sqrt(u**2+v**2)
    #pointX = 
    y = 44
    xPts = [20, 80, 140]
    yPts = [y,y,y]
    plt.figure(1)
    for x,y in zip(xPts,yPts):
        #plt.imshow(u1[0,:,:])
        #plt.plot(u1[:,y,x],'-',label='RAFT')
        plt.plot(u1[:,y,x],'--',label='x = ' + str(x*mm_per_px) + ' mm')
        plt.legend()
    plt.xlabel('Frames')
    plt.ylabel('Velocity (m/s)')
    plt.savefig(path[:-4]+'_comparePts.png')
    plt.figure(2)
    plt.imshow(u1[0,:,:])
    plt.scatter(xPts,yPts,c='red')
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    #plt.show()

def compareAtPoint(input1, input2):
    mm_per_px = 0.02175955
    fps_capture = 130000
    factor = mm_per_px*1e-3*fps_capture
    u1 = input1[:,:,:,0]*factor
    u2 = input2[:,:,:,0]*factor
    #velMag = np.sqrt(u**2+v**2)
    #pointX = 
    y = 40
    xPts = [20, 100, 180]
    yPts = [y,y,y]
    plt.figure(1)
    for x,y in zip(xPts,yPts):
        plt.plot(u1[:,y,x],'-',label='RAFT')
        plt.plot(u2[:,y,x],'--',label='Farneback')
        plt.legend()
    plt.figure(2)
    plt.imshow(u1[0,:,:])
    plt.scatter(xPts,yPts,c='red')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', help="input video")
    parser.add_argument('--path2', help="input video")
    args = parser.parse_args()
    #input1 = np.load(args.path1)
    #input2 = np.load(args.path2)
    plotVelAtPoint(args.path1)
    #compareAtPoint(input1,input2)
    