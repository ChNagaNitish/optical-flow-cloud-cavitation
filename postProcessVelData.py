import numpy as np
import argparse
import matplotlib.pyplot as plt
import 

def plotVelAtPoint(path):
    inputVelData = np.load(path)
    mm_per_px = 0.02175955
    fps_capture = 130000
    factor = mm_per_px*1e-3*fps_capture
    u = inputVelData[:,:,:,0]*factor
    v = inputVelData[:,:,:,1]*factor
    #velMag = np.sqrt(u**2+v**2)
    y = 44
    xPts = [20, 80, 140]
    yPts = [y,y,y]
    plt.figure(1)
    for x,y in zip(xPts,yPts):
        plt.plot(u[:,y,x],'--',label='x = ' + str(x*8*mm_per_px) + ' mm')
        plt.legend()
    plt.xlabel('Frames')
    plt.ylabel('Velocity (m/s)')
    plt.savefig(path[:-4]+'_comparePts.png')
    plt.figure(2)
    plt.imshow(u[0,:,:])
    plt.scatter(xPts,yPts,c='red')
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    #plt.show()

def plotVelAtLines(path):
    inputVelData = np.load(path)
    mm_per_px = 0.02175955
    fps_capture = 130000
    factor = mm_per_px*1e-3*fps_capture
    u = inputVelData[:,:,:,0]*factor
    v = inputVelData[:,:,:,1]*factor
    velMag = np.sqrt(u**2 + v**2)
    xPts = [20, 80, 140]
    plt.figure(1)
    for x in xPts:
        plt.plot(np.mean(u[:,:,x],axis=0),(u.shape[1]-1)*mm_per_px*8-np.arange(u.shape[1])*mm_per_px*8,label='x = ' + str(x*8*mm_per_px) + ' mm')
        plt.legend()
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('y (mm)')
    plt.savefig(path[:-4]+'_compareLines.png')

def plotVelAtHLines(path):
    inputVelData = np.load(path)
    mm_per_px = 0.02175955
    fps_capture = 130000
    factor = mm_per_px*1e-3*fps_capture
    u = inputVelData[:,:,:,0]*factor
    v = inputVelData[:,:,:,1]*factor
    velMag = np.sqrt(u**2 + v**2)
    y = 44
    nFrames = u.shape[0]
    plt.figure(1)
    for i in range(nFrames):
        plt.plot(np.arange(u.shape[2])*mm_per_px*8,u[i,y,:])
        plt.ylabel('Velocity (m/s)')
        plt.xlabel('x (mm)')
        plt.ylim([-10,20])
    #plt.savefig(path[:-4]+'_compareHLines.png')

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
    parser.add_argument('--method', help="type of plot")
    parser.add_argument('--path', help="input video")
    parser.add_argument('--path2', help="input video")
    args = parser.parse_args()
    if args.method == 'points':
        plotVelAtPoint(args.path)
    if args.method == 'lines':
        plotVelAtLines(args.path)
    elif args.method == 'compare':
        input1 = np.load(args.path)
        input2 = np.load(args.path2)
        compareAtPoint(input1,input2)
    
