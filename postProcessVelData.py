import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
import tqdm

'''
This script is used to for postprocessing highspeed videos and velocity fields obtained from Optical Flow methods.
'''

def plotVelAtPoint(path):
    inputVelData = h5py.File(path,'r')
    attrs = dict(inputVelData['velocity'].attrs)
    mm_per_px = attrs['mm_per_px']
    fps_capture = attrs['fps_capture']
    vel_scale_fac = mm_per_px*1e-3*fps_capture
    x_scale_fac = attrs['window_width']*mm_per_px
    y = int(inputVelData['velocity'].shape[1]-2)
    xPtsmm = [2,5,10]
    xPts = [int(x/x_scale_fac) for x in xPtsmm]
    yPts = [y for x in xPtsmm]
    pointData = np.zeros((inputVelData['velocity'].shape[0]),dtype='float32')
    plt.figure(1,figsize=(15,6))
    for x,y in zip(xPts,yPts):
        for chunk_slice in inputVelData['velocity'].iter_chunks():
            np.multiply(np.mean(inputVelData['velocity'][chunk_slice][:,y,x-1:x+2,0],axis=1),vel_scale_fac,out=pointData[chunk_slice[0]])
        plt.plot(np.arange(inputVelData['velocity'].shape[0])/fps_capture,pointData,'--',label='x = ' + str(np.round(x*x_scale_fac,2)) + ' mm')
        plt.legend()
    plt.xlabel('Frames')
    plt.ylabel('Velocity (m/s)')
    plt.savefig(path[:-4]+'_comparePts.png', bbox_inches = 'tight', pad_inches = 0, transparent=False)
    plt.figure(2)
    plt.imshow(inputVelData['velocity'][200,:,:,0]*vel_scale_fac,cmap='viridis')
    plt.scatter(xPts,yPts,c='red')
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    plt.show()

def velAtLines(path):
    inputVelData = h5py.File(path,'r')
    attrs = dict(inputVelData['velocity'].attrs)
    mm_per_px = attrs['mm_per_px']
    fps_capture = attrs['fps_capture']
    vel_scale_fac = mm_per_px*1e-3*fps_capture
    x_scale_fac = attrs['window_width']*mm_per_px
    y_scale_fac = attrs['window_height']*mm_per_px
    xPtsmm = [1.5,3,5,10,15,20]
    xPts = [int(x/x_scale_fac) for x in xPtsmm]
    y_mm = np.arange(inputVelData['velocity'].shape[1]-1,-1,-1)*y_scale_fac
    lines  = np.zeros((len(xPts)+1,len(y_mm)),dtype='float32')
    lines[0,:] = y_mm
    for i,x in enumerate(xPts):
        nFrames = inputVelData['velocity'].shape[0]
        lineData = np.zeros((nFrames,len(y_mm)),dtype='float32')
        for chunk_slice in inputVelData['velocity'].iter_chunks():
            np.multiply(inputVelData['velocity'][chunk_slice][:,:,x,0],vel_scale_fac,out=lineData[chunk_slice[0],:])
        lines[i+1,:] = np.mean(lineData,axis=0)
    np.save(path[:-4]+'_lines.npy',lines)

def plotLines(path):
    lines = np.load(path)
    xPtsmm = [1.5,3,5,10,15,20]
    fig, axs = plt.subplots(1,lines.shape[0]-1, sharey=True)
    for i in range(lines.shape[0]-1):
        axs[i].plot(lines[i+1,:],lines[0,:])
        axs[i].set_xlabel('x = '+str(xPtsmm[i])+'mm')
        axs[i].minorticks_on()
        axs[i].grid(color = 'black', linestyle = '--', linewidth = 0.25)
    fig.supylabel('y [mm]')
    fig.supxlabel('Velocity [m/s]')
    plt.show()


def shock(path):
    pass

def plotVelAtHLines(path,fps):
    inputVelData = h5py.File(path,'r')
    attrs = dict(inputVelData['velocity'].attrs)
    mm_per_px = 0.05902778 #Test1
    #mm_per_px = 0.02175955 #Test5
    fps_capture = 130000
    factor = mm_per_px*1e-3*fps_capture
    fac = 0.5*attrs['window_width']
    y = int(inputVelData['velocity'].shape[1]-3)
    x_mm = np.arange(inputVelData['velocity'].shape[2])*mm_per_px*fac
    print('Loaded Velocity data')
    nFrames = 1000#u.shape[0]
    fig, ax = plt.subplots()
    line, = ax.plot(x_mm,inputVelData['velocity'][0,y,:,0]*factor)
    ax.axhline(0,color='black',linewidth=0.5)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('u [m/s]')
    ax.set_ylim([-15,25])
    title = ax.set_title(f'Frame 1/{nFrames}')
    def updateFrame(frame):
        line.set_ydata(inputVelData['velocity'][frame,y,:,0]*factor)
        title.set_text(f'Frame {frame+1}/{nFrames}')
        if frame%10==0:
            print('Done',frame,'frames')
        return line, title
    print('Started Animation')
    ani = animation.FuncAnimation(fig=fig, func=updateFrame, frames=nFrames-1, blit=True, interval=1000/fps,repeat=False)
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(path[:-4]+'_hline.avi', writer=writer, dpi=150)

def velEvolution(path,ax):
    inputVelData = h5py.File(path,'r')
    attrs = dict(inputVelData['velocity'].attrs)
    mm_per_px = attrs['mm_per_px']
    fps_capture = attrs['fps_capture']
    vel_scale_fac = mm_per_px*1e-3*fps_capture
    x_scale_fac = attrs['window_width']*mm_per_px
    y = int(inputVelData['velocity'].shape[1]-1)
    nFrames = inputVelData['velocity'].shape[0]
    evolData = np.zeros((nFrames,inputVelData['velocity'].shape[2]),dtype='float32')
    for chunk_slice in inputVelData['velocity'].iter_chunks():
        np.multiply(np.mean(inputVelData['velocity'][chunk_slice][:,y-10:y+10,:,0],axis=1),vel_scale_fac,out=evolData[chunk_slice[0],:])
    #levels = np.linspace(-20,0,41)
    contour = ax.contourf(evolData.T,levels=20,extent=[0,nFrames/fps_capture,0,inputVelData['velocity'].shape[2]*x_scale_fac],vmin=-20,vmax=0,cmap='viridis')
    ax.set_xlabel('time [sec]')
    ax.set_ylabel('x[mm]')
    inputVelData.close()
    return contour

def evolCompare(path,evolPath):
    fig, axs = plt.subplots(2)
    evolData = np.load(evolPath)
    velContour = velEvolution(path,axs[1])
    axs[0].contourf(evolData.T,cmap='Blues_r')
    axs[0].set_xlabel('Frames')
    plt.show()

def plotVelAtVLines(path,fps):
    inputVelData = np.load(path)
    mm_per_px = 0.02175955
    fps_capture = 130000
    factor = mm_per_px*1e-3*fps_capture
    fac = 8
    x = 40
    u = inputVelData[:,:,x,0]*factor
    y_mm = np.arange(u.shape[1]-1,-1,-1)*mm_per_px*fac
    nFrames = u.shape[0]
    fig, ax = plt.subplots()
    line, = ax.plot(u[0,:],y_mm)
    ax.axvline(0,color='black',linewidth=0.5)
    ax.set_xlabel('u [m/s]')
    ax.set_ylabel('y [mm]')
    ax.set_xlim([-15,25])
    title = ax.set_title(f'Frame 1/{nFrames}')
    def updateFrame(frame):
        line.set_xdata(u[frame,:])
        title.set_text(f'Frame {frame+1}/{nFrames}')
        return line, title
    ani = animation.FuncAnimation(fig=fig, func=updateFrame, frames=nFrames-1, blit=True, interval=1000/fps,repeat=False)
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(path[:-4]+'_vline.avi', writer=writer, dpi=150)

def compareAlgo(path,legend,output,fps,algo):
    paths = path.split(',')
    nFiles = len(paths)
    mm_per_px = 0.02175955
    fps_capture = 130000
    factor = mm_per_px*1e-3*fps_capture
    dataAll = []
    legends = legend.split(',')
    y = 40
    for f in range(nFiles):
        if f==(nFiles-1) and algo=='piv':
            yPIV=int(y*2)
            dataAll.append(np.load(paths[f])[:,yPIV,:,0]*factor/0.001)
        else:
            dataAll.append(np.load(paths[f])[:,y,:,0]*factor)
    x_mm = np.arange(dataAll[0].shape[1])*mm_per_px*8
    if algo=='piv':
        x_piv = np.arange(dataAll[-1].shape[1])*mm_per_px*4
    else:
        x_piv = x_mm
    nFrames = dataAll[0].shape[0]
    fig, ax = plt.subplots()
    print('Started')
    lines = []
    for f in range(nFiles):
        if f!=(nFiles-1):
            line, = ax.plot(x_mm,dataAll[f][0,:], label=legends[f])
        else:
            line, = ax.plot(x_piv,dataAll[f][0,:], label=legends[f])
        lines.append(line)
    ax.axhline(0,color='black',linewidth=0.5)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('u [m/s]')
    ax.set_ylim([-15,25])
    ax.legend(loc='upper right')
    title = ax.set_title(f'Frame 1/{nFrames}')
    def updateFrame(frame):
        for i,line in enumerate(lines):
            line.set_ydata(dataAll[i][frame,:])
            title.set_text(f'Frame {frame+1}/{nFrames}')
        return *lines, title
    ani = animation.FuncAnimation(fig=fig, func=updateFrame, frames=nFrames-1, blit=True, interval=1000/fps,repeat=False)
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(output, writer=writer, dpi=150)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', help="type of plot")
    parser.add_argument('--path', help="input video")
    parser.add_argument('--evolPath', help="input video")
    parser.add_argument('--legend', help="input video")
    parser.add_argument('--output', help="input video")
    parser.add_argument('--showPlot', default=False)
    parser.add_argument('--algo',default='none')
    parser.add_argument('--fps', default=10, help="fps output video")
    args = parser.parse_args()
    if args.method == 'points':
        plotVelAtPoint(args.path)
    elif args.method == 'lines':
        velAtLines(args.path)
    elif args.method == 'plotlines':
        plotLines(args.path)
    elif args.method == 'hlines':
        plotVelAtHLines(args.path,int(args.fps))
    elif args.method == 'velEvol':
        fig, ax = plt.subplots()
        contour = velEvolution(args.path,ax)
        plt.colorbar(contour)
        plt.show()
    elif args.method == 'evolCompare':
        evolCompare(args.path,args.evolPath)
    elif args.method == 'shock':
        shock(args.path)
    elif args.method == 'vlines':
        plotVelAtVLines(args.path,int(args.fps))
    elif args.method == 'compareAlgo':
        compareAlgo(args.path,args.legend,args.output,int(args.fps),args.algo)
    
