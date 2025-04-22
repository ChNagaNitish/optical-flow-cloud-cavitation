import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plotVelAtPoint(path,algo):
    inputVelData = np.load(path)
    mm_per_px = 0.02175955
    fps_capture = 130000
    factor = mm_per_px*1e-3*fps_capture
    fac = 8
    u = inputVelData[:,:,:,0]*factor
    #v = inputVelData[:,:,:,1]*factor
    #velMag = np.sqrt(u**2+v**2)
    y = 44
    xPts = [20, 80, 140]
    yPts = [y,y,y]
    plt.figure(1)
    for x,y in zip(xPts,yPts):
        if algo=='piv':
            x=int(x/2)
            y=int(y/2)
            fac=16
        plt.plot(u[:,y,x],'--',label='x = ' + str(x*fac*mm_per_px) + ' mm')
        plt.legend()
    plt.xlabel('Frames')
    plt.ylabel('Velocity (m/s)')
    plt.savefig(path[:-4]+'_comparePts.png')
    #plt.figure(2)
    #plt.imshow(u[0,:,:])
    #plt.scatter(xPts,yPts,c='red')
    #plt.xlabel('x (px)')
    #plt.ylabel('y (px)')
    #plt.show()

def plotVelAtLines(path,algo):
    inputVelData = np.load(path)
    mm_per_px = 0.02175955
    fps_capture = 130000
    factor = mm_per_px*1e-3*fps_capture
    fac = 8
    u = inputVelData[:,:,:,0]*factor
    v = inputVelData[:,:,:,1]*factor
    velMag = np.sqrt(u**2 + v**2)
    xPts = [20, 80, 140]
    plt.figure(1)
    for x in xPts:
        if algo=='piv':
            x=int(x/2)
            fac=16
        plt.plot(np.mean(u[:,:,x],axis=0),(u.shape[1]-1)*mm_per_px*fac-np.arange(u.shape[1])*mm_per_px*fac,label='x = ' + str(x*fac*mm_per_px) + ' mm')
        plt.legend()
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('y (mm)')
    plt.savefig(path[:-4]+'_compareLines.png')

def plotVelAtHLines(path,fps):
    inputVelData = np.load(path)
    mm_per_px = 0.02175955
    fps_capture = 130000
    factor = mm_per_px*1e-3*fps_capture
    fac = 8
    y = 44
    u = inputVelData[:,y,:,0]*factor
    x_mm = np.arange(u.shape[1])*mm_per_px*8
    nFrames = u.shape[0]
    fig, ax = plt.subplots()
    line, = ax.plot(x_mm,u[0,:])
    ax.axhline(0,color='black',linewidth=0.5)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('u [m/s]')
    ax.set_ylim([-15,25])
    title = ax.set_title(f'Frame 1/{nFrames}')
    def updateFrame(frame):
        line.set_ydata(u[frame,:])
        title.set_text(f'Frame {frame+1}/{nFrames}')
        return line, title
    ani = animation.FuncAnimation(fig=fig, func=updateFrame, frames=nFrames-1, blit=True, interval=1000/fps,repeat=False)
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(path[:-4]+'_hline.avi', writer=writer, dpi=150)

def compareAlgo(path,legend,output,fps,algo):
    paths = path.split(',')
    nFiles = len(paths)
    mm_per_px = 0.02175955
    fps_capture = 130000
    factor = mm_per_px*1e-3*fps_capture
    fac = 8
    dataAll = []
    legends = legend.split(',')
    y = 44
    for f in nFiles:
        if f==(nFiles-1) and algo=='piv':
            y=int(y/2)
        dataAll.append(np.load(paths[f])[:,y,:,0]*factor)
    x_mm = np.arange(dataAll[0].shape[1])*mm_per_px*8
    if algo=='piv':
        x_piv = np.arange(dataAll[-1].shape[1]*mm_per_px*16)
    nFrames = dataAll[0].shape[0]
    fig, ax = plt.subplots()
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
    parser.add_argument('--legend', help="input video")
    parser.add_argument('--output', help="input video")
    parser.add_argument('--algo',default='none',)
    parser.add_argument('--fps', default=10, help="fps output video")
    args = parser.parse_args()
    if args.method == 'points':
        plotVelAtPoint(args.path,args.algo)
    elif args.method == 'lines':
        plotVelAtLines(args.path,args.algo)
    elif args.method == 'hlines':
        plotVelAtHLines(args.path,int(args.fps))
    elif args.method == 'compareAlgo':
        compareAlgo(args.path,args.legend,args.output,int(args.fps))
    
