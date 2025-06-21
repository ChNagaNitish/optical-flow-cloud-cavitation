import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plotVelAtPoint(path,algo,showPlot):
    inputVelData = np.load(path)
    mm_per_px = 0.05902778 #Test1
    #mm_per_px = 0.02175955 #Test5
    fps_capture = 130000
    factor = mm_per_px*1e-3*fps_capture
    fac = 8
    u = inputVelData[:,:,:,0]*factor
    #v = inputVelData[:,:,:,1]*factor
    #velMag = np.sqrt(u**2+v**2)
    #y = 40 #Test1
    y = 22
    #y = 44 #Test5
    xPts = [5, 60, 120] #Test1
    #xPts = [20, 80, 140] #Test5
    yPts = [y,y,y]
    plt.figure(1,figsize=(15,6))
    for x,y in zip(xPts,yPts):
        if algo=='piv':
            x=int(x/2)
            y=int(y/2)
            fac=16
        plt.plot(u[:,y,x],'--',label='x = ' + str(x*fac*mm_per_px) + ' mm')
        plt.legend()
    plt.xlabel('Frames')
    plt.ylabel('Velocity (m/s)')
    plt.savefig(path[:-4]+'_comparePts.png', bbox_inches = 'tight', pad_inches = 0, transparent=False)
    if bool(showPlot):
        plt.figure(2)
        plt.imshow(u[30,:,:])
        plt.scatter(xPts,yPts,c='red')
        plt.xlabel('x (px)')
        plt.ylabel('y (px)')
        plt.show()

def plotVelAtLines(path,algo):
    inputVelData = np.load(path)
    mm_per_px = 0.02175955
    fps_capture = 130000
    factor = mm_per_px*1e-3*fps_capture
    u = inputVelData[:,:,:,0]*factor
    #v = inputVelData[:,:,:,1]*factor
    #velMag = np.sqrt(u**2 + v**2)
    xPts = [20, 80, 140]
    plt.figure(1)
    for x in xPts:
        if algo=='piv':
            y_mm = np.arange(u.shape[1]-1,-1,-1)*mm_per_px*4
            plt.plot(np.nanmean(u[:,:,int(x*2)],axis=0)/0.001,y_mm,label='x = ' + str(x*8*mm_per_px) + ' mm')
        else:
            y_mm = np.arange(u.shape[1]-1,-1,-1)*mm_per_px*8
            plt.plot(np.mean(u[:,:,x],axis=0),y_mm,label='x = ' + str(x*8*mm_per_px) + ' mm')
        plt.legend()
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('y (mm)')
    plt.xticks(np.arange(-4,27,2))
    plt.minorticks_on()
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.25)
    plt.savefig(path[:-4]+'_compareLines.png')

def plotVelAtHLines(path,fps):
    inputVelData = np.load(path)
    mm_per_px = 0.02175955
    fps_capture = 130000
    factor = mm_per_px*1e-3*fps_capture
    fac = 8
    y = 44
    u = inputVelData[:,y,:,0]*factor
    x_mm = np.arange(u.shape[1])*mm_per_px*fac
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
    parser.add_argument('--legend', help="input video")
    parser.add_argument('--output', help="input video")
    parser.add_argument('--showPlot', default=False)
    parser.add_argument('--algo',default='none')
    parser.add_argument('--fps', default=10, help="fps output video")
    args = parser.parse_args()
    if args.method == 'points':
        plotVelAtPoint(args.path,args.algo,args.showPlot)
    elif args.method == 'lines':
        plotVelAtLines(args.path,args.algo)
    elif args.method == 'hlines':
        plotVelAtHLines(args.path,int(args.fps))
    elif args.method == 'vlines':
        plotVelAtVLines(args.path,int(args.fps))
    elif args.method == 'compareAlgo':
        compareAlgo(args.path,args.legend,args.output,int(args.fps),args.algo)
    
