import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
import tqdm

'''
This script is used to for postprocessing highspeed videos and velocity fields obtained from Optical Flow methods.
'''

def set_journal_style():
    """
    Applies a professional, journal-quality style to Matplotlib plots.
    """
    plt.style.use('seaborn-v0_8-paper') # A good base style
    
    plt.rcParams.update({
        # Font settings for clarity and consistency
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"], # Or "Helvetica"
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 12,

        # Use LaTeX for text rendering for a professional look
        "text.usetex": True,
        "text.latex.preamble": r'\usepackage{amsmath}',

        # Figure and axes settings
        "figure.dpi": 300,
        "figure.edgecolor": "black",
        "figure.facecolor": "white",

        # Axes properties
        "axes.labelpad": 6.0,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.color": "lightgray",
        "grid.linestyle": ":",

        # Tick properties
        "xtick.major.size": 4,
        "xtick.minor.size": 2,
        "xtick.direction": "in",
        "ytick.major.size": 4,
        "ytick.minor.size": 2,
        "ytick.direction": "in",
        
        # Legend properties
        "legend.frameon": True,
        "legend.framealpha": 0.8,
        "legend.facecolor": "white",
        "legend.edgecolor": "black",
        
        # Savefig settings
        "savefig.dpi": 300,
        "savefig.transparent": True,
        "savefig.format": "png", # Vector format for scalability
        "savefig.bbox": "tight", # Automatically adjust plot to fit
    })


def plotVelAtPoint(path):
    inputVelData = h5py.File(path,'r')
    attrs = dict(inputVelData['velocity'].attrs)
    mm_per_px = attrs['mm_per_px']
    fps_capture = attrs['fps_capture']
    vel_scale_fac = mm_per_px*1e-3*fps_capture
    x_scale_fac = attrs['window_width']*mm_per_px
    y = int(inputVelData['velocity'].shape[1]-2)
    xPtsmm = [2,10,20]
    xPts = [int(x/x_scale_fac) for x in xPtsmm]
    yPts = [y for x in xPtsmm]
    pointsData = np.zeros((len(xPts),inputVelData['velocity'].shape[0]),dtype='float32')
    for chunk_slice in inputVelData['velocity'].iter_chunks():
        i=0
        for x,y in zip(xPts,yPts):
            np.multiply(np.mean(inputVelData['velocity'][chunk_slice][:,y,x-1:x+2,0],axis=1),vel_scale_fac,out=pointsData[i,chunk_slice[0]])
            i+=1
    #set_journal_style()
    plt.figure(figsize=(15,6))
    fig,axs = plt.subplots(len(xPts),1)
    for i,x in enumerate(xPts):
        axs[i].plot(np.arange(inputVelData['velocity'].shape[0])/fps_capture,pointsData[i,:],'--')
        axs[i].set_ylabel('x = ' + str(np.round(x*x_scale_fac,2)) + ' mm')
    fig.supxlabel('time [sec]')
    fig.supylabel('u [m/s]')
    plt.savefig(path[:-3]+'_pts.png', bbox_inches = 'tight', pad_inches = 0, transparent=False)
    plt.close()
    '''plt.figure(2)
    plt.imshow(inputVelData['velocity'][200,:,:,0]*vel_scale_fac,cmap='viridis')
    plt.scatter(xPts,yPts,c='red')
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')'''
    #plt.show()

def velAtVLinesAvg(path,method):
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
    np.save(path[:-3]+'_'+method+'.npy',lines)

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
    plt.savefig(path[:-3]+'png')


def plotVelAtHLines(path,fps):
    inputVelData = h5py.File(path,'r')
    attrs = dict(inputVelData['velocity'].attrs)
    mm_per_px = attrs['mm_per_px']
    fps_capture = attrs['fps_capture']
    vel_scale_fac = mm_per_px*1e-3*fps_capture
    x_scale_fac = attrs['window_width']*mm_per_px
    y_scale_fac = attrs['window_height']*mm_per_px
    y = inputVelData['velocity'].shape[1] - 1 # Choosing horizontal line close to bottom wall
    x_mm = np.arange(inputVelData['velocity'].shape[2])*x_scale_fac
    print('Loaded Velocity data')
    nFrames = inputVelData['velocity'].shape[0]
    fig, ax = plt.subplots()
    line, = ax.plot(x_mm,inputVelData['velocity'][0,y,:,0]*vel_scale_fac)
    ax.axhline(0,color='black',linewidth=0.5)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('u [m/s]')
    ax.set_ylim([-15,25])
    title = ax.set_title(f'Frame 1/{nFrames}')
    def updateFrame(frame):
        line.set_ydata(inputVelData['velocity'][frame,y,:,0]*vel_scale_fac)
        title.set_text(f'Frame {frame+1}/{nFrames}')
        if frame%10==0:
            print('Done',frame,'frames')
        return line, title
    print('Started Animation')
    ani = animation.FuncAnimation(fig=fig, func=updateFrame, frames=nFrames-1, blit=True, interval=1000/fps,repeat=False)
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(path[:-3]+'_hline.avi', writer=writer, dpi=150)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', help="type of plot")
    parser.add_argument('--path', help="input video")
    parser.add_argument('--evolPath', help="input video")
    parser.add_argument('--fps', default=10, help="fps output video")
    args = parser.parse_args()
    if args.method == 'points':
        plotVelAtPoint(args.path)
    elif args.method == 'vLinesAvg':
        velAtVLinesAvg(args.path,args.method)
        plotLines(args.path[:-3]+'_'+args.method+'.npy')
    elif args.method == 'hline':
        plotVelAtHLines(args.path,int(args.fps))
    elif args.method == 'velEvol':
        fig, ax = plt.subplots()
        contour = velEvolution(args.path,ax)
        plt.colorbar(contour)
        plt.show()
    elif args.method == 'evolCompare':
        evolCompare(args.path,args.evolPath)
    elif args.method == 'vline':
        plotVelAtVLines(args.path,int(args.fps))
    
