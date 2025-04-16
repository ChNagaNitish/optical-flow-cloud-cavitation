import sys
sys.path.append('core')
import io
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
#import h5py

from raft import RAFT
from utils.utils import InputPadder

#from openpiv import tools, pyprocess, validation, filters, scaling
#############################################################################################

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(frame):
    img = np.array(frame).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def outputVideoWriter(inputVid,videoName):
    original_fps = inputVid.get(cv2.CAP_PROP_FPS)
    new_fps = original_fps #* 10
    frame_width = int(inputVid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(inputVid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    return cv2.VideoWriter(videoName, fourcc, new_fps, (frame_width, frame_height))

'''def quiverImage(flow, curr_frame, step=8, arrow_color=(0, 0, 255), arrow_thickness=1):  # Color: Red (BGR)
    """
    Draws a quiver plot on the current frame using OpenCV.

    Args:
        flow: Optical flow field (height, width, 2).
        curr_frame: The current video frame (height, width, 3).
        step: Spacing between arrows.
        arrow_color: Color of the arrows (BGR).
        arrow_thickness: Thickness of the arrow lines.

    Returns:
        The frame with the quiver plot drawn on it.
    """

    h, w = curr_frame.shape[:2]
    y, x = np.mgrid[0:h:step, 0:w:step].astype(int)  # Ensure integer indices

    u = flow[y, x, 0].astype(np.float32)
    v = flow[y, x, 1].astype(np.float32)

    for i in range(y.shape[0]):
        for j in range(x.shape[1]):
            pt1 = (x[i, j], y[i, j])
            pt2_x = int(x[i, j] + u[i, j])
            pt2_y = int(y[i, j] + v[i, j])
            pt2 = (pt2_x, pt2_y)
            cv2.arrowedLine(curr_frame, pt1, pt2, arrow_color, arrow_thickness, tipLength=0.2)  # Adjust tipLength as needed

    return curr_frame'''

def quiverImage(flow,curr_frame):
    h, w = curr_frame.shape[:2]  # Get image height and width
    # Create a grid of points for the quiver plot
    step = 15 # Adjust step (15) for density of arrows
    y, x = np.mgrid[0:h:step, 0:w:step] 
    u = flow[y, x, 0]
    v = flow[y, x, 1]
    # Create the quiver plot
    plt.figure(figsize=(w / 100, h / 100), dpi=100) # Adjust figure size to match image dimensions
    plt.imshow(curr_frame, cmap='gray')
    plt.quiver(x, y, u, -v, color='red')
    #plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.axis('off')  # Turn off axis labels and ticks
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0) # remove extra white space around plot.
    # Save the plot to a memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)  # Save to buffer
    buf.seek(0)
    # Read the image from the buffer using cv2
    quiver_image = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
    # Resize the quiver image to match the original image size (if necessary)
    quiver_image = cv2.resize(quiver_image, (w, h))
    plt.close()
    del u,v,flow,curr_frame,buf
    return quiver_image

def average_window(arr, window_size):
    h, w = arr.shape[:2]
    window_height, window_width = window_size
    kernel = np.ones(window_size) / (window_height*window_width)
    pad_height = window_height // 2
    pad_width = window_width // 2
    result = np.zeros(arr.shape)#np.zeros((int(h/window_size[0]), int(w/window_size[1]), 2))
    for i in range(0,h,window_size[0]):
        for j in range(0,w,window_size[1]):
            window = arr[i:i+window_size[0], j:j+window_size[1],:]
            result[i, j,:] = np.mean(window,axis=(0,1))
    del arr
    return result[::window_size[0],::window_size[1],:]
def avg_window(flow, window_size):
    h, w = flow.shape[:2]
    window_height, window_width = window_size
    kernel = np.ones(window_size) / (window_height*window_width)
    pad_height = window_height // 2
    pad_width = window_width // 2
    padded_arr = np.pad(flow, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='reflect')
    averaged_arr = np.zeros((h // window_height, w // window_width, 2), dtype=flow.dtype)
    for channel in range(2):
        averaged_arr[:, :, channel] = cv2.filter2D(padded_arr[:, :, channel], -1, kernel)[pad_height:-pad_height:window_height, pad_width:-pad_width:window_width]
    return averaged_arr

def raftOpticalFlow(args, inputVid, outputVid,saveVel):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        ret, prev_frame = inputVid.read()
        frame_count = int(inputVid.get(cv2.CAP_PROP_FRAME_COUNT))
        velData = []
        for frame_index in tqdm.tqdm(range(1,frame_count), desc="Processing Video Frame Pairs"):
            ret, curr_frame = inputVid.read()
            image1 = load_image(prev_frame)
            image2 = load_image(curr_frame)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flo = flow_up[0].permute(1,2,0).cpu().numpy()
            velData.append(average_window(flo,[8,8]))
            #flowImage = quiverImage(flo,curr_frame)
            #outputVid.write(flowImage)
            prev_frame = curr_frame
        inputVid.release()
        outputVid.release()
        np.save(saveVel,np.array(velData))
    

def farnebackMethod(inputVid,outputVid,saveVel):
    ret, prev_frame = inputVid.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = int(inputVid.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = prev_frame.shape[:2]
    window_size = [8,8]
    window_height, window_width = window_size
    kernel = np.ones(window_size) / (window_height*window_width)
    velData = np.zeros((frame_count-1, h//window_height, w//window_width, 2), dtype='float64')
    flow=None
    for frame_index in tqdm.tqdm(range(1,frame_count), desc="Processing Video Frame Pairs"):
        ret, curr_frame = inputVid.read()
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0)
        #velData.append(average_window(flow,[8,8]))
        #flow = flow[::8,::8]
        #flowImage = quiverImage(flow,curr_frame)
        #outputVid.write(flowImage)
        velData[frame_index - 1] = avg_window(flow,window_size)
        prev_gray = curr_gray
    inputVid.release()
    outputVid.release()
    np.save(saveVel,velData)

'''def openPIV(inputVid, outputVid):
    # Parameters for openPIV
    dt = 1.0 / 130000.0  # Time between frames (adjust based on your video's frame rate)
    winsize = 16
    searchsize = 32
    overlap = 8
    sig2noise_threshold = 1.05
    outlier_threshold = 2
    mm_per_pixel = 0.02175955
    scaling_factor = 1/mm_per_pixel
    extract_every_n_frames = 1  # Process every nth frame
    output_video_fps = 10  # Frames per second of the output video
    ret, prev_frame = inputVid.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = int(inputVid.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_index in tqdm.tqdm(range(1,frame_count), desc="Processing Video Frame Pairs"):
        ret, curr_frame = inputVid.read()
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        h, w = curr_frame.shape[:2]
        u0, v0, sig2noise = pyprocess.extended_search_area_piv(prev_gray.astype(np.int32),curr_gray.astype(np.int32),window_size=winsize,overlap=overlap,dt=dt,search_area_size=searchsize,sig2noise_method='peak2peak')
        x, y = pyprocess.get_coordinates(image_size=prev_gray.shape,search_area_size=searchsize,overlap=overlap)
        invalid_mask = validation.sig2noise_val(sig2noise,threshold = sig2noise_threshold)
        u2, v2 = filters.replace_outliers(u0, v0,invalid_mask,method='localmean',max_iter=3,kernel_size=3)
        x, y, u3, v3 = scaling.uniform(x, y, u2, v2,scaling_factor = scaling_factor)
        x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)
        fig, ax = plt.subplots()
        extent = [x[0,0],x[-1,-1],y[-1,-1],y[0,0]]
        plt.imshow(prev_gray,extent=extent)
        plt.quiver(x,y,u3,v3,color='red')
        plt.axis('off')  # Turn off axis labels and ticks
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0) # remove extra white space around plot.
        # Save the plot to a memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)  # Save to buffer
        buf.seek(0)
        # Read the image from the buffer using cv2
        quiver_image = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
        # Resize the quiver image to match the original image size (if necessary)
        quiver_image = cv2.resize(quiver_image, (w, h))
        plt.close()
        outputVid.write(quiver_image)
        prev_gray = curr_gray'''

#############################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='farneback', help="raft or farneback or piv")
    parser.add_argument('--model', help="raft-model or farneback-case or piv-case")
    parser.add_argument('--path', help="input video")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    inputVid = cv2.VideoCapture(args.path)
    if args.method == 'raft':
        new_args = argparse.Namespace()
        setattr(new_args, 'model', args.model)
        setattr(new_args, 'path', args.path)
        setattr(new_args, 'small', args.small)
        setattr(new_args, 'mixed_precision', args.mixed_precision)
        setattr(new_args, 'alternate_corr', args.alternate_corr)
        outputVideoPath = args.path[:-4]+'_'+args.model.split("/")[-1][:-4]+'.avi'
        outputVid = outputVideoWriter(inputVid,outputVideoPath)
        outputVelocityPath = args.path[:-4]+'_'+args.model.split("/")[-1][:-4]+'.npy'
        raftOpticalFlow(new_args, inputVid, outputVid, outputVelocityPath)
    elif args.method == 'farneback':
        outputVideoPath = args.path[:-4]+'_'+args.method+'_'+args.model+'.avi'
        outputVid = outputVideoWriter(inputVid,outputVideoPath)
        outputVelocityPath = args.path[:-4]+'_'+args.method+'_'+args.model+'.npy'
        farnebackMethod(inputVid, outputVid, outputVelocityPath)
    '''elif args.method == 'piv':
        outputVideoPath = args.path[:-4]+'_'+args.method+'.avi'
        outputVid = outputVideoWriter(inputVid,outputVideoPath)
        openPIV(inputVid, outputVid)'''
#############################################################################################
