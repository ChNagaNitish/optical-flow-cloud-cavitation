import sys
sys.path.append('core')
import argparse
import cv2
import numpy as np
import torch
import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import h5py



#############################################################################################

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(frame):
    img = np.array(frame).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def raftOpticalFlow(args, inputVid, window_height, window_width, roi, saveVel, mm_per_px, fps_capture):
    from raft import RAFT
    from utils.utils import InputPadder
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        ret, prev_frame = inputVid.read()
        if roi[-1]==-1:
            if roi[1]==-1:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)[roi[0]:,roi[2]:]
            else:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)[roi[0]:roi[1],roi[2]:]
        else:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)[roi[0]:roi[1],roi[2]:roi[3]]
        frame_count = int(inputVid.get(cv2.CAP_PROP_FRAME_COUNT))
        h, w = prev_frame.shape[:2]
        pad_height = window_height // 2
        pad_width = window_width // 2
        kernel = np.ones([window_height, window_width]) / (window_height*window_width)
        data_shape = (frame_count-1, h//window_height, w//window_width, 2)
        chunk_shape = (10, h//window_height, w//window_width, 2)
        flow=None
        f = h5py.File(saveVel, 'w')
        velData = f.create_dataset('velocity',shape=data_shape,chunks=chunk_shape,dtype='float32')
        for frame_index in tqdm.tqdm(range(1,frame_count), desc="Processing Video Frame Pairs"):
            ret, curr_frame = inputVid.read()
            if roi[-1]==-1:
                if roi[1]==-1:
                    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)[roi[0]:,roi[2]:]
                else:
                    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)[roi[0]:roi[1],roi[2]:]
            else:
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)[roi[0]:roi[1],roi[2]:roi[3]]
            image1 = load_image(prev_frame)
            image2 = load_image(curr_frame)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow = flow_up[0].permute(1,2,0).cpu().numpy()
            if window_width!=1:
                padded_arr = np.pad(flow, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='reflect')
                averaged_arr = np.zeros((h // window_height, w // window_width, 2), dtype=flow.dtype)
                for channel in range(2):
                    averaged_arr[:, :, channel] = cv2.filter2D(padded_arr[:, :, channel], -1, kernel)[pad_height:-pad_height:window_height, pad_width:-pad_width:window_width]
                velData[frame_index-1,:,:,:] = averaged_arr
            else:
                velData[frame_index-1,:,:,:] = flow
            prev_frame = curr_frame
        velData.attrs['window_height'] = window_height
        velData.attrs['window_width'] = window_width
        velData.attrs['mm_per_px'] = mm_per_px
        velData.attrs['fps_capture'] = fps_capture
        f.close()
        inputVid.release()
    

def farnebackMethod(inputVid, window_height, window_width, roi, saveVel, mm_per_px, fps_capture):
    ret, prev_frame = inputVid.read()
    if roi[-1]==-1:
        if roi[1]==-1:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)[roi[0]:,roi[2]:]
        else:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)[roi[0]:roi[1],roi[2]:]
    else:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)[roi[0]:roi[1],roi[2]:roi[3]]
    frame_count = int(inputVid.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = prev_gray.shape
    pad_height = window_height // 2
    pad_width = window_width // 2
    kernel = np.ones([window_height, window_width]) / (window_height*window_width)
    data_shape = (frame_count-1, h//window_height, w//window_width, 2)
    chunk_shape = (10, h//window_height, w//window_width, 2)
    #velData = np.zeros((frame_count-1, h//window_height, w//window_width, 2), dtype='float32')
    flow=None
    f = h5py.File(saveVel, 'w')
    velData = f.create_dataset('velocity',shape=data_shape,chunks=chunk_shape,dtype='float32')
    for frame_index in tqdm.tqdm(range(1,frame_count), desc="Processing Video Frame Pairs"):
        ret, curr_frame = inputVid.read()
        if roi[-1]==-1:
            if roi[1]==-1:
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)[roi[0]:,roi[2]:]
            else:
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)[roi[0]:roi[1],roi[2]:]
        else:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)[roi[0]:roi[1],roi[2]:roi[3]]
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0)
        if window_width!=1:
            padded_arr = np.pad(flow, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='reflect')
            averaged_arr = np.zeros((h // window_height, w // window_width, 2), dtype=flow.dtype)
            for channel in range(2):
                averaged_arr[:, :, channel] = cv2.filter2D(padded_arr[:, :, channel], -1, kernel)[pad_height:-pad_height:window_height, pad_width:-pad_width:window_width]
            velData[frame_index-1,:,:,:] = averaged_arr
        else:
            velData[frame_index-1,:,:,:] = flow
        prev_gray = curr_gray
    velData.attrs['window_height'] = window_height
    velData.attrs['window_width'] = window_width
    velData.attrs['mm_per_px'] = mm_per_px
    velData.attrs['fps_capture'] = fps_capture
    f.close()
    inputVid.release()
    #np.save(saveVel,velData)

#############################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='farneback', help="raft or farneback or piv")
    parser.add_argument('--model', default='c1',help="raft-model or farneback-case or piv-case")
    parser.add_argument('--path', help="input video")
    parser.add_argument('--win_h', default=4, help="Averging window - height")
    parser.add_argument('--win_w', default=4, help="Averging window - width")
    parser.add_argument('--imgScale', default=0.001, help="Image Scale(mm/px)")
    parser.add_argument('--fpsCam', default=130000, help="FPS of camera used for capturing")
    parser.add_argument('--roi', type=int, nargs=4, default=[0,-1,0,-1], help="Region of interest in pixel indices - hs he ws we")
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
        outputVelocityPath = args.path[:-4]+'_'+args.model.split("/")[-1][:-4]+'.npy'
        raftOpticalFlow(new_args, inputVid, int(args.win_h), int(args.win_w), args.roi, outputVelocityPath, float(args.imgScale), int(args.fpsCam))
    elif args.method == 'farneback':
        outputVelocityPath = args.path[:-4]+'_fb'+args.model+'.h5'
        farnebackMethod(inputVid, int(args.win_h), int(args.win_w), args.roi, outputVelocityPath, float(args.imgScale), int(args.fpsCam))
#############################################################################################
