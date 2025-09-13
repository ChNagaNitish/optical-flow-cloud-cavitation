import sys
sys.path.append('core')
import argparse
import cv2
import numpy as np
import torch
import tqdm
import h5py

#############################################################################################

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def crop_frame(frame, roi):
    """
    Crops a frame using the given ROI tuple (y_start, y_end, x_start, x_end).
    Handles the -1 convention from the original code, where -1 means 'to the end'.
    """
    y_start, y_end, x_start, x_end = roi
    y_slice = slice(y_start, y_end if y_end != -1 else None)
    x_slice = slice(x_start, x_end if x_end != -1 else None)
    return frame[y_slice, x_slice]

def apply_clahe(frame,clip_limit,tile_size):
    # Convert to LAB color space, apply CLAHE to L-channel, and convert back to BGR
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def load_image(frame):
    """Prepares a single frame for the RAFT model."""
    # Convert BGR frame to RGB, then to a tensor
    img = frame[..., ::-1].copy()
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def raftOpticalFlow(args, inputVid):
    """
    Calculates optical flow using the RAFT model.
    Includes frame cropping and optional CLAHE preprocessing.
    """
    from raft import RAFT
    from utils.utils import InputPadder

    # Unpack arguments for clarity
    roi = args.roi
    window_height = int(args.win_h)
    window_width = int(args.win_w)
    mm_per_px = float(args.imgScale)
    fps_capture = int(args.fpsCam)
    outputVelocityPath = args.path[:-4] + '_' + args.model.split("/")[-1][:-4] + '.h5'

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        ret, prev_frame_raw = inputVid.read()
        if not ret:
            print("Error: Could not read the first frame.")
            return

        # --- Preprocess the first frame ---
        prev_frame_cropped = crop_frame(prev_frame_raw, roi)

        if args.use_clahe:
            prev_frame_processed = apply_clahe(prev_frame_cropped,args.clahe_clip_limit,args.clahe_tile_size)
        else:
            prev_frame_processed = prev_frame_cropped

        frame_count = int(inputVid.get(cv2.CAP_PROP_FRAME_COUNT))
        h, w = prev_frame_processed.shape[:2]

        # Prepare for window averaging and data saving
        pad_height = window_height // 2
        pad_width = window_width // 2
        kernel = np.ones([window_height, window_width]) / (window_height * window_width)
        data_shape = (frame_count - 1, h // window_height, w // window_width, 2)
        chunk_shape = (10, h // window_height, w // window_width, 2)
        
        with h5py.File(outputVelocityPath, 'w') as f:
            velData = f.create_dataset('velocity', shape=data_shape, chunks=chunk_shape, dtype='float32')

            for frame_index in tqdm.tqdm(range(1, frame_count), desc="Processing Video with RAFT"):
                ret, curr_frame_raw = inputVid.read()
                if not ret:
                    break

                # --- Preprocess the current frame ---
                curr_frame_cropped = crop_frame(curr_frame_raw, roi)

                if args.use_clahe:
                    curr_frame_processed = apply_clahe(curr_frame_cropped,args.clahe_clip_limit,args.clahe_tile_size)
                else:
                    curr_frame_processed = curr_frame_cropped

                # Load images for RAFT
                image1 = load_image(prev_frame_processed)
                image2 = load_image(curr_frame_processed)
                
                # Get optical flow
                _, flow_up = model(image1, image2, iters=20, test_mode=True)
                flow = flow_up[0].permute(1, 2, 0).cpu().numpy()

                # Perform window averaging if needed
                if window_width > 1 or window_height > 1:
                    padded_arr = np.pad(flow, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='reflect')
                    averaged_arr = np.zeros((h // window_height, w // window_width, 2), dtype=flow.dtype)
                    for channel in range(2):
                        averaged_arr[:, :, channel] = cv2.filter2D(padded_arr[:, :, channel], -1, kernel)[pad_height:-pad_height:window_height, pad_width:-pad_width:window_width]
                    velData[frame_index - 1, :, :, :] = averaged_arr
                else:
                    velData[frame_index - 1, :, :, :] = flow
                
                prev_frame_processed = curr_frame_processed

            # Save metadata
            velData.attrs['window_height'] = window_height
            velData.attrs['window_width'] = window_width
            velData.attrs['mm_per_px'] = mm_per_px
            velData.attrs['fps_capture'] = fps_capture
            velData.attrs['roi'] = roi

    inputVid.release()

def farnebackMethod(args, inputVid):
    """
    Calculates optical flow using the Farneback method.
    Includes frame cropping and optional CLAHE preprocessing.
    """
    # Unpack arguments
    roi = args.roi
    window_height = int(args.win_h)
    window_width = int(args.win_w)
    mm_per_px = float(args.imgScale)
    fps_capture = int(args.fpsCam)
    outputVelocityPath = args.path[:-4] + '_fb' + args.model + '.h5'
    
    ret, prev_frame_raw = inputVid.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
        
    # --- Preprocess the first frame ---
    prev_gray = cv2.cvtColor(prev_frame_raw, cv2.COLOR_BGR2GRAY)
    prev_gray_cropped = crop_frame(prev_gray, roi)
    
    if args.use_clahe:
        prev_gray_processed = apply_clahe(prev_frame_cropped,args.clahe_clip_limit,args.clahe_tile_size)
    else:
        prev_gray_processed = prev_gray_cropped

    frame_count = int(inputVid.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = prev_gray_processed.shape
    
    # Prepare for window averaging and data saving
    pad_height = window_height // 2
    pad_width = window_width // 2
    kernel = np.ones([window_height, window_width]) / (window_height * window_width)
    data_shape = (frame_count - 1, h // window_height, w // window_width, 2)
    chunk_shape = (10, h // window_height, w // window_width, 2)
    flow = None
    
    with h5py.File(outputVelocityPath, 'w') as f:
        velData = f.create_dataset('velocity', shape=data_shape, chunks=chunk_shape, dtype='float32')
        
        for frame_index in tqdm.tqdm(range(1, frame_count), desc="Processing Video with Farneback"):
            ret, curr_frame_raw = inputVid.read()
            if not ret:
                break

            # --- Preprocess the current frame ---
            curr_gray = cv2.cvtColor(curr_frame_raw, cv2.COLOR_BGR2GRAY)
            curr_gray_cropped = crop_frame(curr_gray, roi)
            
            if args.use_clahe:
                curr_gray_processed = clahe.apply(curr_gray_cropped,args.clahe_clip_limit,args.clahe_tile_size)
            else:
                curr_gray_processed = curr_gray_cropped
            
            # Get optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray_processed, curr_gray_processed, flow, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Perform window averaging if needed
            if window_width > 1 or window_height > 1:
                padded_arr = np.pad(flow, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='reflect')
                averaged_arr = np.zeros((h // window_height, w // window_width, 2), dtype=flow.dtype)
                for channel in range(2):
                    averaged_arr[:, :, channel] = cv2.filter2D(padded_arr[:, :, channel], -1, kernel)[pad_height:-pad_height:window_height, pad_width:-pad_width:window_width]
                velData[frame_index - 1, :, :, :] = averaged_arr
            else:
                velData[frame_index - 1, :, :, :] = flow
            
            prev_gray_processed = curr_gray_processed

        # Save metadata
        velData.attrs['window_height'] = window_height
        velData.attrs['window_width'] = window_width
        velData.attrs['mm_per_px'] = mm_per_px
        velData.attrs['fps_capture'] = fps_capture
        velData.attrs['roi'] = roi
        
    inputVid.release()

#############################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='farneback', help="Optical flow method: 'raft' or 'farneback'")
    parser.add_argument('--model', default='c1', help="For RAFT: path to model. For Farneback: a case identifier string.")
    parser.add_argument('--path', help="Path to the input video file")
    parser.add_argument('--win_h', default=4, help="Averaging window height")
    parser.add_argument('--win_w', default=4, help="Averaging window width")
    parser.add_argument('--imgScale', default=0.001, help="Image Scale (mm/px)")
    parser.add_argument('--fpsCam', default=130000, help="FPS of the camera used for capturing")
    parser.add_argument('--roi', type=int, nargs=4, default=[0, -1, 0, -1], help="Region of interest: y_start y_end x_start x_end. Use -1 for 'to the end'.")
    
    # --- RAFT specific arguments ---
    parser.add_argument('--small', action='store_true', help='Use small RAFT model')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision for RAFT')
    parser.add_argument('--alternate_corr', action='store_true', help='Use efficient correlation implementation for RAFT')

    # --- CLAHE arguments ---
    parser.add_argument('--use_clahe', action='store_true', help='Apply CLAHE as a preprocessing step')
    parser.add_argument('--clahe_clip_limit', type=float, default=2.0, help='Clip limit for CLAHE')
    parser.add_argument('--clahe_tile_size', type=int, default=8, help='Tile grid size for CLAHE (e.g., 8 for an 8x8 grid)')

    args = parser.parse_args()

    # Check if a video path is provided
    if not args.path:
        parser.error("--path to the video file is required.")
        sys.exit(1)

    inputVid = cv2.VideoCapture(args.path)
    if not inputVid.isOpened():
        print(f"Error: Could not open video file at {args.path}")
        sys.exit(1)
        
    if args.method == 'raft':
        # Check if RAFT dependencies are available
        try:
            from raft import RAFT
            from utils.utils import InputPadder
        except ImportError:
            print("Error: 'raft' and 'utils' modules are required for the RAFT method.")
            print("Please ensure the RAFT source code is in your Python path.")
            sys.exit(1)
        raftOpticalFlow(args, inputVid)

    elif args.method == 'farneback':
        farnebackMethod(args, inputVid)
    
    else:
        print(f"Error: Unknown method '{args.method}'. Please choose 'raft' or 'farneback'.")

#############################################################################################
