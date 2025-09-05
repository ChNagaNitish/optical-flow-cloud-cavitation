import matplotlib.pyplot as plt
import io
import argparse
import cv2
import numpy as np
import h5py

def outputVideoWriter(inputVid, videoName, new_fps):
    """
    Creates a video writer object to save the output video.

    Args:
        inputVid (cv2.VideoCapture): Input video capture object to get properties.
        videoName (str): Path and name for the output video file.
        new_fps (int): Frames per second for the output video.

    Returns:
        cv2.VideoWriter: The video writer object.
    """
    frame_width = int(inputVid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(inputVid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Using 'mp4v' for better compatibility with .mp4 extension
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(videoName, fourcc, new_fps, (frame_width, frame_height))

def quiverVideo(inputVid, flowPath, outputVid):
    """
    Generates a video of quiver plots overlaid on the input video frames.

    Args:
        inputVid (cv2.VideoCapture): The input video.
        flowPath (str): The path to the HDF5 file containing velocity data.
        outputVid (cv2.VideoWriter): The video writer for the output.
    """
    ret, prev_frame = inputVid.read()
    if not ret:
        print("Error: Could not read the first frame of the video.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = int(inputVid.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = prev_frame.shape[:2]

    # --- MODIFIED SECTION: Load data from H5 file ---
    with h5py.File(flowPath, 'r') as hf:
        # Check if the 'velocity' dataset exists
        if 'velocity' not in hf:
            print(f"Error: 'velocity' dataset not found in {flowPath}")
            return
            
        velData = hf['velocity'][:]
        
        # Read window dimensions from dataset attributes
        try:
            window_height = hf['velocity'].attrs['window_height']
            window_width = hf['velocity'].attrs['window_width']
            print(f"Read window size from attributes: {window_width}x{window_height}")
        except KeyError:
            print("Error: 'window_height' or 'window_width' attributes not found in the H5 file.")
            print("Please ensure these attributes are set on the 'velocity' dataset.")
            return
    # --- END OF MODIFIED SECTION ---

    for frame_index in range(1, frame_count):
        ret, curr_frame = inputVid.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_index}. Stopping.")
            break
            
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        y, x = np.mgrid[0:h:window_height, 0:w:window_width]

        # Ensure velData has enough frames
        if frame_index - 1 >= velData.shape[0]:
            print(f"Warning: Velocity data has fewer frames than the video. Stopping at frame {frame_index-1}.")
            break

        u = velData[frame_index-1, :, :, 0]
        v = -velData[frame_index-1, :, :, 1] # Invert v for correct plotting direction

        # Create the plot for the current frame
        plt.figure(figsize=(w / 100, h / 100), dpi=100)
        plt.imshow(prev_gray, cmap='gray')
        
        # Draw quiver plot, subsampling for clarity (e.g., every 4th arrow)
        plt.quiver(x[::4, ::4], y[::4, ::4], u[::4, ::4], v[::4, ::4], color='red', scale=90)
        
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Save plot to a buffer and read it into an OpenCV image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        quiver_image = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
        quiver_image = cv2.resize(quiver_image, (w, h))
        plt.close()
        
        prev_gray = curr_gray
        
        outputVid.write(quiver_image)

    print("Video processing complete.")
    inputVid.release()
    outputVid.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a quiver plot video from cloud cavitation velocity data.")
    parser.add_argument('--path', required=True, help="Path to the input video file (e.g., video.mp4)")
    parser.add_argument('--velocity', required=True, help='Path to the input velocity data file (e.g., velocity.h5)')
    parser.add_argument('--fps', default=10, type=int, help='Frames per second of the output quiver video')
    args = parser.parse_args()
    
    inputVid = cv2.VideoCapture(args.path)
    if not inputVid.isOpened():
        print(f"Error: Could not open input video file: {args.path}")
    else:
        # Determine output path based on the velocity file name
        output_name = args.velocity.rsplit('.', 1)[0]
        outputVideoPath = f'{output_name}_quiver.mp4'
        
        outputVid = outputVideoWriter(inputVid, outputVideoPath, args.fps)
        
        print(f"Starting quiver video generation...")
        print(f"Input video: {args.path}")
        print(f"Velocity data: {args.velocity}")
        print(f"Output video: {outputVideoPath}")
        
        quiverVideo(inputVid, args.velocity, outputVid)
