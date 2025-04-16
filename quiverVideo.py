import matplotlib.pyplot as plt
import io
import argparse
import cv2
import numpy as np

def outputVideoWriter(inputVid,videoName,new_fps):
    original_fps = inputVid.get(cv2.CAP_PROP_FPS)
    frame_width = int(inputVid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(inputVid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    return cv2.VideoWriter(videoName, fourcc, new_fps, (frame_width, frame_height))

def quiverVideo(inputVid,flowPath,outputVid):
    ret, prev_frame = inputVid.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = int(inputVid.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = prev_frame.shape[:2]
    window_size = [8,8]
    window_height, window_width = window_size
    velData = np.load(flowPath)
    for frame_index in range(1,frame_count):
        ret, curr_frame = inputVid.read()
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        y, x = np.mgrid[0:h:window_height, 0:w:window_width] 
        plt.figure(figsize=(w / 100, h / 100), dpi=100)
        plt.imshow(prev_gray, cmap='gray')
        plt.quiver(x, y, velData[frame_index-1,:,:,0], -velData[frame_index-1,:,:,1], color='red')
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        quiver_image = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
        quiver_image = cv2.resize(quiver_image, (w, h))
        plt.close()
        prev_gray = curr_gray
        outputVid.write(quiver_image)
    inputVid.release()
    outputVid.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help="input video")
    parser.add_argument('--velocity', help='input velocity field')
    parser.add_argument('--fps', default=1, help='fps of quiver video')
    args = parser.parse_args()
    inputVid = cv2.VideoCapture(args.path)
    outputVideoPath = args.velocity[:-4]+'.avi'
    outputVid = outputVideoWriter(inputVid,outputVideoPath,args.fps)
    quiverVideo(inputVid,args.velocity,outputVid)
