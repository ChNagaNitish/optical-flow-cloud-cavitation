import cv2
import argparse

def speed_up_video(input_video, output_video, speed_factor):
  """
  Speeds up a video by a given factor.

  Args:
    input_video: Path to the input video file.
    output_video: Path to the output video file.
    speed_factor: The factor by which to speed up the video (e.g., 2 for 2x speed).
  """

  cap = cv2.VideoCapture(input_video)
  if not cap.isOpened():
    print("Error opening video file")
    return

  original_fps = cap.get(cv2.CAP_PROP_FPS)
  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  # Calculate the new frame rate
  
  new_fps = float(speed_factor)

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_video, fourcc, new_fps, (frame_width, frame_height))

  while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
      break

    out.write(frame)

  cap.release()
  out.release()
  #cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help="input video file")
    parser.add_argument('--fps', help="fps for output video")
    args = parser.parse_args()
    speed_up_video(args.file, args.file[:-4]+"_"+args.fps+"fps.mp4", args.fps)