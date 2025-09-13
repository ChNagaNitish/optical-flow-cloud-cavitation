# optical-flow-cloud-cavitation

## Purpose of Different Scripts
**gpuJobScript.sh** -> To submit job on ARC(Falcon cluster) using GPUs <br/>
**serialJob.sh** -> To submit postprocessing scripts on ARC(Owl cluster) <br/>
**postProcessVelData.py** -> To generate various plots and animations with the generated velocity data from Optical Flow methods <br/>
**quiverVideo.py** -> To overlay velocity vectors on top of the input video <br/>
**speedUp.py** -> To speed up videos for better visualization <br/>
**tracking.py** -> The main script to run Optical flow methods(RAFT and Farneback) on input Videos <br/>

## tracking.py
**Examples** <br/>
-> python3 tracking.py --method farneback --model c1 --path 32_50f.avi --win_h 4 --win_w 4 --roi 145 333 4 -1 --imgScale 0.0477273 --fpsCam 13000 <br/>
-> python3 tracking.py --method=raft --model=models/raft-cloudcav.pth --path 32_50f.avi --win_h 4 --win_w 4 --roi  0 360 60 1276 --imgScale 0.02175955 --fpsCam 130000 --use_clahe <br/> <br/>
**Arguments** <br/>
--method -> to use raft or farneback optical flow <br/>
--model -> for Farneback it represents case number for different numerical parameters, for RAFT it is the model parameters that should be used. For RAFT models, you can look inside the models folder and use them accordingly <br/>
--path -> path for the input video <br/>
--win_h -> window height for averaging velocity <br/>
--win_w -> window width for avergaing velocity <br/>
--roi -> region of interest for the input video. Let's say the video frame size is 384*1280 and we are only interested in some part of it, we provide the starting pixel position in height(top), ending position in height(bottom), starting position in width(left), ending in width(right). If you do not provide, it will process the whole video frame <br/>
--imgScale -> the calibration value obtained from experiment. It is in mm/px. Default is 0.001. <br/>
--fpsCam -> the framerate at which video is captured <br/>
--use_clahe -> A preprocessing step to make the contrast uniform. No inputs needed for this arguement. Just using it will activate preprocessing step. Recommended to use it. <br/>
**Note:** The velocity is saved as .h5 format in px/frame units and the imgScale, fpsCam, win_h, win_w are saved as attributes for postprocessing later <br/>

## quiverVideo.py
**Example** <br/>
-> python3 quiverVideo.py --path 32_50f.avi --velocity 32_50f_raft-cloudcav.h5 --fps 10 <br/>

## postProcessVelData.py
**Examples** <br/>
python3 postProcessVelData.py --method vLinesAvg --path 32_50f_raft-cloudcav.h5 <br/>
python3 postProcessVelData.py --method hline --path 32_50f_raft-cloudcav.h5 <br/>
python3 postProcessVelData.py --method points --path 32_50f_raft-cloudcav.h5 <br/>
**Note:** so far only the above mentioned 3 different methods are working. points will generate a time series at 2,10,20mm x location from throat. hline will generate a video showing velocity evolution at a line close to bottom wall. vLinesAvg will plot time averaged velocity along the verticle lines at x points 1.5,3,5,10,20mm <br/>
