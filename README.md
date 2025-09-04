# optical-flow-cloud-cavitation

## Purpose of Different Scripts
**gpuJobScript.h** -> To submit job on ARC using GPUs <br/>
**postProcessVelData.py** -> To generate various plots and animations with the generated velocity data from Optical Flow methods <br/>
**quiverVideo.py** -> To overlay velocity vectors on top of the input video <br/>
**speedUp.py** -> To speed up videos for better visualization <br/>
**tracking.py** -> The main script to run Optical flow methods(RAFT and Farneback) on input Videos <br/>

## tracking.py
-> python3 tracking.py --method farneback --model c1 --path 32_50f.avi --win_h 4 --win_w 4 --roi 145 333 4 -1 --imgScale 0.0477273 --fpsCam 13000 <br/>
-> python3 tracking.py --method=raft --model=models/raft-sintel.pth --path 32_50f.avi --win_h 4 --win_w 4 --roi 145 333 4 -1 --imgScale 0.0477273 --fpsCam 13000 <br/> <br/>
**Arguments** <br/>
--method -> to use raft or farneback optical flow <br/>
--model -> for Farneback it represents case number for different numerical parameters, for RAFT it is the model parameters that should be used. For RAFT models, you can look inside the models folder and use them accordingly <br/>
--path -> path for the input video <br/>
--win_h -> window height for averaging velocity <br/>
--win_w -> window width for avergaing velocity <br/>
--roi -> region of interest for the input video. Let's say the video frame size is 384*1280 and we are only interested in some part of it, we provide the starting pixel position in height(top), ending position in height(bottom), starting position in width(left), ending in width(right). If you do not provide, it will process the whole video frame <br/>
--imgScale -> the calibration value obtained from experiment. It is in mm/px. Default is 0.001. <br/>
--fpsCam -> the framerate at which video is captured <br/>
**Note:** The velocity is saved as .h5 format in px/frame units and the imgScale, fpsCam, win_h, win_w are saved as attributes for postprocessing later

## quiverVideo.py
Coming soon. Needs to be updated for the new .h5 format

## postProcessVelData.py
Coming soon. Needs to be updated for the new .h5 format

python3 postProcessVelData.py --method=lines --path=48_farneback_default.npy <br/>
python3 postProcessVelData.py --method=compareAlgo --path=48_farneback_default.npy,48_raft-sintel.npy --legend=farneback-default,raft-sintel --output=48_compare_farneback_raft.avi --fps=20 --algo=none
