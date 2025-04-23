# optical-flow-cloud-cavitation
Example usages of scripts <br/>
python3 tracking.py --method=farneback --model=default --path=48.avi <br/>
python3 tracking.py --method=raft --model=models/raft-sintel.pth --path=48.avi <br/>
python3 postProcessVelData.py --method=lines --path=48_farneback_default.npy <br/>
python3 postProcessVelData.py --method=compareAlgo --path=48_farneback_default.npy,48_raft-sintel.npy --legend=farneback-default,raft-sintel --output=48_compare_farneback_raft.avi --fps=20 --algo=none
