import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
from PIL import Image

allfiles =  ['1_circle_static','2_circle_static','3_circle_static','4_circle_static','5_circle_static',
             '6_circle_static','7_circle_static','8_circle_static','9_circle_static','10_circle_static',
             '1_circle_super','2_circle_super','3_circle_super','4_circle_super','5_circle_super',
             '6_circle_super','7_circle_super','8_circle_super','9_circle_super','10_circle_super']

conv_size = {'static':'0','small':'50','med':'100','large':'150','super':'200'}
dwnsmpl = {'0': 1,'50': 1,'100': 2,'150': 3,'200':4}


def remove_padding(img):
    h,w = img.shape
    keep = np.arange(200,500)
    return img[keep,:]

def shrink (img):
    img = Image.fromarray(img)
    w,h = img.size
    img = img.resize((360, 95))
    return np.array(img)

# Load data (deserialize)
with open('spidercam/circling.pkl', 'rb') as handle:
    circling = pickle.load(handle)

fps = 60

for fname in allfiles:
    sz = conv_size[fname.split('_')[-1]]

    path = f'data/circling/{fname}.mp4'
    cap = cv2.VideoCapture(path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = dict.fromkeys(range(n_frames))
    frames['n_frames'] = n_frames

    # Loop until the end of the video
    for i in range(0, int(n_frames)):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # desaturate
        img = Image.fromarray(frame).convert('L')

        # remove black border
        img = remove_padding(np.array(img))

        # hitograme equalization
        img = cv2.equalizeHist(np.array(img)) # histogram equalization

        # store processed frame 
        if i == 0:
            h,w = img.shape
            frames['shape'] = img.shape

        frames[i] = img[0:h,:] 
    
    # release the video capture object
    cap.release()

    n_frames = frames['n_frames']
    frame_tms = 1000*np.arange(n_frames)/fps

    if sz != '0':
        deg = circling[sz]['deg']
        traj_tms = 1000*np.arange(len(deg))/100
    else:
        deg = (np.arange(1200) * ((360*3)/n_frames))%360
        traj_tms = 1000*np.arange(len(deg))/60

    out = cv2.VideoWriter(f'data/circling/{fname}_processed.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 60, (360,95), False)

    newframes = {}
    newframes['n_frames'] = 0
    newframes['shape'] = (360, 95)
    
    count = 0
    for f in range(n_frames):
        if f%dwnsmpl[sz] == 0:
            frame = frames[f]
            idx = np.argmin(abs(traj_tms - frame_tms[f]))
            if idx+1 != len(traj_tms):
                a = (deg[idx]*3.5555555).astype(int)
                newframes[count] = shrink(np.hstack((frame[:,a:] ,frame[:,:a])))
                out.write(newframes[count])
                count = count + 1
                newframes['sim_len'] = traj_tms[idx]//dwnsmpl[sz]
                newframes['n_frames'] = newframes['n_frames']+1
    
    out.release()

    path = path.replace('.mp4','.pkl')
    with open(path, 'wb') as fp:
        pickle.dump(newframes, fp)