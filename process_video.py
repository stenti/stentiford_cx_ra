import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path
import cv2
from PIL import Image
import math
import seaborn as sns

def remove_padding(img):
    h,w = img.shape
    keep = np.arange(200,500)
    return img[keep,:]

def shrink (img):
    w,h = img.size
    img = img.resize((360, 95))#int(h * (360/w))))
    return img

def process (path):
    # Creating a VideoCapture object to read the video
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

        # resize image
        img = shrink(Image.fromarray(img))

        # hitograme equalization
        img = cv2.equalizeHist(np.array(img)) # histogram equalization

        # posterize
        # img = np.array(ImageOps.posterize(Image.fromarray(img), 2))

        # store processed frame 
        if i == 0:
            h,w = img.shape
            frames['shape'] = img.shape
            print('saving shape')
        #     frames = np.ndarray((h,w,n_frames))
        # frames[:,:,i] = img[0:h,:] # need to account for some variation in how the border was removed

        frames[i] = img[0:h,:] # need to account for some variation in how the border was removed
    
    # release the video capture object
    cap.release()

    path = path.replace('.mp4','.pkl')
    with open(path, 'wb') as fp:
        pickle.dump(frames, fp)

    return frames

def mean2(x):
    y = np.sum(x) / np.size(x)
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r

def activations_autocorrelation (r2,r4):
    activ = np.vstack([r2,r4])
    h,w = activ.shape
    print(f'Calculating correlations between R cell activations')

    corMat = np.zeros((w,w))
    for first in range(w):
        for second in range(w):
            im1 = activ[:,first]
            im2 = activ[:,second]
            im1 = cv2.normalize(im1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            im2 = cv2.normalize(im2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            corMat[first,second] = corr2(im1,im2)
    return corMat

def video_autocorrelation (frames):
    n = frames['n_frames']
    print(f'Calculating correlations between frames')

    corMat = np.zeros((n,n))
    for first in range(n):
        for second in range(n):
            im1 = frames[first]
            im2 = frames[second]
            im1 = cv2.normalize(im1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            im2 = cv2.normalize(im2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            corMat[first,second] = corr2(im1,im2)
    return corMat

# PLOT ALL PANORAMAS
def plot_panoramas (f_names,f_num, var = {}, d = {}):
    col = 3
    colours = ['green','blue','red']
    colours = sns.color_palette("tab10", 3)
    label = ['Low','High','Fail']
    fig, axes = plt.subplots(len(f_names)//col,col, figsize=(10,15))
    sns.set_style('white')
    for count,fname in enumerate(f_names):
        n = fname.split('/')[-1]
        n = n.replace('_static','')
        n = n.replace('_super','')
        if os.path.isfile(f'{fname}.pkl'):
            with open(f'{fname}.pkl', 'rb') as fp:
                frames = pickle.load(fp)
            
            axes[count//col,count%col].imshow(frames[f_num[count]], aspect='auto', cmap = 'gray')
        if n in var.keys():
            c = colours[var[n]]
            # axes[count//col,count%col].spines['bottom'].set_color(c)
            # axes[count//col,count%col].spines['top'].set_color(c) 
            # axes[count//col,count%col].spines['right'].set_color(c)
            axes[count//col,count%col].spines['left'].set_color(c)
            axes[count//col,count%col].spines['left'].set_linewidth(3)
            axes[count//col,count%col].set_ylabel(f'{label[var[n]]} {d[n]}m' )
        else:
            axes[count//col,count%col].axis('off')
        axes[count//col,count%col].set_xticks([])
        axes[count//col,count%col].set_yticks([])
    plt.savefig(f'results/allPanoramas.png', bbox_inches='tight')
    plt.savefig(f'results/allPanoramas.svg', bbox_inches='tight')
    plt.show()


all = ['data/rotation/12_3rev_static','data/rotation/19_3rev_static', 'data/rotation/17_3rev_static',
            'data/rotation/25_3rev_static', 'data/rotation/24_3rev_static', 'data/rotation/23_3rev_static',

           'data/rotation/1_3rev_static','data/rotation/2_3rev_static','data/rotation/4_3rev_static',
           'data/rotation/5_3rev_static','data/rotation/6_3rev_static','data/rotation/7_3rev_static',
           'data/rotation/9_3rev_static','data/rotation/10_3rev_static','data/rotation/21_3rev_static',
           'data/rotation/11_3rev_static','data/rotation/13_3rev_static','data/rotation/14_3rev_static',
           'data/rotation/15_3rev_static','data/rotation/20_3rev_static','data/rotation/18_3rev_static',
           'data/rotation/22_3rev_static','data/rotation/16_3rev_static','',
           '','','',
           'data/circling/1_circle_super','','',
           'data/circling/2_circle_super','','',
           'data/circling/3_circle_super','','',
           'data/circling/4_circle_super','','',
           'data/circling/5_circle_super','data/circling/6_circle_super','data/circling/7_circle_super',
           'data/circling/8_circle_super','data/circling/9_circle_super','data/circling/10_circle_super'
           ]
f_num = [0,60,0,140,0,170,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

var = { '1_3rev' : 0,
        '2_3rev' : 2,
        '4_3rev' : 1,
        '5_3rev' : 0,
        '6_3rev' : 0,
        '7_3rev' : 1,
        '9_3rev' : 1,
        '10_3rev' : 0,
        '11_3rev' : 0,
        '12_3rev' : 1,
        '13_3rev' : 1,
        '14_3rev' : 1,
        '15_3rev' : 0,
        '16_3rev' : 0,
        '17_3rev' : 1,
        '18_3rev' : 0,
        '19_3rev' : 0,
        '20_3rev' : 0,
        '21_3rev' : 0,
        '22_3rev' : 1,
        '23_3rev' : 1,
        '24_3rev' : 0,
        '25_3rev' : 2,
        '1_circle' : 2,
        '2_circle' : 1,
        '3_circle' : 0,
        '4_circle' : 2,
        '5_circle' : 0,
        '6_circle' : 2,
        '7_circle' : 2,
        '8_circle' : 0,
        '9_circle' : 1,
        '10_circle' : 0}

d = { '1_3rev' : 5,
        '2_3rev' : 2,
        '4_3rev' : 2,
        '5_3rev' : 5,
        '6_3rev' : 5,
        '7_3rev' : 10,
        '9_3rev' : 6,
        '10_3rev' : 12,
        '11_3rev' : 10,
        '12_3rev' : 30,
        '13_3rev' : 3,
        '14_3rev' : 4,
        '15_3rev' : 1,
        '16_3rev' : 8,
        '17_3rev' : 20,
        '18_3rev' : 6,
        '19_3rev' : 50,
        '20_3rev' : 6,
        '21_3rev' : 3,
        '22_3rev' : 10,
        '23_3rev' : 20,
        '24_3rev' : 50,
        '25_3rev' : 30,
        '1_circle' : 5,
        '2_circle' : 50,
        '3_circle' : 0.1,
        '4_circle' : 0.5,
        '5_circle' : 0.2,
        '6_circle' : 0.2,
        '7_circle' : 0.2,
        '8_circle' : 3,
        '9_circle' : 5,
        '10_circle' : 3}

# plot_panoramas(all,f_num, var = var, d = d)

