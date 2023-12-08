import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import pickle

path = 'data/simple_stim'

fps = 60
rpm = 10


t_single_frame = 1000/fps # ms
t_1rev = 60000/rpm # ms
n_1rev = int(np.ceil(t_1rev/t_single_frame))


# GENERATE IMAGES
width = 360
height = 95

one_bar = np.zeros((height,width))
one_bar[:,:15] = 255

two_bar = np.zeros((height,width))
two_bar[:,:15] = 255
two_bar[:,180:195] = 255

one_bar_shifted = np.zeros((height,width))
one_bar_shifted[:,180:195] = 255


# GENERATE 1 BAR ROTATION
n_rev = 3
n_frames = n_1rev * n_rev

frames = dict.fromkeys(range(n_frames))
frames['n_frames'] = n_frames
frames['shape'] = one_bar.shape
for j in range(n_rev):
    for i in range(width-1,-1 ,-1):
        window = np.hstack((one_bar[:,i:] ,one_bar[:,:i]) )
        frames[i+(j*n_1rev)] = window

# for i in range(0,n_frames,50):
#     plt.imshow(frames[i])
#     plt.show()

with open(f'{path}/bar_stim_static.pkl', 'wb') as fp:
    pickle.dump(frames, fp)



# GENERATE 2 BAR ROTATION EXPERIEMNT
# rotate 3 times each
n_rev = 3
n_frames = n_1rev * n_rev * 3

for p,bar_pos in enumerate([one_bar,one_bar_shifted]):
    frames = dict.fromkeys(range(n_frames))
    frames['n_frames'] = n_frames
    frames['shape'] = one_bar.shape
    for j in range(n_rev):
        for i in range(width-1,-1 ,-1):
            window = np.hstack((one_bar[:,i:] ,one_bar[:,:i]) )
            frames[i+(j*n_1rev)] = window

    for j in range(n_rev,n_rev*2):
        for i in range(width-1,-1 ,-1):
            window = np.hstack((two_bar[:,i:] ,two_bar[:,:i]) )
            frames[i+(j*n_1rev)] = window

    for j in range(n_rev*2,n_rev*3):
        for i in range(width-1,-1 ,-1):
            window = np.hstack((bar_pos[:,i:] ,bar_pos[:,:i]) )
            frames[i+(j*n_1rev)] = window

    # for i in range(0,n_frames,90):
    #     plt.imshow(frames[i])
    #     plt.show()

    with open(f'{path}/two_bar_{p}_static.pkl', 'wb') as fp:
        pickle.dump(frames, fp)




# LOAD DEWAR IMAGES

for fname in ['triangles','triangles_com']:
    tri = np.squeeze(np.array(Image.open(f'{path}/{fname}.png'))[:,:,0])
    tri = tri/np.max(tri)
    tri[tri != 0] = -1
    tri = tri+1
    plt.imshow(tri)
    plt.show()

    # GENERATE 1  ROTATION
    n_rev = 3
    n_frames = n_1rev * n_rev

    frames = dict.fromkeys(range(n_frames))
    frames['n_frames'] = n_frames
    frames['shape'] = tri.shape
    for j in range(n_rev):
        for i in range(width-1,-1 ,-1):
            window = np.hstack((tri[:,i:] ,tri[:,:i]) )
            frames[i+(j*n_1rev)] = window

    # for i in range(0,n_frames,50):
    #     plt.imshow(frames[i])
    #     plt.show()

    with open(f'{path}/{fname}_static.pkl', 'wb') as fp:
        pickle.dump(frames, fp)