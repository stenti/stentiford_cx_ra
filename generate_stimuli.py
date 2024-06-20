import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import pickle
import matplotlib.animation as animation

def get_params():
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

    inv_bar = np.ones((height,width))*255
    inv_bar[:,:15] = 0

    two_bar = np.zeros((height,width))
    two_bar[:,:15] = 255
    two_bar[:,180:195] = 255

    one_bar_shifted = np.zeros((height,width))
    one_bar_shifted[:,180:195] = 255

    params = {'path':path,
              'fps':fps,
              'rpm':rpm,
              't_1rev':t_1rev,
              'n_1rev':n_1rev,
              'width':width,
              'height':height,
              'one_bar':one_bar,
              'inv_bar':inv_bar,
              'two_bar':two_bar,
              'one_bar_shifted':one_bar_shifted
              }
    
    return params

def generate_1bar():
    params = get_params()
    path = params['path']
    # GENERATE 1 BAR ROTATION
    n_rev = 3
    n_frames = params['n_1rev'] * n_rev

    Frame = []
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    frames = dict.fromkeys(range(n_frames))
    frames['n_frames'] = n_frames
    frames['shape'] = params['one_bar'].shape
    for j in range(n_rev):
        for i in range(params['width']-1,-1 ,-1):
            window = np.hstack((params['one_bar'][:,i:] ,params['one_bar'][:,:i]) )
            frames[i+(j*params['n_1rev'])] = window

            img = ax.imshow(frames[i],animated=True)
            Frame.append([img])

    ani = animation.ArtistAnimation(fig, Frame, interval=16, blit=True,repeat_delay=1000)
    ani.save(f'{path}/bar_stim_static.gif')

    with open(f'{path}/bar_stim_static.pkl', 'wb') as fp:
        pickle.dump(frames, fp)

    # GENERATE 1 BAR ROTATION INVERSE
    n_rev = 3
    n_frames = params['n_1rev'] * n_rev

    frames = dict.fromkeys(range(n_frames))
    frames['n_frames'] = n_frames
    frames['shape'] = params['inv_bar'].shape
    for j in range(n_rev):
        for i in range(params['width']-1,-1 ,-1):
            window = np.hstack((params['inv_bar'][:,i:] ,params['inv_bar'][:,:i]) )
            frames[i+(j*params['n_1rev'])] = window

    with open(f'{path}/bar_inv_stim_static.pkl', 'wb') as fp:
        pickle.dump(frames, fp)



def generate_2bar():
    params = get_params()
    # GENERATE 2 BAR ROTATION EXPERIEMNT
    # rotate 3 times each
    n_rev = 3
    n_frames = params['n_1rev'] * n_rev * 3

    for p,bar_pos in enumerate([params['one_bar'],params['one_bar_shifted']]):
        frames = dict.fromkeys(range(n_frames))
        frames['n_frames'] = n_frames
        frames['shape'] = params['one_bar'].shape
        for j in range(n_rev):
            for i in range(params['width']-1,-1 ,-1):
                window = np.hstack((params['one_bar'][:,i:] ,params['one_bar'][:,:i]) )
                frames[i+(j*params['n_1rev'])] = window

        for j in range(n_rev,n_rev*2):
            for i in range(params['width']-1,-1 ,-1):
                window = np.hstack((params['two_bar'][:,i:] ,params['two_bar'][:,:i]) )
                frames[i+(j*params['n_1rev'])] = window

        for j in range(n_rev*2,n_rev*3):
            for i in range(params['width']-1,-1 ,-1):
                window = np.hstack((bar_pos[:,i:] ,bar_pos[:,:i]) )
                frames[i+(j*params['n_1rev'])] = window

        with open(f'{params['path']}/two_bar_{p}_static.pkl', 'wb') as fp:
            pickle.dump(frames, fp)


def generate_fromAV(av,rotations = 3):
    params = get_params()

    scale = 10
    width = 360 *scale
    height = 95

    one_bar = np.zeros((height,width))
    one_bar[:,:15*scale] = 255

    sim_len = ((360*rotations)/av)*1000
    n_frames = int(sim_len/(1000/params['fps']))
    av = np.ones(n_frames)*av

    pixels = np.rint((av/params['fps'])*scale).astype(int)

    frames = dict.fromkeys(range(n_frames))
    frames['n_frames'] = n_frames

    for i in range(n_frames):
        one_bar = np.hstack((one_bar[:,pixels[i]:] ,one_bar[:,:pixels[i]]) )
        frames[i] = np.array(Image.fromarray(one_bar).resize((360, 95)))

    frames['shape'] = frames[i].shape

    return frames


def generate_1bar_manyrevs(n_rev):
    params = get_params()
    path = params['path']
    # GENERATE 1 BAR ROTATION
    n_frames = params['n_1rev'] * n_rev

    Frame = []
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    frames = dict.fromkeys(range(n_frames))
    frames['n_frames'] = n_frames
    frames['shape'] = params['one_bar'].shape
    for j in range(n_rev):
        for i in range(params['width']-1,-1 ,-1):
            window = np.hstack((params['one_bar'][:,i:] ,params['one_bar'][:,:i]) )
            frames[i+(j*params['n_1rev'])] = window

            img = ax.imshow(frames[i],animated=True)
            Frame.append([img])

    with open(f'{path}/bar_stim_static_{n_rev}revs.pkl', 'wb') as fp:
        pickle.dump(frames, fp)

    # GENERATE 1 BAR ROTATION INVERSE
    n_rev = 3
    n_frames = params['n_1rev'] * n_rev

    frames = dict.fromkeys(range(n_frames))
    frames['n_frames'] = n_frames
    frames['shape'] = params['inv_bar'].shape
    for j in range(n_rev):
        for i in range(params['width']-1,-1 ,-1):
            window = np.hstack((params['inv_bar'][:,i:] ,params['inv_bar'][:,:i]) )
            frames[i+(j*params['n_1rev'])] = window

    with open(f'{path}/bar_inv_stim_static_{n_rev}revs.pkl', 'wb') as fp:
        pickle.dump(frames, fp)

# generate_1bar()
# generate_1bar_manyrevs(20)