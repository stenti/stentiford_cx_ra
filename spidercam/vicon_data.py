import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pickle

df = pd.read_csv('spidercam/vicon_spidercam.csv')
fps = 100
smooth = 400
p = 2

print(df.head()) 

bound = [2450,5550,5850,12050,12350,21400,21700,33600]

circling = {}
circling['50'] = {"X":df['TX'][bound[0]:bound[1]]-df['TX'][0], "Y": df['TY'][bound[0]:bound[1]]-df['TY'][0]}
circling['100'] = {"X":df['TX'][bound[2]:bound[3]]-df['TX'][0], "Y": df['TY'][bound[2]:bound[3]]-df['TY'][0]}
circling['150'] = {"X":df['TX'][bound[4]:bound[5]]-df['TX'][0], "Y": df['TY'][bound[4]:bound[5]]-df['TY'][0]}
circling['200'] = {"X":df['TX'][bound[6]:bound[7]]-df['TX'][0], "Y": df['TY'][bound[6]:bound[7]]-df['TY'][0]}

nframes = [bound[1]-bound[0],bound[3]-bound[2],bound[5]-bound[4],bound[7]-bound[6]]
s = [nframes[0]/fps,nframes[1]/fps,nframes[2]/fps,nframes[3]/fps]

for circle in ['50','100','150','200']:
    xy = np.vstack([circling[circle]['X'],circling[circle]['Y']])
    dxy = np.diff(xy)
    a = np.arctan2(dxy[0,:], dxy[1,:])
    circling[circle]['rad'] = a
    circling[circle]['deg'] = (180 * a / np.pi)%360

for circle in ['50','100','150','200']:
    yhat_ = signal.savgol_filter(circling[circle]['Y'],smooth, p)
    xhat_ = signal.savgol_filter(circling[circle]['X'], smooth, p)
    xy = np.vstack([xhat_,yhat_])
    dxy = np.diff(xy)
    a = np.arctan2(dxy[0,:], dxy[1,:])
    circling[circle]['rad'] = a
    circling[circle]['deg'] = (180 * a / np.pi)%360
    circling[circle]['rad_unwrap'] = np.unwrap(a)

# Store data (serialize)
with open('spidercam/circling.pkl', 'wb') as handle:
    pickle.dump(circling, handle, protocol=pickle.HIGHEST_PROTOCOL)