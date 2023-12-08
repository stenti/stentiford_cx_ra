import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def multivariate_gaussian(pos, mu, Sigma):
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def gen_rf (img_size,center_point,sigma = 50.):
    X = np.linspace(0, img_size[1], img_size[1])
    Y = np.linspace(0, img_size[0], img_size[0])
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mu = np.array([center_point[0], center_point[1]])
    Sigma = np.array([[ sigma , -0.5], [-0.5,  sigma]])

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    return multivariate_gaussian(pos, mu, Sigma)

def gauss_dis (target, pos, sig = 40):
    d = np.sqrt((target[0] - pos[0])**2 + (target[1] - pos[1])**2)
    return np.exp(-np.power(d, 2.) / (2 * np.power(sig, 2.)))

#-----------------------------------------------------
# Hexagonal grid
#-----------------------------------------------------

def getHexGridCenters(img_size, rad):
    d = rad*2
    h = img_size[0]
    w = img_size[1]

    pad = int(h//5)
    scale = (3*pad)/h

    vspace = np.sqrt(3) * rad
    hspace = (3/4)*d

    nh = int(w//hspace)
    nv = int(h//vspace) 

    grid = {}
    for i,n in enumerate(np.floor(np.arange(0,nv,.5)).astype(int)):
        for j,m in enumerate(np.floor(np.arange(0,nh,.5)).astype(int)):
            if i%2 and j%2:
                x = ((3*n)+(3/2)) *rad
                y = ((m+.5)*np.sqrt(3)) *rad
                if x<=h and y<=w:
                    grid[i,m] = ((x*scale)+pad,y)
            elif not i%2 and not j%2:
                x = (3*n) *rad
                y = (m*np.sqrt(3)) *rad
                if x<=h and y<=w:
                    grid[i,m] = ((x*scale)+pad,y)
    return grid

def plotgridcenters (img_size, grid):
    for k in grid.keys():
        plt.scatter(grid[k][1],grid[k][0])
    plt.ylim([0,img_size[0]])
    plt.xlim([0,img_size[1]])
    plt.show()

def getWeights(grid, plot = False):
    # option for random choice between 0 and 1 (more lower values)
    rands = np.exp(np.arange(0,5,.1))
    rands = rands/np.max(rands)

    W = np.zeros((len(grid.keys()),len(grid.keys())))
    for i,k in enumerate(grid.keys()):
        for j,l in enumerate(grid.keys()):
            if i != j:
                d = gauss_dis((grid[k][1],grid[k][0]),(grid[l][1],grid[l][0]))
                W[i,j] =  d * np.random.choice(rands) #np.random.rand() 
    if plot:
        plt.imshow(W, origin='lower', cmap ='jet')
        plt.colorbar()
        plt.show()
    return W

def getMatRFs(img_size, grid, sigma):
    RFs = np.zeros((img_size[0],img_size[1],len(grid.keys())))
    for i,k in enumerate(grid.keys()):
        RFs[:,:,i] = gen_rf (img_size,(grid[k][1],grid[k][0]),sigma)
    return RFs

def getGridRFs(img_size, grid, sigma):
    gridRFs = {}
    for i,k in enumerate(grid.keys()):
        gridRFs[k] = gen_rf (img_size,(grid[k][1],grid[k][0]),sigma)
    return gridRFs

def plotGridRF(pos, gridRFs):
    plt.imshow(gridRFs[pos], origin='lower', cmap ='jet')
    plt.show()

def plotMatRF(n, RFs):
    plt.imshow(RFs[:,:,n], origin='lower', cmap ='jet')
    plt.show()

def plotAllMatRFs(RFs):
    n = RFs.shape[2]
    fig, axes = plt.subplots(n//2,2)
    j = 0
    for i in np.arange(n):
        if i==n//2: j = 1
        axes[i%(n//2),j].imshow(RFs[:,:,i], origin='lower', cmap ='jet')
    plt.show()

def plotAllRFs(gridRFs):
    n = len(gridRFs.keys())
    fig, axes = plt.subplots(n//2,2, figsize=(5, 15))
    j = 0
    for i,k in enumerate(gridRFs.keys()):
        mx = np.max(gridRFs[k])
        mn = np.min(gridRFs[k])
        # print(mx,mn)
        if i==n//2: j = 1
        axes[i%(n//2),j].imshow(gridRFs[k], aspect='auto', origin='lower', cmap ='jet', vmin = -mx, vmax = mx)
        axes[i%(n//2),j].set_xticks([])
        axes[i%(n//2),j].set_yticks([])
    plt.show()

def plotStackedRFs(gridRFs):
    n = len(gridRFs.keys())
    fig, axes = plt.subplots(1, figsize=(10,3))
    j = 0
    combRF = np.zeros(gridRFs[(0,0)].shape)
    for i,k in enumerate(gridRFs.keys()):
        combRF = combRF + gridRFs[k]
        mn = np.min(gridRFs[k])
    mx = np.max(combRF)
    mn = np.min(combRF)
    axes.imshow(combRF, aspect='auto', origin='lower', cmap ='Reds', vmin = 0, vmax = mx)

    plt.show()

def norm (x):
    return ((x - np.min(x)) / (np.max(x) - np.min(x)) *2) -1


def getCombinedRFs(img_size,gridRFs, W):
    combRFs = {}
    for i,k in enumerate(gridRFs.keys()):
        exc = norm(gridRFs[k])
        Z = np.zeros(img_size)
        for j,l in enumerate(gridRFs.keys()):
            if i != j:
                Z = Z + (gridRFs[l] * W[i,j])

        rf = (exc-norm(Z)) 
        combRFs[k] = (rf)
    return combRFs

def getActivations (frames, shift = 0): 
    n_frames = frames['n_frames']
    img_size = frames[0].shape

    R4grid = getHexGridCenters(img_size, 25) # 25 225 for imagesize (90,360) # 18 200 for imagesize (60,360) 
    R2grid = getHexGridCenters(img_size, 20) # 20 225 for imagesize (90,360) # 15 150 for imagesize (60,360) 
    R2RF = (getGridRFs(img_size,R2grid,225)) #250
    R4RF = (getGridRFs(img_size,R4grid,225))
    n_r2 = len(R2RF.keys())
    n_r4 = len(R4RF.keys())

    r4_A = np.zeros((n_r4,n_frames))
    r2_A = np.zeros((n_r2,n_frames))

    for f in range(n_frames):
        frame = frames[f]
        frame = np.hstack((frame[:,shift:] ,frame[:,:shift]) )
        for i,k in enumerate(R4RF.keys()):
            r4_A[i,f] = np.sum(np.multiply(frame,R4RF[k]))
        r4_A[:,f] = norm(r4_A[:,f])
        for i,k in enumerate(R2RF.keys()):
            r2_A[i,f] = np.sum(np.multiply(frame,R2RF[k]))
        r2_A[:,f] = norm(r2_A[:,f])

    r4_A[r4_A<0] = 0
    r2_A[r2_A<0] = 0

    return r4_A, r2_A





# import random

# #----------------------------------------------------------
# # Generate some examples
# #----------------------------------------------------------
# # R4
# type = 'R4'
# rad = 25 # 25 for imagesize (90,360) # 18 for imagesize (60,360) 
# sigma = 30#50 # 250 for imagesize (90,360) # 200 for imagesize (60,360)

# # # R2
# # type = 'R2'
# # rad = 20 #20 for imagesize (90,360) # 15 for imagesize (60,360) 
# # sigma = 225 #225 for imagesize (90,360) # 150 for imagesize (60,360)

# img_size = (95, 360)

# grid = getHexGridCenters(img_size, rad)
# # plotgridcenters(img_size,grid)
# W = getWeights(grid)#, plot=True) 
# num = random.randint(10, 30)
# # print(num)
# # np.save(f'data/weights/{type}_weights',W, allow_pickle=True)

# RFs = getGridRFs(img_size,grid,sigma)
# plotGridRF((0,0), RFs)

# combRFs = getCombinedRFs(img_size,RFs, W)
# plotGridRF((2,5), combRFs)

# plotAllRFs(RFs)
# plotStackedRFs(RFs)
# plotAllRFs(combRFs)

# RFs = getMatRFs(img_size,grid,sigma)
# plotMatRF(25, RFs)
# np.save(f'data/{type}_RFs',RFs, allow_pickle=True)
