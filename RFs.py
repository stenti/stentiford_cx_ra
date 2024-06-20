import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def norm (x):
    return ((x - np.min(x)) / (np.max(x) - np.min(x)) *2) -1

def load_weights(grid, scale = 1):
    if grid == 'square':
        w_R2 = np.load(f'data/RFs/square_w_R2.npy') * scale
        w_R4 = np.load(f'data/RFs/square_w_R4.npy') * scale
    elif grid == 'hex':
        w_R2 = np.load(f'data/RFs/hex_w_R2.npy') * scale
        w_R4 = np.load(f'data/RFs/hex_w_R4.npy') * scale
    elif grid == 'random':
        w_R2 = np.load(f'data/RFs/rand_w_R2.npy') * scale
        w_R4 = np.load(f'data/RFs/rand_w_R4.npy') * scale
    return w_R2,w_R4

def get_activations (frames, grid = 'square', shift = 0, sigma = 225): 
    n_frames = frames['n_frames']-1

    #load RFs
    if grid == 'square':
        rf_R2 = np.load(f'data/RFs/square_rf_R2_s{sigma}.npy')
        rf_R4 = np.load(f'data/RFs/square_rf_R4_s{sigma}.npy')
    elif grid == 'hex':
        rf_R2 = np.load(f'data/RFs/hex_rf_R2_s{sigma}.npy')
        rf_R4 = np.load(f'data/RFs/hex_rf_R4_s{sigma}.npy')
    elif grid == 'random':
        rf_R2 = np.load(f'data/RFs/rand_rf_R2_s{sigma}.npy')
        rf_R4 = np.load(f'data/RFs/rand_rf_R4_s{sigma}.npy')

    _,_,n_r2 = rf_R2.shape
    _,_,n_r4 = rf_R4.shape

    r4_A = np.zeros((n_r4,n_frames))
    r2_A = np.zeros((n_r2,n_frames))

    for f in range(n_frames):
        frame = frames[f]
        frame = np.hstack((frame[:,shift:] ,frame[:,:shift]) )
        for i in np.arange(n_r4):
            r4_A[i,f] = np.sum(np.multiply(frame,rf_R4[:,:,i]))
        r4_A[:,f] = norm(r4_A[:,f])
        for i in np.arange(n_r2):
            r2_A[i,f] = np.sum(np.multiply(frame,rf_R2[:,:,i]))
        r2_A[:,f] = norm(r2_A[:,f])

    r4_A[r4_A<0] = 0
    r2_A[r2_A<0] = 0

    return r4_A, r2_A


def get_square_grid(r,c):
    grid = np.zeros((r,c,2))
    for i in range(r):
        for j in range(c):
            grid[i,j,0] = j
            grid[i,j,1] = i
    return grid

def rescale_square_grid (grid,img_size):
    mx_x = np.max(grid[:,:,0]) + 1
    mx_y = np.max(grid[:,:,1])



    scale = img_size[1]/mx_x
    pad = (img_size[0]-(mx_y*scale))/2

    grid[:,:,0] = grid[:,:,0] * scale
    grid[:,:,1] = (grid[:,:,1] * scale) + pad
    return grid

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

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def getSquareRFs(grid, img_size, sigma):
    grid_x = grid[:,:,0].ravel()
    grid_y = grid[:,:,1].ravel()
    RFs = np.zeros((img_size[0],img_size[1],len(grid_x)))
    for i in range(len(grid_x)):
        if grid_x[i] == 0:
            rf_1 = gen_rf (img_size,(0,grid_y[i]),sigma)
            rf_2 = gen_rf (img_size,(360,grid_y[i]),sigma)
            RFs[:,:,i] = np.hstack([rf_1[:,:180],rf_2[:,180:]])
        else:
            RFs[:,:,i] = gen_rf (img_size,(grid_x[i],grid_y[i]),sigma)
    return RFs

def plot_square_grid(grid):
    w,h = grid[-1,-1]
    pad_h = 0# (95-h)/2
    pad_w = 0#(360-w)/2

    plt.figure()
    plt.scatter(grid[:,:,0]+pad_w,grid[:,:,1]+pad_h)
    plt.xlim([0,360])
    plt.ylim([0,360])
    plt.show()

def plot_square_grids(grid1,grid2):
    plt.figure()
    for grid in [grid1,grid2]:
        w,h = grid[-1,-1]
        pad_h = 0# (95-h)/2
        pad_w = 0#(360-w)/2

        plt.scatter(grid[:,:,0]+pad_w,grid[:,:,1]+pad_h)
    plt.xlim([0,360])
    plt.ylim([0,360])
    plt.show()

def gauss_dis (target, pos, sig = 40):
    d1 = np.sqrt((target[0] - pos[0])**2 + (target[1] - pos[1])**2)
    d2 = np.sqrt((target[0]+360 - pos[0])**2 + (target[1] - pos[1])**2)
    d3 = np.sqrt((target[0]-360 - pos[0])**2 + (target[1] - pos[1])**2)
    d = min(d1,d2,d3)
    return np.exp(-np.power(d, 2.) / (2 * np.power(sig, 2.)))

def getSquareWeights(grid, label = ''):
    # option for random choice between 0 and 1 (more lower values)
    rands = np.exp(np.arange(0,5,.1))
    rands = rands/np.max(rands)

    grid_x = grid[:,:,0].ravel()
    grid_y = grid[:,:,1].ravel()

    W = np.zeros((len(grid_x),len(grid_x)))
    for i in range(len(grid_x)):
        for j in range(len(grid_x)):
            if i != j:
                d = gauss_dis((grid_x[i],grid_y[i]),(grid_x[j],grid_y[j]))
                W[i,j] =  d * np.random.choice(rands) #np.random.rand() 
    if not(label == ''):
        plt.imshow(W, origin='lower', cmap ='jet')
        plt.colorbar()
        plt.savefig(f'data/RFs/plots/{label}.png', bbox_inches='tight')
        plt.savefig(f'data/RFs/plots/{label}.svg', bbox_inches='tight')
        plt.show()
    return W

def getHexWeights(grid, label = ''):
    # option for random choice between 0 and 1 (more lower values)
    rands = np.exp(np.arange(0,5,.1))
    rands = rands/np.max(rands)

    W = np.zeros((len(grid.keys()),len(grid.keys())))
    for i,k in enumerate(grid.keys()):
        for j,l in enumerate(grid.keys()):
            if i != j:
                d = gauss_dis((grid[k][1],grid[k][0]),(grid[l][1],grid[l][0]))
                W[i,j] =  d * np.random.choice(rands) #np.random.rand() 
    if not(label == ''):
        plt.imshow(W, origin='lower', cmap ='jet')
        plt.colorbar()
        plt.savefig(f'data/RFs/plots/{label}.png', bbox_inches='tight')
        plt.savefig(f'data/RFs/plots/{label}.svg', bbox_inches='tight')
        plt.show()
    return W

def get_hex_grid(img_size, rad):
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

def plot_hex_grid (img_size, grid):
    for k in grid.keys():
        plt.scatter(grid[k][1],grid[k][0])
    plt.ylim([0,img_size[0]])
    plt.xlim([0,img_size[1]])
    plt.show()

def getHexRFs(img_size, grid, sigma):
    RFs = np.zeros((img_size[0],img_size[1],len(grid.keys())))
    for i,k in enumerate(grid.keys()):
        RFs[:,:,i] = gen_rf (img_size,(grid[k][1],grid[k][0]),sigma)
    return RFs


def plotRF(n, RFs):
    plt.imshow(RFs[:,:,n], origin='lower', cmap ='jet')
    plt.show()

def plotAllRFs(RFs,label):
    n = RFs.shape[2]
    fig, axes = plt.subplots(n//2,2, figsize=(4,n//3))
    custom_params = {"axes.spines.right": False, 
                    "axes.spines.top": False,
                    "axes.spines.left": False,
                    "axes.spines.bottom": False}
    sns.set_theme(style="white", palette="pastel")#, rc=custom_params)
    plt.suptitle(label)
    j = 0
    for i in np.arange(n):
        if i==n//2: j = 1
        axes[i%(n//2),j].imshow(RFs[:,:,i], origin='lower', cmap ='jet')
        axes[i%(n//2),j].set_xticks([])
        axes[i%(n//2),j].set_yticks([])
    plt.savefig(f'data/RFs/plots/{label}_RFs.png', bbox_inches='tight')
    plt.savefig(f'data/RFs/plots/{label}_RFs.svg', bbox_inches='tight')
    plt.show()

def plotIndRFs(RFs,label):
    n = RFs.shape[2]
    sns.set_theme(style="white", palette="pastel")
    for i in np.arange(n):
        plt.imshow(RFs[:,:,i], origin='lower', cmap ='jet')
        plt.set_xticks([])
        plt.set_yticks([])
    plt.savefig(f'data/RFs/plots/Ind/{label}_RF{i}.png', bbox_inches='tight')
    plt.savefig(f'data/RFs/plots/Ind/{label}_RF{i}.svg', bbox_inches='tight')
    plt.show()


def getCombinedRFs(RFs, W):
    combRFs = np.zeros(RFs.shape)
    n = RFs.shape[2]
    for i in np.arange(n):
        exc = norm(RFs[:,:,i])
        Z = np.zeros(exc.shape)
        for j in np.arange(n):
            if i != j:
                Z = Z + (RFs[:,:,j] * W[i,j])

        rf = exc-norm(Z) 
        combRFs[:,:,i] = rf
    return combRFs


def get_rand_grid(n,img_size):
    height = img_size[0]
    width = img_size[1]
    buffer = 5

    grid = {}
    for i in np.arange(n):
        y = np.random.randint(0+buffer,height+buffer)
        x = np.random.randint(0,width/2)
        grid[i] = (y,x)
        grid[i+n] = (y,width-x)
    return grid





# For generating and saving the RFs and weights
def gen_all(img_size = (95, 360),sigma = 225):
    # for R2 = 42 fields
    # 3 rows of 14
    R2_square = get_square_grid(3,14)
    R2_square = rescale_square_grid (R2_square,img_size)
    # plot_square_grid(R2_square)
    w_R2 = getSquareWeights(R2_square, label=f'square_w_R2_s{sigma}') 
    rf_R2 = getSquareRFs(R2_square,img_size,sigma)
    plotAllRFs(rf_R2,'R2_square')
    plotAllRFs(getCombinedRFs(rf_R2, w_R2),'R2_square_comb')
    np.save(f'data/RFs/square_rf_R2_s{sigma}',rf_R2, allow_pickle=True)
    np.save(f'data/RFs/square_w_R2',w_R2, allow_pickle=True)


    # for R4 = 26 fields
    # 2 rows of 13
    R4_square = get_square_grid(2,13)
    R4_square = rescale_square_grid (R4_square,img_size)
    # plot_square_grid(R4_square)
    w_R4 = getSquareWeights(R4_square, label=f'square_w_R4_s{sigma}') 
    rf_R4 = getSquareRFs(R4_square,img_size,sigma)
    plotAllRFs(rf_R4, 'R4_square')
    plotAllRFs(getCombinedRFs(rf_R4, w_R4),'R4_square_comb')
    np.save(f'data/RFs/square_rf_R4_s{sigma}',rf_R4, allow_pickle=True)
    np.save(f'data/RFs/square_w_R4',w_R4, allow_pickle=True)

    # R4
    rad = 25 # 25 for imagesize (90,360) # 18 for imagesize (60,360) 
    R4_hex = get_hex_grid(img_size, rad)
    # plot_hex_grid(img_size,R4_hex)
    w_R4 = getHexWeights(R4_hex, label=f'hex_w_R4_s{sigma}') 
    rf_R4 = getHexRFs(img_size,R4_hex,sigma)
    plotAllRFs(rf_R4, 'R4_hex')
    plotAllRFs(getCombinedRFs(rf_R4, w_R4),'R4_hex_comb')
    np.save(f'data/RFs/hex_rf_R4_s{sigma}',rf_R4, allow_pickle=True)
    np.save(f'data/RFs/hex_w_R4',w_R4, allow_pickle=True)
    plotIndRFs(rf_R4, 'R4_hex')

    # # R2
    rad = 20 #20 for imagesize (90,360) # 15 for imagesize (60,360) 
    R2_hex = get_hex_grid(img_size, rad)
    # plot_hex_grid(img_size,R2_hex)
    w_R2 = getHexWeights(R2_hex, label=f'hex_w_R2_s{sigma}') 
    rf_R2 = getHexRFs(img_size,R2_hex,sigma)
    plotAllRFs(rf_R2, 'R2_hex')
    plotAllRFs(getCombinedRFs(rf_R2, w_R2),'R2_hex_comb')
    np.save(f'data/RFs/hex_rf_R2_s{sigma}',rf_R2, allow_pickle=True)
    np.save(f'data/RFs/hex_w_R2',w_R2, allow_pickle=True)
    plotIndRFs(rf_R2, 'R2_hex')


    # for R2 = 42 fields
    # random positions for 21
    R2_rand = get_rand_grid(21,img_size)
    # plot_hex_grid(img_size,R2_rand)
    w_R2 = getHexWeights(R2_rand, label=f'random_w_R2_s{sigma}') 
    rf_R2 = getHexRFs(img_size,R2_rand,sigma)
    plotAllRFs(rf_R2, 'R2_random')
    plotAllRFs(getCombinedRFs(rf_R2, w_R2),'R2_random_comb')
    np.save(f'data/RFs/rand_rf_R2_s{sigma}',rf_R2, allow_pickle=True)
    np.save(f'data/RFs/rand_w_R2',w_R2, allow_pickle=True)


    # for R4 = 26 fields
    # random positions for 13
    R4_rand = get_rand_grid(13,img_size)
    # plot_hex_grid(img_size,R4_rand)
    w_R4 = getHexWeights(R4_rand, label=f'random_w_R4_s{sigma}') 
    rf_R4 = getHexRFs(img_size,R4_rand,sigma)
    plotAllRFs(rf_R4, 'R4_random')
    plotAllRFs(getCombinedRFs(rf_R4, w_R4),'R4_random_comb')
    np.save(f'data/RFs/rand_rf_R4_s{sigma}',rf_R4, allow_pickle=True)
    np.save(f'data/RFs/rand_w_R4',w_R4, allow_pickle=True)


# gen_all()