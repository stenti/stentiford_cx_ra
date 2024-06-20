import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng()

def reshape(data):
    RMSE = np.squeeze(data[:,:,:,2])
    rep = RMSE.shape[0]*RMSE.shape[1]
    RMSE = np.reshape(RMSE,(rep,RMSE.shape[2]))
    return RMSE

rotate_on = reshape(np.load('results/rotation_20_shifting_learnon.npy'))
rotate_off = reshape(np.load('results/rotation_20_shifting_learnoff.npy'))

sup_sta_on = reshape(np.load('results/circling_20_shifting_learnon_super_static.npy'))
sta_on = sup_sta_on[:,10:]
sup_on = sup_sta_on[:,:10]
sup_sta_off = reshape(np.load('results/circling_20_shifting_learnoff_super_static.npy'))
sta_off = sup_sta_off[:,10:]
sup_off = sup_sta_off[:,:10]

r_count = np.delete(np.load('results/rotation_counts.npy'),2,axis=0)
r_corr = np.delete(np.vstack([np.load('results/rotation_correlations.npy'),np.load('results/extra_correlations.npy')]),2,axis=0)

c_count = np.load('results/all_counts.npy')
c_corr = np.load('results/correlations.npy')

a = np.append(np.arange(49//5*5),[0,2,3,4])
sup_count = c_count[a%5==4,:]
sup_corr = c_corr[a%5==4,:]

all_sup_on = np.hstack([rotate_on,sup_on])
all_sup_off = np.hstack([rotate_off,sup_off])
all_sup_count = np.vstack([r_count,sup_count])
all_sup_corr = np.vstack([r_corr,sup_corr])

rotate_n = rotate_on.shape[1]
circle_n = sta_on.shape[1]


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

# RMSE = all_sup_off
# count = all_sup_count
# corr = all_sup_corr
# nm = 'all_super_off'

RMSE = all_sup_on
count = all_sup_count
corr = all_sup_corr
nm = 'all_super_on'

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

nans = np.zeros(RMSE.shape)
nans[np.isnan(RMSE)] = 1
# RMSE[np.isnan(RMSE)] = 60
Av = np.nanmean(RMSE, axis = 0)
print(Av)
res = stats.bootstrap([RMSE], np.nanstd, confidence_level=0.95,random_state=rng)
SE = res.standard_error

fig, axes = plt.subplots(1,figsize=(8,4))
for i in range(len(Av)):
    if i >= rotate_n: j = i+1 
    else: j = i
    axes.scatter(np.ones(RMSE.shape[0])*j,RMSE[:,i,], alpha = .05)
ticks = np.arange(len(Av))
ticks[ticks>=rotate_n] = ticks[ticks>=rotate_n] +1
axes.scatter(ticks,Av, c='black')
axes.errorbar(ticks,Av, yerr=SE,ls = '', c='black',capsize=7)
axes.set_xticks(ticks,np.append(np.arange(rotate_n)+1,np.arange(circle_n)+1))
axes.spines[['right', 'top']].set_visible(False)
axes.set_ylabel('Error (deg)')
plt.savefig(f'results/comparision/test_RMSE_{nm}', bbox_inches='tight')
plt.show()


mxfail = 0
nfails = np.zeros(Av.shape)
fail_group = np.zeros(Av.shape)
fig, axes = plt.subplots(1,figsize=(8,1.5))
for i in range(len(Av)):
    if i>= rotate_n: j = i+1 
    else: j = i
    nfails[i] = np.sum(nans[:,i])
    axes.bar(j,nfails[i])
    mxfail = np.max([mxfail,nfails[i]])
    if nfails[i]>0:
        fail_group[i] = 1
    print(nfails)
axes.set_ylabel('# failed simulations')
axes.set_ylim([0,mxfail+1])
axes.set_xticks(ticks,np.append(np.arange(rotate_n)+1,np.arange(circle_n)+1))
axes.spines[['right', 'top']].set_visible(False)
plt.savefig(f'results/comparision/fails_{nm}', bbox_inches='tight')
plt.show()

# COMPARING MEASUREs OF IMAGE DIFFERENCE

def plot_comp(label,comb,features,title):
    fig, axes = plt.subplots(1,len(features)-1, figsize=(12,2.5))
    plt.suptitle(title)
    plt.tight_layout()
    for n in range(len(features)-1):
        success = comb[label == 0,n]
        #print(success)
        Poor = comb[label == 1,n]
        SD = np.std(success)
        successSE = SD/len(success)
        res_s = stats.bootstrap([success], np.std, confidence_level=0.95,random_state=rng)
        res_f = stats.bootstrap([Poor], np.std, confidence_level=0.95,random_state=rng)
        SD = np.std(Poor)
        PoorSE = SD/len(Poor)
        SE = [successSE,PoorSE]
        SE = [res_s.standard_error,res_f.standard_error]

        successAv = [np.mean(success),np.mean(Poor)]

        tt = stats.ttest_ind(success, Poor, equal_var=False)
        #print(f'{features[n]} ttest: {tt}')

        axes[n].set_title(f'p = {np.around(tt[1],decimals=5)}',fontsize=10)
        axes[n].scatter(np.zeros(len(success)),success, alpha = .1)
        axes[n].scatter(np.ones(len(Poor)),Poor, alpha = .1)
        axes[n].scatter(range(2),successAv, c='black')
        axes[n].errorbar(range(2),successAv, yerr=SE,ls = '', c='black',capsize=7)
        axes[n].set_ylabel(features[n])
        axes[n].set_xticks([0,1],['Low','High'])
        axes[n].set_xlabel('Variance')
        axes[n].set_xlim([-1,2])
        axes[n].spines[['right', 'top']].set_visible(False)
    plt.savefig(f'results/comparision/{title}_{nm}.png', bbox_inches='tight')
    plt.show()


# print(count)

comb = np.hstack([np.reshape(nfails,(len(SE),1)),count[:,2:],corr,np.reshape(Av,(len(SE),1))])
features = ['# bump fail','Mean spikes','# R neurons active','# multipolar R neurons','Mean activations correlation','Mean frame correlation','Mean RMSE']


SE_lim = .15
SE = np.array(SE)
# print(f' SE: {SE}')
label = np.ones(SE.shape)
label[SE<SE_lim] = 0
# print(label)
# plot_comp(label,comb,features,'SE')


def plot_comp_3(label,comb,features,title):

    fig, axes = plt.subplots(1,len(features)-1, figsize=(12,3))
    plt.suptitle(title)
    plt.tight_layout()
    for n in range(len(features)-1):
        a = comb[label == 0,n]
        b = comb[label == 1,n]
        c = comb[label == 2,n]
        res_a = stats.bootstrap([a], np.std, confidence_level=0.95,random_state=rng)
        res_b = stats.bootstrap([b], np.std, confidence_level=0.95,random_state=rng)
        res_c = stats.bootstrap([c], np.std, confidence_level=0.95,random_state=rng)
        SE = [res_a.standard_error,res_b.standard_error,res_c.standard_error]

        Avg = [np.mean(a),np.mean(b),np.mean(c)]
        anova = stats.f_oneway(a, b, c)
        tukey = stats.tukey_hsd(a, b, c)
        print(tukey)
        
        axes[n].set_title(f'p = {np.around(anova[1],decimals=5)}',fontsize=10)
        axes[n].scatter(np.zeros(len(a)),a, alpha = .1)
        axes[n].scatter(np.ones(len(b)),b, alpha = .1)
        axes[n].scatter(np.ones(len(c))*2,c, alpha = .1)
        axes[n].scatter(range(len(Avg)),Avg, c='black')
        axes[n].errorbar(range(len(Avg)),Avg, yerr=SE,ls = '', c='black',capsize=7)
        axes[n].set_ylabel(features[n])
        axes[n].set_xticks([0,1,2],['Low','High','Fails'])
        axes[n].set_xlim([-1,3])
        axes[n].spines[['right', 'top']].set_visible(False)
    plt.savefig(f'results/comparision/{title}_3_{nm}.png', bbox_inches='tight')
    plt.show()



SE_lim = .15
label = np.ones(SE.shape)
label[SE<SE_lim] = 0
label[nfails>3.2] = 2
plot_comp_3(label,comb,features,'SE')
print(label)





def plot_comp_rot_circ(label,Av):
    fig = plt.figure( figsize=(12,2.5))
    plt.tight_layout()

    success = Av[label == 0]
    #print(success)
    Poor = Av[label == 1]
    SD = np.std(success)
    successSE = SD/len(success)
    res_s = stats.bootstrap([success], np.std, confidence_level=0.95,random_state=rng)
    res_f = stats.bootstrap([Poor], np.std, confidence_level=0.95,random_state=rng)
    SD = np.std(Poor)
    PoorSE = SD/len(Poor)
    SE = [successSE,PoorSE]
    SE = [res_s.standard_error,res_f.standard_error]

    successAv = [np.mean(success),np.mean(Poor)]

    tt = stats.ttest_ind(success, Poor, equal_var=False)
    #print(f'{features[n]} ttest: {tt}')

    plt.suptitle(f'p = {np.around(tt[1],decimals=5)}',fontsize=10)
    plt.scatter(np.zeros(len(success)),success, alpha = .1)
    plt.scatter(np.ones(len(Poor)),Poor, alpha = .1)
    plt.scatter(range(2),successAv, c='black')
    plt.errorbar(range(2),successAv, yerr=SE,ls = '', c='black',capsize=7)
    plt.ylabel(RMSE)
    plt.xticks([0,1],['Low','High'])
    plt.xlabel('Variance')
    plt.xlim([-1,2])
    plt.savefig(f'results/comparision/rot_circ.png', bbox_inches='tight')
    plt.show()


label = np.ones(SE.shape)
label[:rotate_n] = 0
plot_comp_rot_circ(label,Av)
print(label)