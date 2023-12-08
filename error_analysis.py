import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng()

def reshape(data):
    RMSE = np.squeeze(data[:,:,:,2])
    rep = RMSE.shape[0]*RMSE.shape[1]
    RMSE = np.reshape(RMSE,(rep,RMSE.shape[2]))
    return RMSE

# rotate_on = np.delete(reshape(np.load('hexRFs/results/all_performance_25_shifting_learn0.005symetric0.1.npy')),2,axis = 1)
# rotate_off = np.delete(reshape(np.load('hexRFs/results/all_performance_25_shifting_nolearn0.005symetric0.1.npy')),2,axis = 1)

rotate_on = reshape(np.load('hexRFs/results/all_performance_20_shifting_learnon_0.01symetric0.06.npy'))
rotate_off = reshape(np.load('hexRFs/results/all_performance_20_shifting_learnoff_0.01symetric0.06.npy'))

circling_on_1 = reshape(np.load('translation/results/circling_25_shifting_learn0.005symetric0.1.npy'))
circling_off_1 = reshape(np.load('translation/results/circling_25_shifting_nolearn0.005symetric0.1.npy'))

circling_on_2 = reshape(np.load('translation/results/circling_25_shifting_learnon_0.005symetric0.1_half2.npy'))
circling_off_2 = reshape(np.load('translation/results/circling_25_shifting_learnoff_0.005symetric0.1_half2.npy'))

circling_on = np.hstack([circling_on_1,circling_on_2])
circling_off = np.hstack([circling_off_1,circling_off_2])

# double_3_off = reshape(np.load('translation/results/circling_double_3_25_shifting_learnoff_0.005symetric0.1.npy'))
double_3_on = reshape(np.load('translation/results/circling_double_3_20_shifting_learnon_0.01symetric0.06.npy'))

# double_on = reshape(np.load('translation/results/circling_double_25_shifting_learnon_0.005symetric0.1.npy'))


r_count = np.delete(np.load('hexRFs/results/all_counts.npy'),2,axis=0)
r_corr = np.delete(np.vstack([np.load('hexRFs/results/all_correlations.npy'),np.load('hexRFs/results/extra_correlations.npy')]),2,axis=0)

c_count = np.load('translation/results/all_counts.npy')
c_corr = np.load('translation/results/correlations.npy')

d3_count = np.load('translation/results/all_counts_double_3.npy')
d3_corr = np.load('translation/results/correlations_double_3.npy')

# d_count = np.load('translation/results/all_counts_double.npy')
# d_corr = np.load('translation/results/correlations_double.npy')

# static_on = reshape(np.load('translation/results/circling_25_shifting_learnon_0.005symetric0.1_static.npy'))
# sta_off = reshape(np.load('translation/results/circling_25_shifting_learnoff_0.005symetric0.1_static.npy'))
# super_on = reshape(np.load('translation/results/circling_25_shifting_learnon_0.005symetric0.1_super.npy'))
# sup_off = reshape(np.load('translation/results/circling_25_shifting_learnoff_0.005symetric0.1_super.npy'))

sta_off = reshape(np.load('translation/results/circling_10_shifting_learnoff_0.005symetric0.1_static.npy'))
sup_off = reshape(np.load('translation/results/circling_10_shifting_learnoff_0.005symetric0.1_super.npy'))

sta_on = reshape(np.load('translation/results/circling_25_shifting_learnon_0.005symetric0.1_static_noisy.npy'))
sta_off = reshape(np.load('translation/results/circling_25_shifting_learnoff_0.005symetric0.1_static_noisy.npy'))
sup_on = reshape(np.load('translation/results/circling_25_shifting_learnon_0.005symetric0.1_super_noisy.npy'))
sup_off = reshape(np.load('translation/results/circling_25_shifting_learnoff_0.005symetric0.1_super_noisy.npy'))



sup_sta_on = reshape(np.load('translation/results/circling_20_shifting_learnon_0.01symetric0.06_super_static.npy'))
sta_on = sup_sta_on[:,10:]
sup_on = sup_sta_on[:,:10]
sup_sta_off = reshape(np.load('translation/results/circling_20_shifting_learnoff_0.01symetric0.06_super_static.npy'))
sta_off = sup_sta_off[:,10:]
sup_off = sup_sta_off[:,:10]

# print(sup_off)

a = np.append(np.arange(circling_off.shape[1]//5*5),[0,2,3,4])

# sta_on = circling_on[:,a%5==0]
# sml_on = circling_on[:,a%5==1]
# med_on = circling_on[:,a%5==2]
# lar_on = circling_on[:,a%5==3]
# sup_on = circling_on[:,a%5==4]

# sta_off = circling_off[:,a%5==0]
# sml_off = circling_off[:,a%5==1]
# med_off = circling_off[:,a%5==2]
# lar_off = circling_off[:,a%5==3]
# sup_off = circling_off[:,a%5==4]

sta_count = c_count[a%5==0,:]
sml_count = c_count[a%5==1,:]
med_count = c_count[a%5==2,:]
lar_count = c_count[a%5==3,:]
sup_count = c_count[a%5==4,:]

sta_corr = c_corr[a%5==0,:]
sml_corr = c_corr[a%5==1,:]
med_corr = c_corr[a%5==2,:]
lar_corr = c_corr[a%5==3,:]
sup_corr = c_corr[a%5==4,:]


all_sta_on = np.hstack([rotate_on,sta_on])
all_sta_off = np.hstack([rotate_off,sta_off])
all_sta_count = np.vstack([r_count,sta_count])
all_sta_corr = np.vstack([r_corr,sta_corr])
all_d3_on = np.hstack([rotate_on,double_3_on])

all_sup_on = np.hstack([rotate_on,sup_on])
all_sup_off = np.hstack([rotate_off,sup_off])
all_sup_count = np.vstack([r_count,sup_count])
all_sup_corr = np.vstack([r_corr,sup_corr])


rotate_n = rotate_on.shape[1]
circle_n = sta_on.shape[1]


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# RMSE = sup_off
# count = sup_count
# corr = sup_corr
# nm = 'only_super_off'

# RMSE = sta_off
# count = sta_count
# corr = sta_corr
# nm = 'only_static_off'

# RMSE = rotate_on
# count = r_count
# corr = r_corr

# RMSE = all_sta_on
# count = all_sta_count
# corr = all_sta_corr
# nm = 'all_static_on'


# RMSE = all_sta_off
# count = all_sta_count
# corr = all_sta_corr
# nm = 'all_static_off'


# RMSE = all_sup_off
# count = all_sup_count
# corr = all_sup_corr
# nm = 'all_super_off'

RMSE = all_sup_on
count = all_sup_count
corr = all_sup_corr
nm = 'all_super_on'

# RMSE = all_d3_on
# count = []
# corr = []
# nm = 'double_3_on'

# RMSE = double_3_off
# count = d3_count
# corr = d3_corr

thresh = 1.5

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------


# Av = np.nanmean(RMSE, axis = 0)
# res = stats.bootstrap([RMSE], np.nanstd, confidence_level=0.95,random_state=rng)
# SE = res.standard_error

# fig, axes = plt.subplots(1,figsize=(8,4))
# for i in range(len(Av)):
#     axes.scatter(np.ones(RMSE.shape[0])*i,RMSE[:,i,], alpha = .05)
# axes.scatter(range(len(Av)),Av, c='black')
# axes.errorbar(range(len(Av)),Av, yerr=SE,ls = '', c='black',capsize=7)
# axes.set_xticks(range(len(Av)),np.arange(len(Av))+1)
# # axes.hlines([2],[-1],[len(Av)],['gray'])
# axes.hlines([1.5],[-1],[len(Av)],['gray'])

# axes.vlines([22.5],[-1],[25],['k'])
# axes.spines[['right', 'top']].set_visible(False)
# axes.set_ylabel('Error (deg)')
# axes.set_xlabel('Natural Scene')
# # axes.set_xlim([-1,n_scenes+n_scenes2])
# axes.set_ylim([0,30])
# # plt.savefig(f'translation/results/separation/test_RMSE_{nm}', bbox_inches='tight')
# plt.show()


# nans = np.zeros(RMSE.shape)
# nans[np.isnan(RMSE)] = 1
# mxfail = 0

# nfails = np.zeros(Av.shape)
# fail_group = np.zeros(Av.shape)
# fig, axes = plt.subplots(1,figsize=(8,3))
# for i in range(len(Av)):
#     nfails[i] = np.sum(nans[:,i])
#     axes.bar(i,nfails[i])
#     mxfail = np.max([mxfail,nfails[i]])
#     if nfails[i]>0:
#         fail_group[i] = 1
# axes.set_ylabel('# failed simulations')
# axes.set_xlabel('Natural Scene')
# axes.vlines([22.5],[-1],[25],['k'])
# # axes.set_xlim([-1,n_scenes])
# axes.set_ylim([0,mxfail+1])
# axes.set_xticks(range(len(Av)),np.arange(len(Av))+1)
# axes.spines[['right', 'top']].set_visible(False)
# plt.show()

nans = np.zeros(RMSE.shape)
nans[np.isnan(RMSE)] = 1
# RMSE[np.isnan(RMSE)] = 60
Av = np.nanmean(RMSE, axis = 0)
res = stats.bootstrap([RMSE], np.nanstd, confidence_level=0.95,random_state=rng)
SE = res.standard_error

print(Av)
print(SE)

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
# axes.hlines([2],[-1],[len(Av)],['gray'])
# axes.hlines([1.5],[-1],[len(Av)],['gray'])
# axes.vlines([22.5],[-1],[25],['k'])
axes.spines[['right', 'top']].set_visible(False)
axes.set_ylabel('Error (deg)')
# axes.set_xlabel('Natural Scene')
# axes.set_xlim([-1,n_scenes+n_scenes2])
# axes.set_ylim([0,25])
plt.savefig(f'translation/results/separation/test_RMSE_{nm}', bbox_inches='tight')
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
axes.set_ylabel('# failed simulations')
# axes.set_xlabel('Natural Scene')
# axes.vlines([22.5],[-1],[25],['k'])
# axes.set_xlim([-1,n_scenes])
axes.set_ylim([0,mxfail+1])
axes.set_xticks(ticks,np.append(np.arange(rotate_n)+1,np.arange(circle_n)+1))
axes.spines[['right', 'top']].set_visible(False)
plt.savefig(f'translation/results/separation/fails_{nm}', bbox_inches='tight')
plt.show()

# print(nfails)

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
    plt.savefig(f'translation/results/separation/{title}_{nm}.png', bbox_inches='tight')
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
plot_comp(label,comb,features,'SE')


# print(Av)
label = np.ones(Av.shape)
label[Av<thresh] = 0
# print(label)
plot_comp(label,comb,features,f'thresh_{thresh}')


# print(fail_group)
label = fail_group
plot_comp(label,comb,features,f'failure')


histlim = 5
histwidth = np.nanmax(RMSE, axis = 0) - np.nanmin(RMSE, axis = 0)
label = np.ones(histwidth.shape)
label[histwidth<histlim] = 0
plot_comp(label,comb,features,'histwidth')

# histwidth = np.nanmax(RMSE, axis = 0) - np.nanmin(RMSE, axis = 0)
# fig, axes = plt.subplots(6,6, figsize=(15,10))
# plt.tight_layout()
# for v in range(len(Av)):
#     for i in range(20):
#         if histwidth[v]<histlim:
#             colour = 'Blue'
#         else: colour = 'red'
#         axes[v%6,v//6].hist(RMSE[:,v],bins = np.arange(0,5,.2), color=colour)
#         axes[v%6,v//6].set_xlabel(f'{np.around(histwidth[v],4)}   {[histwidth[v]<histlim]}')
#         axes[v%6,v//6].set_ylabel(f'Scene {v+1}')
#         # axes[v%6,v//6].ylim([0,100])
# plt.savefig(f'translation/results/separation/hists_{nm}.png', bbox_inches='tight')
# plt.show()



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
    plt.savefig(f'translation/results/separation/{title}_3_{nm}.png', bbox_inches='tight')
    plt.show()



SE_lim = .15
label = np.ones(SE.shape)
label[SE<SE_lim] = 0
label[nfails>3.2] = 2
print(label)
plot_comp_3(label,comb,features,'SE')