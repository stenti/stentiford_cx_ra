import numpy as np
import matplotlib.pyplot as plt
import pygenn.genn_model as genn
import generateRFs as generateRFs
import process_video as process_video
import os.path
import pickle
from scipy.signal import find_peaks

# params = {'EPG_EPG_init': 0.3, 
#         'EPG_PEN_init': 2, 
#         'EPG_R_init': 0.1, 
#         'EPG_D7_init': 0.5, 
#         'R_EPG_init': -35, 
#         'PEN_EPG_init': 1.5, 
#         'D7_EPG_init': -35., 
#         'R_R_init': -5, 
#         'Rx_EPG_init': -0.4, 
#         'wmin': 1., 
#         'eta': 0.03, 
#         'rho': 0.06, 
#         'r_scale': 0.5, 
#         'I': 0.175, 
#         'inh_tau': 5.0, 
#         'exc_tau': 20.0}

params = {'EPG_EPG_init': 0.02, 
        'EPG_PEN_init': 0.13, 
        'EPG_R_init': 0.01, 
        'EPG_D7_init': 0.05, 
        'R_EPG_init': -1.3,
        'PEN_EPG_init': 0.14,
        'D7_EPG_init': -2.6,
        'R_R_init': -.3, 
        'Rx_EPG_init': -0.05, 
        'wmin': .2, 
        'eta': 0.01, 
        'rho': 0.06, 
        'r_scale': 0.35, 
        'I': 0.13, 
        'inh_tau': 50.0, 
        'exc_tau': 100.0}


fpath_super =  ['circling/1_circle_super','circling/2_circle_super','circling/3_circle_super','circling/4_circle_super','circling/5_circle_super',
                'circling/6_circle_super','circling/7_circle_super','circling/8_circle_super','circling/9_circle_super','circling/10_circle_super']

fpath_static =  ['circling/1_circle_static','circling/2_circle_static','circling/3_circle_static','circling/4_circle_static','circling/5_circle_static',
                'circling/6_circle_static','circling/7_circle_static','circling/8_circle_static','circling/9_circle_static','circling/10_circle_static']

fpath_rotate = ['rotation/1_3rev_static','rotation/2_3rev_static','rotation/4_3rev_static','rotation/5_3rev_static','rotation/6_3rev_static','rotation/7_3rev_static',
                'rotation/9_3rev_static','rotation/10_3rev_static','rotation/11_3rev_static','rotation/12_3rev_static','rotation/13_3rev_static','rotation/14_3rev_static',
                'rotation/15_3rev_static','rotation/16_3rev_static','rotation/17_3rev_static','rotation/18_3rev_static','rotation/19_3rev_static','rotation/20_3rev_static',
                'rotation/21_3rev_static','rotation/22_3rev_static','rotation/23_3rev_static','rotation/24_3rev_static','rotation/25_3rev_static']

fpath_simple = ['simple_stim/bar_stim_static','simple_stim/triangles_com_static','simple_stim/triangles_static']


allfiles = fpath_rotate
lbl = 'rotation'
learning = 'on'
plot_correlations = False
plot_figures = True

conv_size = {'static':'0','small':'50','med':'100','large':'150','super':'200'}
# speed = {'0':0.13,'50':.074,'100':.074,'150':.074,'200':.074}
speed = {'0':params['I'],'50':.074,'100':.074,'150':.074,'200':.074}


n_exp = 1 #20
shift = 22 # pixels approx degrees
n_shift = 1 #16
performance = np.zeros((n_exp,n_shift,len(allfiles),3))
spike_counts = np.zeros((len(allfiles),5))
correllations = np.zeros((len(allfiles),2))
bump = {}
for exp in range(n_exp):
    n_R4, n_R2 = (26,42)
    n_EPG = 16
    R4toEPG = np.random.rand(n_R4,n_EPG) * params['Rx_EPG_init']
    R2toEPG = np.random.rand(n_R2,n_EPG) * params['Rx_EPG_init']
    for sh in range(n_shift):
        print(f'LOOP {exp}:{n_exp}')
        print(f'shift {sh}:{n_shift}')
        for count,fpath in enumerate(allfiles):
            
            print(f'processing file: {fpath}')
            sz = conv_size[fpath.split('_')[-1]]
            fname = fpath.split('/')[-1]

            fps = 60

            if os.path.isfile(f'data/{fpath}.pkl'):
                with open(f'data/{fpath}.pkl', 'rb') as fp:
                    frames = pickle.load(fp)
                # print('Completed loading frames from file')
            else:
                path = f'data/{fpath}.mp4'
                frames = process_video.process(path)

            im_width, im_height = frames['shape']
            img_size = frames['shape']
            n_frames = frames['n_frames']

            [r4_A,r2_A] = generateRFs.getActivations(frames, shift = shift*sh)
            # print('Completed generating activations')

            r_scale = params['r_scale']

            #---------------------------------------------------------------------------
            # Parameters
            #---------------------------------------------------------------------------
            n_EPG = 16 # Number of E-PG Cells in model
            n_PEN = 16 # Number of P-EN Cells in model

            n_D7 = 8 # Number of delta7 Cells in model

            n_R = 1 # Number of general Ring Cells in model
            n_R4, n_R2 = (r4_A.shape[0], r2_A.shape[0]) # Number of visual Ring Cells in model


            # Experiment Parameters
            #---------------------------------------------------------------------------

            init = 1000 # ms
            t_single_frame = 1000/fps # ms
            if 'sim_len' in frames :
                sim_len = int(frames['sim_len'])
            else:
                sim_len = int(t_single_frame * frames['n_frames']) # ms
            t_1rev = sim_len//3               # ms in 1 revolution
            sim_len = sim_len + init     #ms
            n_1rev = int(t_1rev//t_single_frame)        # number of frames in 1 revolution
            r_single_frame = 360/n_1rev                 # rotation in single frame

            # rotate thru frames 
            tm_in = np.arange(init,sim_len,t_single_frame)
            tm_in = tm_in.astype(int)

            # Ring neurons
            R4_in = np.zeros((r4_A.shape[0],len(tm_in)))
            R2_in = np.zeros((r2_A.shape[0],len(tm_in)))
            loop = range(np.min([r4_A.shape[1], len(tm_in)]))
            for i in loop:
                R4_in[:,i] = r4_A[:,i]
                R2_in[:,i] = r2_A[:,i]

            # PEN input
            PEN_in = np.zeros((n_PEN,len(tm_in)))
            PEN_in[:8,:(n_1rev*2)] = speed[sz] #on for the first 2 rotations
            PEN_in[8:,:(n_1rev*2)] = -speed[sz]

            # Neuron parameters
            #---------------------------------------------------------------------------
            lif_params = {"C": 0.2, "TauM": 20.0, 
                        "Vrest": -70.0, "Vreset": -70.0, "Vthresh": -45.0, "Ioffset": 0.0,"TauRefrac": 2.0}
            lif_init = {"V": -52.0, "RefracTime": 0.0} 

            stim_tms = np.arange(0,10.0,0.1)
            stim_init = {"startSpike": [0], "endSpike": [len(stim_tms)]}
            init_bump = np.array([float(i == 1+(n_EPG//2)) or float(i == 1) for i in range(n_EPG)])

            # Synapse parameters
            #--------------------------------------------------------------------------
            GABAA_post_syn_params = params["inh_tau"]
            NMDA_post_syn_params = params["exc_tau"]

            if learning == 'on':
                stdp_params = {"tau": 50.0,"rho": params["rho"],"eta": params["eta"],"wMin": -params["wmin"],"wMax": .0, "start": 0, "end": float(tm_in[-1])}
            else:
                stdp_params = {"tau": 50.0,"rho": params["rho"],"eta": params["eta"],"wMin": -params["wmin"],"wMax": .0, "start": 0, "end": float(tm_in[int(n_1rev*2)])}

            learning = lbl

            # Define weights
            #---------------------------------------------------------------------------
            EPGtoEPG = np.load('data/weights/EPG_EPG_weights.npy') * params["EPG_EPG_init"]
            EPGtoPEN = np.load('data/weights/EPG_PEN_weights.npy') * params["EPG_PEN_init"]
            PENtoEPG = np.load('data/weights/PEN_EPG_weights.npy') * params["PEN_EPG_init"]
            RtoEPG = np.load('data/weights/R_EPG_weights.npy') * params["R_EPG_init"]
            EPGtoR = np.load('data/weights/EPG_R_weights.npy') * params["EPG_R_init"]
            D7toEPG = np.load('data/weights/D7_EPG_weights.npy') * params["D7_EPG_init"]
            EPGtoD7 = np.load('data/weights/EPG_D7_weights.npy') * params["EPG_D7_init"]
            R2toR2 = np.load(f'data/weights/R2_weights.npy') * params["R_R_init"]
            R4toR4 = np.load(f'data/weights/R4_weights.npy') * params["R_R_init"]

            # Weight update model
            #---------------------------------------------------------------------------
            timed_stdp = genn.create_custom_weight_update_class(
                "timed_stdp",
                param_names=[
                    "tau",      # 0 - Plasticity time constant (ms)
                    "rho",      # 1 - Target rate
                    "eta",      # 2 - Learning rate
                    "wMin",     # 3 - Minimum weight
                    "wMax",     # 4 - Maximum weight
                    "start",
                    "end"
                ],
                var_name_types=[("g", "scalar")],
                sim_code=
                """
                $(addToInSyn, $(g));
                if($(t) > $(start) && $(t) < $(end)){
                    scalar dt = $(t) - $(sT_post);
                    scalar timing = exp(-dt / $(tau)) - $(rho);
                    scalar newWeight = $(g) + ($(eta) * timing);
                    $(g) = fmin($(wMax), fmax($(wMin), newWeight));
                }
                """,
                learn_post_code=
                """
                if($(t) > $(start) && $(t) < $(end)){
                    scalar dt = $(t) - $(sT_pre);
                    scalar timing = exp(-dt / $(tau)) - $(rho);
                    scalar newWeight = $(g) + ($(eta) * timing);
                    $(g) = fmin($(wMax), fmax($(wMin), newWeight));
                }
                """,
                is_pre_spike_time_required=True,
                is_post_spike_time_required=True)
            

            #---------------------------------------------------------------------------
            # Build model
            #---------------------------------------------------------------------------
            model = genn.GeNNModel("float", "cx_ra", backend="SingleThreadedCPU") #RUN ON CPU
            # model = genn.GeNNModel("float", "cx_ra") #RUN ON GPU
            model.dT = 1.0

            # Neuron Populations
            #---------------------------------------------------------------------------
            EPG = model.add_neuron_population("EPG", n_EPG, "LIF", lif_params, lif_init)
            PEN = model.add_neuron_population("PEN", n_PEN, "LIF", lif_params, lif_init)
            R = model.add_neuron_population("R", n_R, "LIF", lif_params, lif_init)
            D7 = model.add_neuron_population("D7", n_D7, "LIF", lif_params, lif_init)
            R4 = model.add_neuron_population("R4", n_R4, "LIF", lif_params, lif_init)
            R2 = model.add_neuron_population("R2", n_R2, "LIF", lif_params, lif_init)

            R4.spike_recording_enabled = True
            R2.spike_recording_enabled = True
            EPG.spike_recording_enabled = True
            PEN.spike_recording_enabled = True
            R.spike_recording_enabled = True
            D7.spike_recording_enabled = True

            stim = model.add_neuron_population("Stim", 1, "SpikeSourceArray", {}, stim_init)
            stim.set_extra_global_param("spikeTimes", stim_tms)


            # Synapse Populations
            #---------------------------------------------------------------------------
            model.add_synapse_population(
                "Stim_EPG", "DENSE_INDIVIDUALG", 0,
                stim, EPG,
                "StaticPulse", {}, {"g": init_bump}, {}, {},
                "ExpCurr", {"tau": 1.0}, {})

            model.add_synapse_population(
                "EPG_EPG", "DENSE_INDIVIDUALG", 0,
                EPG, EPG,
                "StaticPulse", {}, {"g": EPGtoEPG.ravel()}, {}, {},
                "ExpCurr", NMDA_post_syn_params, {})

            model.add_synapse_population(
                "EPG_PEN", "DENSE_INDIVIDUALG", 0,
                EPG, PEN,
                "StaticPulse", {}, {"g": EPGtoPEN.ravel()}, {}, {},
                "ExpCurr", NMDA_post_syn_params, {})

            model.add_synapse_population(
                "PEN_EPG", "DENSE_INDIVIDUALG", 0,
                PEN, EPG,
                "StaticPulse", {}, {"g": PENtoEPG.ravel()}, {}, {},
                "ExpCurr", NMDA_post_syn_params, {})

            model.add_synapse_population(
                "R_EPG", "DENSE_INDIVIDUALG", 0,
                R, EPG,
                "StaticPulse", {}, {"g": RtoEPG.ravel()}, {}, {},
                "ExpCurr", GABAA_post_syn_params, {})

            model.add_synapse_population(
                "EPG_R", "DENSE_INDIVIDUALG", 0,
                EPG, R,
                "StaticPulse", {}, {"g": EPGtoR.ravel()}, {}, {},
                "ExpCurr", NMDA_post_syn_params, {})

            model.add_synapse_population(
                "D7_EPG", "DENSE_INDIVIDUALG", 0,
                D7, EPG,
                "StaticPulse", {}, {"g": D7toEPG.ravel()}, {}, {},
                "ExpCurr", GABAA_post_syn_params, {})

            model.add_synapse_population(
                "EPG_D7", "DENSE_INDIVIDUALG", 0,
                EPG, D7,
                "StaticPulse", {}, {"g": EPGtoD7.ravel()}, {}, {},
                "ExpCurr", NMDA_post_syn_params, {})

            R2_EPG_synapse = model.add_synapse_population(
                "R2_EPG", "DENSE_INDIVIDUALG", 0,
                R2, EPG,
                timed_stdp, stdp_params,  
                {"g": R2toEPG.ravel()}, {}, {},
                "ExpCurr", GABAA_post_syn_params, {})

            R4_EPG_synapse = model.add_synapse_population(
                "R4_EPG", "DENSE_INDIVIDUALG", 0,
                R4, EPG,
                timed_stdp, stdp_params,     
                {"g": R4toEPG.ravel()}, {}, {},
                "ExpCurr", GABAA_post_syn_params, {})

            model.add_synapse_population(
                "R2_R2", "DENSE_INDIVIDUALG", 0,
                R2, R2,
                'StaticPulse', {}, {"g": R2toR2.ravel()}, {}, {},
                "ExpCurr", GABAA_post_syn_params, {})

            model.add_synapse_population(
                "R4_R4", "DENSE_INDIVIDUALG", 0,
                R4, R4,
                'StaticPulse', {}, {"g": R4toR4.ravel()}, {}, {},
                "ExpCurr", GABAA_post_syn_params, {})


            # Current sources
            #---------------------------------------------------------------------------
            I_in = genn.create_custom_current_source_class(
                "I_in",
                var_name_types=[("magnitude", "scalar")],
                injection_code="$(injectCurrent, $(magnitude));")

            R4_input = model.add_current_source("R4_input", I_in,
                                                    R4, {}, {"magnitude": 0.0})
            R2_input = model.add_current_source("R2_input", I_in,
                                                    R2, {}, {"magnitude": 0.0})
            PEN_input = model.add_current_source("PEN_input", I_in,
                                                    PEN, {}, {"magnitude": 0.0})

            #---------------------------------------------------------------------------
            # Simulation
            #---------------------------------------------------------------------------
            model.build()
            model.load(num_recording_timesteps=sim_len)


            R4_EPG_g = np.zeros((3, n_R4, n_EPG))
            R2_EPG_g = np.zeros((3, n_R2, n_EPG))
            view_R4_EPG_g = R4_EPG_synapse.vars["g"].view.reshape((n_R4 , n_EPG))
            view_R2_EPG_g = R2_EPG_synapse.vars["g"].view.reshape((n_R2 , n_EPG))

            tmp = 0
            while model.timestep < sim_len:
                idx = np.where(tm_in == model.timestep)[0]
                if len(idx) :
                    PEN_input.vars["magnitude"].view[:] = PEN_in[:,idx[0]]
                    PEN_input.push_var_to_device("magnitude")
                    R4_input.vars["magnitude"].view[:] = R4_in[:,idx[0]] * r_scale 
                    R4_input.push_var_to_device("magnitude")
                    R2_input.vars["magnitude"].view[:] = R2_in[:,idx[0]] * r_scale
                    R2_input.push_var_to_device("magnitude")

                if model.timestep == 1 or model.timestep == int(tm_in[n_1rev]) or model.timestep == int(tm_in[(n_1rev*2)]):
                    # print('Saving weight snapshot')
                    R4_EPG_synapse.pull_var_from_device("g")
                    R2_EPG_synapse.pull_var_from_device("g")
                    R4_EPG_g[tmp,:,:] = view_R4_EPG_g[:]
                    R2_EPG_g[tmp,:,:] = view_R2_EPG_g[:]
                    tmp = tmp + 1

                model.step_time()
 
            #---------------------------------------------------------------------------
            # Retrieve spiking data
            #---------------------------------------------------------------------------
            model.pull_recording_buffers_from_device()

            EPG_spike_times, EPG_spike_ids = EPG.spike_recording_data
            PEN_spike_times, PEN_spike_ids = PEN.spike_recording_data
            R_spike_times, R_spike_ids = R.spike_recording_data
            D7_spike_times, D7_spike_ids = D7.spike_recording_data
            R2_spike_times, R2_spike_ids = R2.spike_recording_data
            R4_spike_times, R4_spike_ids = R4.spike_recording_data

            R4_EPG_synapse.pull_var_from_device("g")
            R2_EPG_synapse.pull_var_from_device("g")
            R4_EPG_g[2,:,:] = view_R4_EPG_g[:]
            R2_EPG_g[2,:,:] = view_R2_EPG_g[:]

            #---------------------------------------------------------------------------
            # Remap EPG spikes onto ring
            #---------------------------------------------------------------------------
            remap = {0:15, 1:1, 2:3, 3:5, 4:7, 5:9, 6:11, 7:13, 8:0, 9:2, 10:4, 11:6, 12:8, 13:10, 14:12, 15:14} #new weights mapping

            for i,id in enumerate(EPG_spike_ids):
                EPG_spike_ids[i] = remap[id]

            #---------------------------------------------------------------------------
            # Plot spike trains
            #---------------------------------------------------------------------------


            # FIND MOST ACTIVE CELL
            bin_sz = t_single_frame
            med_tms = np.arange(0,sim_len,bin_sz)
            med_cell = np.zeros(len(med_tms))
            med_cell[:] = np.nan
            binned = (EPG_spike_times//bin_sz).astype(int)

            id_unwrapped = np.unwrap(EPG_spike_ids, period = 16)
            gt_tms = np.arange(init,sim_len,bin_sz)
            gt = np.append(np.zeros(len(med_tms)-len(gt_tms)), np.arange(0,len(gt_tms)) * ((n_EPG*3)/len(gt_tms)))
            angle = (((gt%n_EPG)*22.5)//22.5).astype(int)

            for i,tm in enumerate(med_tms):
                cells = id_unwrapped[binned == i]
                if len(cells):
                    med_cell[i] = np.median(cells)
            med_cell = med_cell-med_cell[0]

            test_RMSE = np.sqrt(np.nanmean(np.square(np.subtract(gt[(n_1rev*2):],med_cell[(n_1rev*2):]))))
            if np.isnan(np.subtract(gt[(n_1rev*2):],med_cell[(n_1rev*2):])).sum() > 500:
                test_RMSE = np.nan
            print(f'test_RMSE = {test_RMSE}')
            performance[exp,sh,count,2] = test_RMSE


            if plot_figures:
                fig = plt.figure(figsize=(6, 4))
                fig.suptitle(fname)
                gs = fig.add_gridspec(3,1)
                ax1 = fig.add_subplot(gs[0,:])
                ax2 = fig.add_subplot(gs[1:,:])

                im1 = ax1.imshow(frames[0], aspect='auto', cmap = 'gray')#, vmax= .4, vmin = -.1)
                im2 = ax2.imshow(np.vstack([r2_A,r4_A]), aspect='auto', origin = 'lower')#, vmax= .4, vmin = -.1)
                ax1.set_yticks([])
                ax1.set_xticks([])
                ax2.set_yticks([0,41,67])
                ax2.set_xlabel('Frame')
                ax2.set_ylabel('Ring neuron')


                fig.colorbar(im1, ax=ax1)
                fig.colorbar(im2, ax=ax2)
                # fig.colorbar(im3, ax=axes[2])
                plt.savefig(f'results/{fpath}_activations.png', bbox_inches='tight')
                plt.savefig(f'results/{fpath}_activations.svg', bbox_inches='tight')
                plt.close(fig)

                # PAPER FIGURE B
                fig = plt.figure(figsize=(11, 4))
                fig.suptitle(f'{fname} error:{np.around(test_RMSE,4)} learning:{learning}')
                gs = fig.add_gridspec(6,18)
                ax1 = fig.add_subplot(gs[:2, :10])
                ax2 = fig.add_subplot(gs[2:4, :10])
                ax6 = fig.add_subplot(gs[4:, :10])
                ax3 = fig.add_subplot(gs[:, 11:13])
                ax4 = fig.add_subplot(gs[:, 13:15])
                ax5 = fig.add_subplot(gs[:, 15:17])

                ax1.fill_between([tm_in[(n_1rev*2)],sim_len], -1, n_R2 + n_R4 + 1,color='green', alpha=0.1)
                ax2.fill_between([tm_in[(n_1rev*2)],sim_len], -1, n_EPG + 1,color='green', alpha=0.1)
                ax6.fill_between([tm_in[(n_1rev*2)],sim_len], -1, 1081,color='green', alpha=0.1)
                ax2.scatter(EPG_spike_times, EPG_spike_ids, s=1, label=f'EPG spikes')
                ax1.scatter(R2_spike_times, R2_spike_ids , s=1, label=f'R2 spikes', c = 'Gold')
                ax1.scatter(R4_spike_times, R4_spike_ids + n_R2, s=1, label=f'R4 spikes', c = 'Orange')
                ax6.plot(med_tms,gt*22.5, c = 'Black', label='Ground Truth')
                ax6.scatter(med_tms,(med_cell*22.5), label='estimate', alpha = .7,s=1)
                # Label axes
                ax1.set_ylabel("Ring Neuron")
                ax2.set_ylabel("EPG neuron")
                ax6.set_ylabel("Rotation (deg)")
                ax6.set_xlabel("Time (ms)")
                ax2.set_yticks([0,15])
                ax1.set_yticks([0,41,67])
                ax6.set_yticks([0,360,720])
                ax1.set_xticks([])
                ax2.set_xticks([])
                ax1.legend(loc = 'upper left')#,bbox_to_anchor=(1., .95))
                ax2.legend(loc = 'upper left')#,bbox_to_anchor=(1., 0.1))
                ax6.legend(loc = 'upper left')
                ax2.set_xlim([0,sim_len])
                ax1.set_xlim([0,sim_len])
                ax6.set_xlim([0,sim_len])
                ax1.set_ylim([0,n_R2 + n_R4])
                ax2.set_ylim([0,n_EPG])
                ax6.set_ylim([0,1080])

                ax3.imshow(np.vstack([R2_EPG_g[0,:],R4_EPG_g[0,:]]), vmin=-params["wMin"], vmax=0.01,aspect='auto',)
                ax3.set_ylabel("Ring neurons")
                ax3.set_xlabel("EPG")
                ax3.set_title('Initial')
                ax4.imshow(np.vstack([R2_EPG_g[1,:],R4_EPG_g[1,:]]), vmin=-params["wMin"], vmax=0.01,aspect='auto',)
                ax4.set_xlabel("EPG")
                ax4.set_title('1 rev')            
                img = ax5.imshow(np.vstack([R2_EPG_g[2,:],R4_EPG_g[2,:]]), vmin=-params["wMin"], vmax=0.01, aspect='auto',)
                ax5.set_xlabel("EPG")
                ax5.set_title('2 rev')
                # fig.colorbar(img, ax=ax5, ticks = [0,-.2])
                ax3.set_yticks([0,41,67])
                ax4.set_yticks([])
                ax5.set_yticks([])
                ax3.set_xticks([0,15])
                ax4.set_xticks([0,15])
                ax5.set_xticks([0,15])

                plt.savefig(f'results/{fpath}_{learning}.png', bbox_inches='tight')
                plt.savefig(f'results/{fpath}_{learning}.svg', bbox_inches='tight')
                plt.close(fig)



            if plot_correlations:
                
                n_frames = frames['n_frames']
                n_1rev = n_frames//3
                # Matrix output  
                fig = plt.figure(figsize=(3,2))
                fig.suptitle(fname)
                ax = plt.axes()
                a_cor = process_video.activations_autocorrelation(r2_A,r4_A)
                im = ax.imshow(a_cor, origin = 'lower', vmax=1, vmin=0)
                ax.set_ylim([0,n_frames])
                ax.set_xlim([0,n_frames])
                ax.set_xticks([0,n_1rev,n_1rev*2],[0,360,720])
                ax.set_yticks([0,n_1rev,n_1rev*2],[0,360,720])
                ax.set_ylabel('Angle (deg)')
                ax.set_xlabel('Angle (deg)')
                plt.colorbar(im, ax = ax)
                plt.savefig(f'results/{fpath}_R_autocorrelation.png', bbox_inches='tight')
                plt.savefig(f'results/{fpath}_R_autocorrelation.svg', bbox_inches='tight')
                plt.close(fig)


                fig = plt.figure(figsize=(3,2))
                fig.suptitle(fname)
                ax = plt.axes()
                v_cor = process_video.video_autocorrelation(frames)
                im = ax.imshow(v_cor, origin = 'lower', vmax=1, vmin=0)
                ax.set_ylim([0,n_frames])
                ax.set_xlim([0,n_frames])
                ax.set_xticks([0,n_1rev,n_1rev*2],[0,360,720])
                ax.set_yticks([0,n_1rev,n_1rev*2],[0,360,720])
                ax.set_ylabel('Angle (deg)')
                ax.set_xlabel('Angle (deg)')
                plt.colorbar(im, ax = ax)
                plt.savefig(f'results/{fpath}_vid_autocorrelation.png', bbox_inches='tight')
                plt.savefig(f'results/{fpath}_vid_autocorrelation.svg', bbox_inches='tight')
                plt.close(fig)

                correllations[count,0] = np.mean(a_cor)
                correllations[count,1] = np.mean(v_cor)

                # SPIKE COUNT
                spk_count = np.zeros(n_R4+n_R2)
                for i in range(n_R4):
                    spk_count[i] = (len(R4_spike_ids[R4_spike_ids == i]))
                for i in range(n_R2):
                    spk_count[n_R4+i] = (len(R2_spike_ids[R2_spike_ids == i]))

                spike_counts[count,:] = [int(np.sum(spk_count)),int(np.max(spk_count)),np.mean(spk_count),0,0]

                R_mat = np.vstack([R2_mat,R4_mat])
                R_mat = R_mat[~np.all(R_mat == 0, axis=1)]
                R_mat = np.hstack([R_mat,R_mat[:,0:2]])
                totalR = R_mat.shape[0]
                spike_counts[count,3] = totalR # save total number of R neurons active
                angles = (np.arange(0,R_mat.shape[1])*22.5)*(np.pi/180)
                for r in range(totalR):
                    peaks, _ = find_peaks(R_mat[r,:],threshold=2, distance = 4)
                    # axes[r//4,r%4].plot(angles[peaks],R_mat[peaks,:], "x")
                    if len(np.unique(angles[peaks]))>1:
                        spike_counts[count,4] = spike_counts[count,4] + 1 #COUNT ALL CELLS THAT HAVE BIPOLAR DIRECTIONAL TUNING




        np.save(f'results/{lbl}_{n_exp}_shifting_{n_shift}_learn{learning}',performance)

if plot_correlations:
    np.save(f'results/{lbl}_spike_counts',spike_counts)
    np.save(f'results/{lbl}_correlations',correllations)
