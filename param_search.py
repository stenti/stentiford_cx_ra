import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import pygenn.genn_model as genn

def grid_search():
    t = time.time()
    data = []
    count = 0
    for j,EPG_PEN in enumerate(np.arange(.05,.25,.02)):
        for m,PEN_EPG in enumerate(np.arange(.05,.25,.02)):
            for n,R_EPG in enumerate([0.5,0.75,1,1.5,2,3,5]):
                for p,D7_EPG in enumerate([0.5,0.75,1,1.5,2,3,5]):

                    params = {
                    'EPG_EPG_init': 0.02,
                    'EPG_PEN_init': np.around(EPG_PEN,3),
                    'EPG_R_init':   0.01,
                    'EPG_D7_init':  0.05,

                    'R_EPG_init':  -np.around(R_EPG,3),
                    'PEN_EPG_init': np.around(PEN_EPG,3),
                    'D7_EPG_init': -np.around(D7_EPG,3),
                    'inh_tau':      50,
                    'exc_tau':      100,
                    'I':            0.13,
                    }

                    outputs = model.model(params)

                    params['success'] = outputs['success']
                    params['Rotations'] = outputs['Rotations']
                    params['av_Bump_width']= outputs['av_Bump_width']
                    params['min_Bump_width']= outputs['min_Bump_width']
                    params['max_Bump_width'] = outputs['max_Bump_width']
                    params['activity_ratio'] = outputs['activity_ratio']
                    params['last100'] = outputs['last100']
                    params['first100'] = outputs['first100']
                    params['last_width'] = outputs['last_width']
                

                    print(f'{count}/15679 {params}')
                    count = count +1

                    data.append(params)

        df = pd.DataFrame(data)
        df.to_pickle(f'systematic_param_search/results/spread/grid_search_I0.13_2.pkl')

    print(f'time taken: {time.time() - t}')



def model (params):
    output = True

    #---------------------------------------------------------------------------
    # Parameters
    #---------------------------------------------------------------------------
    n_EPG = 16 # Number of E-PG Cells in model
    n_PEN = 16 # Number of P-EN Cells in model

    n_D7 = 8 # Number of delta7 Cells in model

    n_R = 1 # Number of general Ring Cells in model

    # Experiment Parameters
    #---------------------------------------------------------------------------

    fps = 60
    init = 1000 # ms
    t_single_frame = 1000/fps # ms
    sim_len = 20000

    # rotate thru frames 
    tm_in = np.arange(init,sim_len,t_single_frame)

    #  PEN input
    PEN_in = np.zeros((n_PEN,len(tm_in)))
    PEN_in[:8,:] = params['I']
    PEN_in[8:,:] = -params['I']

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
    EPG_EPG_init =  params['EPG_EPG_init']
    EPG_PEN_init =  params['EPG_PEN_init']
    R_EPG_init =    params['R_EPG_init']
    EPG_R_init =    params['EPG_R_init']
    PEN_EPG_init =  params['PEN_EPG_init']
    D7_EPG_init =   params['D7_EPG_init']
    EPG_D7_init =   params['EPG_D7_init']


    GABAA_post_syn_params = {"tau": 50.0}
    NMDA_post_syn_params = {"tau": 100.0}

    # Define weights
    #---------------------------------------------------------------------------


    EPGtoEPG = np.load('cx_ra/weights/EPG_EPG_weights.npy') * EPG_EPG_init
    EPGtoPEN = np.load('cx_ra/weights/EPG_PEN_weights.npy') * EPG_PEN_init
    PENtoEPG = np.load('cx_ra/weights/PEN_EPG_weights.npy') * PEN_EPG_init
    RtoEPG = np.load('cx_ra/weights/R_EPG_weights.npy') * R_EPG_init
    EPGtoR = np.load('cx_ra/weights/EPG_R_weights.npy') * EPG_R_init
    D7toEPG = np.load('cx_ra/weights/D7_EPG_weights.npy') * D7_EPG_init
    EPGtoD7 = np.load('cx_ra/weights/EPG_D7_weights.npy') * EPG_D7_init


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

    # Current sources
    #---------------------------------------------------------------------------
    I_in = genn.create_custom_current_source_class(
        "I_in",
        var_name_types=[("magnitude", "scalar")],
        injection_code="$(injectCurrent, $(magnitude));")

    PEN_input = model.add_current_source("PEN_input", I_in,
                                            PEN, {}, {"magnitude": 0.0})

    #---------------------------------------------------------------------------
    # Simulation
    #---------------------------------------------------------------------------
    model.build()
    model.load(num_recording_timesteps=sim_len)

    while model.timestep < sim_len:
        idx = np.where(tm_in == model.timestep)[0]
        if len(idx) :
            PEN_input.vars["magnitude"].view[:] = PEN_in[:,idx[0]]
            PEN_input.push_var_to_device("magnitude")

        model.step_time()


    #---------------------------------------------------------------------------
    # Retrieve spiking data
    #---------------------------------------------------------------------------
    model.pull_recording_buffers_from_device()

    EPG_spike_times, EPG_spike_ids = EPG.spike_recording_data

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

    for i,tm in enumerate(med_tms):
        cells = id_unwrapped[binned == i]
        if len(cells):
            med_cell[i] = np.median(cells)
    med_cell = med_cell-med_cell[0]


    bin_sz = t_single_frame #20 #ms
    med_tms = np.arange(0,sim_len,bin_sz)
    med_cell = np.zeros(len(med_tms))
    med_cell[:] = np.nan
    bump_width = np.zeros(len(med_tms))
    bump_width[:] = np.nan
    binned = (EPG_spike_times//bin_sz).astype(int)

    activity_ratio = len(np.unique(binned))/len(med_tms)

    id_unwrapped = np.unwrap(EPG_spike_ids, period = 16)

    for i,tm in enumerate(med_tms):
        cells = id_unwrapped[binned == i]
        if len(cells):
            med_cell[i] = np.median(cells)
            bump_width[i] = len(np.unique(cells))
    med_cell = abs(med_cell-med_cell[0]) # ONLY WORKS IF ROTATION IN ONE DIRECTION

    first_100 = abs(np.nanmean(med_cell[:10]*22.5))
    last_100 = abs(np.nanmean(med_cell[:-10]*22.5))
    last_width = np.nanmean(bump_width[:-10])
    av_bump_width = np.nanmean(bump_width)
    rotations = np.nanmax(med_cell)/16    #this isnt doing what i think - direction of movement doesnt matter

    output = False
    if last_100 > first_100:               #this isnt doing what i think - direction of movement doesnt matter
        if last_width > 2:
            if av_bump_width > 2 and av_bump_width < 9:
                if rotations < 10 and rotations > 1:
                    if activity_ratio > 0.5:
                        # plot_figures = True
                        output = True

    outputs = {'success':output,
                'Rotations': rotations,
                'av_Bump_width': av_bump_width,
                'min_Bump_width': np.nanmin(bump_width),
                'max_Bump_width': np.nanmax(bump_width),
                'activity_ratio': activity_ratio,
                'last100':last_100,
                'first100':first_100,
                'last_width':last_width
                }

    return outputs



grid_search()