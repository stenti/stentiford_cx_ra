import numpy as np
import matplotlib.pyplot as plt
import pygenn.genn_model as genn
import pandas as pd
from scipy.signal import find_peaks
import generate_stimuli as generate_stimuli

AVs = np.arange(10,100,5)
dataset = []

def get_current(av):
    return (0.0017915225715372924 * av) + (4.402231988516501e-06 * av**2) + 0.004988342068399486

for av in AVs:
    frames = generate_stimuli.generate_fromAV(av)
    print(f'AV: {av}')
    
    fname = f'bar_{av}'

    fps = 60

    im_width, im_height = frames['shape']
    img_size = frames['shape']
    n_frames = frames['n_frames']

    #---------------------------------------------------------------------------
    # Parameters
    #---------------------------------------------------------------------------
    n_EPG = 16 # Number of E-PG Cells in model
    n_PEN = 16 # Number of P-EN Cells in model

    n_D7 = 8 # Number of delta7 Cells in model

    n_R = 1 # Number of general Ring Cells in model

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

    I = get_current(av)
    gt = np.cumsum(np.ones(n_frames) * av/fps) 

    # PEN input
    PEN_in = np.zeros((n_PEN,len(tm_in)))
    PEN_in[:8,:] = I
    PEN_in[8:,:] = -I

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
    EPG_EPG_init =  0.02
    EPG_PEN_init =  0.13
    R_EPG_init =   -1.3
    EPG_R_init =    0.01
    PEN_EPG_init =  0.14
    D7_EPG_init =  -2.6
    EPG_D7_init =   0.05
    D7_D7_init =   -0.0
    R_R_init =     -.3 #-.08    -.2 #
    Rx_EPG_init =  -0.05

    GABAA_post_syn_params = {"tau": 50.0}
    NMDA_post_syn_params = {"tau": 100.0}


    # Define weights
    #---------------------------------------------------------------------------
    EPGtoEPG = np.load('data/weights/EPG_EPG_weights.npy') * EPG_EPG_init
    EPGtoPEN = np.load('data/weights/EPG_PEN_weights.npy') * EPG_PEN_init
    PENtoEPG = np.load('data/weights/PEN_EPG_weights.npy') * PEN_EPG_init
    RtoEPG = np.load('data/weights/R_EPG_weights.npy') * R_EPG_init
    EPGtoR = np.load('data/weights/EPG_R_weights.npy') * EPG_R_init
    D7toEPG = np.load('data/weights/D7_EPG_weights.npy') * D7_EPG_init
    EPGtoD7 = np.load('data/weights/EPG_D7_weights.npy') * EPG_D7_init

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


    tmp = 0
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
    PEN_spike_times, PEN_spike_ids = PEN.spike_recording_data
    R_spike_times, R_spike_ids = R.spike_recording_data
    D7_spike_times, D7_spike_ids = D7.spike_recording_data


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
    gt = np.append(np.zeros(len(med_tms)-len(gt)), gt)
    angle = (((gt%n_EPG)*22.5)//22.5).astype(int)

    for i,tm in enumerate(med_tms):
        cells = id_unwrapped[binned == i]
        if len(cells):
            med_cell[i] = np.median(cells)
    med_cell = med_cell-med_cell[0]

    test_RMSE = np.sqrt(np.nanmean(np.square(np.subtract(gt[(n_1rev*2):]/22.5,med_cell[(n_1rev*2):]))))
    print(f'test_RMSE = {test_RMSE}')

    data = {'AV':av,'RMSE':test_RMSE}
    dataset.append(data)

df = pd.DataFrame(dataset)
df.to_pickle(f'results/AVgrid/sweep_av_grids_3rev_PENonly.pkl')