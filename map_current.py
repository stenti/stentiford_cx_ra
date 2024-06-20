import numpy as np
import matplotlib.pyplot as plt
import pygenn.genn_model as genn
from PIL import Image
import scipy.stats as stats
from scipy.signal import find_peaks
import seaborn as sns

sns.set_theme()
sns.set_style('white')

search = True
curve_fit = True
plot = False


if search:
    fps = 60

    stdsim_len = 15000
    input = np.arange(0.065,.25,0.001)
    output = np.zeros(len(input))
    AV = np.zeros(len(input))
    output2 = np.zeros(len(input))
    print(len(input))

    for count,I_input in enumerate(input):
        print(f'Sim count: {count}/{len(input)} I_input: {I_input}')
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

        sim_len = stdsim_len + init

        tm_in = np.arange(init,sim_len,t_single_frame)
        tm_in = tm_in.astype(int)

        # PEN input
        PEN_in = np.zeros((n_PEN,len(tm_in)))
        PEN_in[:8,:] = I_input
        PEN_in[8:,:] = -I_input

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


        # FIND MOST ACTIVE CELL
        bin_sz = t_single_frame #20 #ms
        med_tms = np.arange(0,sim_len,bin_sz)
        med_cell = np.zeros(len(med_tms))
        med_cell[:] = np.nan
        binned = (EPG_spike_times//bin_sz)*bin_sz

        id_unwrapped = np.unwrap(EPG_spike_ids, period = 16)

        n_angles = int(360//22.5)
        for i,tm in enumerate(med_tms):
            cells = id_unwrapped[binned == tm]
            if len(cells):
                med_cell[i] = np.median(cells)
        med_cell = med_cell-med_cell[0]

        if plot:
            fig, axes = plt.subplots(3, sharex=True, figsize=(6, 5))
            fig.suptitle(f'dg travelled = {np.nanmax(med_cell*22.5)} AV = {np.nanmax(med_cell*22.5)/15}')
            axes[1].scatter(EPG_spike_times, EPG_spike_ids, s=1, label=f'EPG')
            axes[0].scatter(PEN_spike_times, PEN_spike_ids, s=1, label=f'PEN')
            axes[2].plot(med_tms,(med_cell*22.5), label='estimate')

            # Label axes
            axes[0].set_ylabel("PEN Neuron")
            axes[1].set_ylabel("EPG neuron")
            axes[2].set_ylabel("Heading Estimate (deg)")
            axes[2].set_xlabel("Time (ms)")
            axes[0].set_yticks([0,15])
            axes[1].set_yticks([0,15])
            axes[1].set_ylim([-1,16])
            axes[0].legend(loc = 'upper left')
            axes[1].legend(loc = 'upper left')
            axes[2].legend(loc = 'upper left')

            plt.xlim([0,sim_len])

            plt.show()

        AV[count] = np.nanmax(med_cell*22.5)/15
        print(I_input,AV[count])

        output[count] = med_tms[np.where(med_cell>16)[0][0]]

        tm_1rev = med_tms[np.where(med_cell>16)[0][0]]
        n_fr_1rev = tm_1rev/t_single_frame
        a_1fm = 360/n_fr_1rev
        output2[count] = a_1fm

    np.save('results/currentMap/current_map',np.vstack([input,output]))
    np.save('results/currentMap/AV',np.vstack([input,AV]))
    np.save('results/currentMap/angularChange_map',np.vstack([input,output2]))

    # input and output of fuction to get xurrent from AV
    out = input        # current input to model
    inp = AV     # degress travelled in 1 second

else:
    # inout = np.load('results/currentMap/current_map_2.npy')
    # # input = inout[0,:]
    # # output = inout[1,:]/1000
    # output = inout[0,:]         # current input to model
    # input = inout[1,:]/1000     # time for one revolution in seconds

    # inout2 = np.load('results/currentMap/angularChange_map.npy')
    # output2 = inout[1,:]/1000

    inout = np.load('results/currentMap/AV.npy')
    out = inout[0,:]         # current input to model
    inp = inout[1,:]     # degress travelled in 1 second

# plt.scatter(input,output)
# plt.ylabel('Input current')
# plt.xlabel('Time for 1 rev (s)')
# plt.show()

# input = 360/input

inp = np.append([0],inp)
out = np.append([0],out)

inp = inp[out<.2]
out = out[out<.2]

plt.scatter(inp,out)
plt.ylabel('Input current')
plt.xlabel('angular velocity (deg/s)')
plt.show()

# plt.scatter(input,output2)
# plt.xlabel('Input current')
# plt.ylabel('Angle change per frame')
# plt.show()


if curve_fit:    

    # curve-fit() function imported from scipy
    from scipy.optimize import curve_fit
    
    # Test function with coefficients as parameters

    def test_line(x, a):
        return a * x
    
    def test_sin(x, a, b):
        return a * np.sin(b * x)

    def test_exp(x, a, b):
        return a*np.exp(b*x)

    def test_square(x, a, b, c):
        return (a * x) + (b * x**2) + c
    
    def test_3(x, a, b, c, d):
        return (a * x) + (b * x**2) + (c * x**3) + d
    
    def test_4(x, a, b, c, d, e):
        return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + e
    
    def test_5(x, a, b, c, d, e, f):
        return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f
    
    # curve_fit() function takes the test-function
    # x-data and y-data as argument and returns 
    # the coefficients a and b in param and
    # the estimated covariance of param in param_cov

    p0, param_cov = curve_fit(test_line, inp,out)
    ans0 = test_line(inp, p0[0])

    p1, param_cov = curve_fit(test_sin, inp,out)
    ans1 = test_sin(inp, p1[0], p1[1])

    p2, param_cov = curve_fit(test_exp, inp,out)
    ans2 = test_exp(inp, p2[0], p2[1])

    p3, param_cov = curve_fit(test_square, inp,out)
    ans3 = test_square(inp, p3[0], p3[1], p3[2])
    print(p3[0], p3[1], p3[2])

    p4, param_cov = curve_fit(test_3, inp,out)
    ans4 = test_3(inp, p4[0], p4[1], p4[2],p4[3])
    # print(p4[0], p4[1], p4[2],p4[3])

    p5, param_cov = curve_fit(test_4, inp,out)
    ans5 = test_4(inp, p5[0], p5[1], p5[2],p5[3],p5[4])

    p6, param_cov = curve_fit(test_5, inp,out)
    ans6 = test_5(inp, p6[0], p6[1], p6[2],p6[3],p6[4],p6[5])
    

    plt.figure(figsize=(3,3))
    cmap = sns.color_palette("rocket")
    plt.plot(inp,out, '.', label ="data",c = cmap[2])
    # plt.plot(inp, ans0, '--', label ="line fit")
    # plt.plot(inp, ans1, '--', label ="sine fit")
    # plt.plot(inp, ans2, '--', label ="exp fit")
    plt.plot(inp, ans3, '--', label ="x2 fit",c = cmap[0])
    # plt.plot(inp, ans4, '--', label ="x3 fit")
    # plt.plot(inp, ans5, '--', label ="x4 fit")
    # plt.plot(inp, ans6, '--', label ="x5 fit")
    # plt.legend()
    plt.ylabel('Input current')
    plt.xlabel('Angular Velocity (deg/s)')

    sns.despine()
    
    plt.savefig('results/currentMap/currentMap.svg', bbox_inches='tight')
    plt.savefig('results/currentMap/currentMap.png', bbox_inches='tight')
    plt.show()


