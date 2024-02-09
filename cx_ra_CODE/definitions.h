#pragma once
#define EXPORT_VAR extern
#define EXPORT_FUNC
// Standard C++ includes
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

// Standard C includes
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#define DT 1.00000000000000000e+00f
typedef float scalar;
#define SCALAR_MIN 1.175494351e-38f
#define SCALAR_MAX 3.402823466e+38f

#define TIME_MIN 1.175494351e-38f
#define TIME_MAX 3.402823466e+38f

// ------------------------------------------------------------------------
// bit tool macros
#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x
#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1
#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
EXPORT_VAR unsigned long long iT;
EXPORT_VAR float t;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
EXPORT_VAR double initTime;
EXPORT_VAR double initSparseTime;
EXPORT_VAR double neuronUpdateTime;
EXPORT_VAR double presynapticUpdateTime;
EXPORT_VAR double postsynapticUpdateTime;
EXPORT_VAR double synapseDynamicsTime;
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
#define spikeCount_D7 glbSpkCntD7[0]
#define spike_D7 glbSpkD7
#define glbSpkShiftD7 0

EXPORT_VAR unsigned int* glbSpkCntD7;
EXPORT_VAR unsigned int* glbSpkD7;
EXPORT_VAR uint32_t* recordSpkD7;
EXPORT_VAR scalar* VD7;
EXPORT_VAR scalar* RefracTimeD7;
#define spikeCount_EPG glbSpkCntEPG[0]
#define spike_EPG glbSpkEPG
#define glbSpkShiftEPG 0

EXPORT_VAR unsigned int* glbSpkCntEPG;
EXPORT_VAR unsigned int* glbSpkEPG;
EXPORT_VAR uint32_t* recordSpkEPG;
EXPORT_VAR float* sTEPG;
EXPORT_VAR scalar* VEPG;
EXPORT_VAR scalar* RefracTimeEPG;
#define spikeCount_PEN glbSpkCntPEN[0]
#define spike_PEN glbSpkPEN
#define glbSpkShiftPEN 0

EXPORT_VAR unsigned int* glbSpkCntPEN;
EXPORT_VAR unsigned int* glbSpkPEN;
EXPORT_VAR uint32_t* recordSpkPEN;
EXPORT_VAR scalar* VPEN;
EXPORT_VAR scalar* RefracTimePEN;
// current source variables
EXPORT_VAR scalar* magnitudePEN_input;
#define spikeCount_R glbSpkCntR[0]
#define spike_R glbSpkR
#define glbSpkShiftR 0

EXPORT_VAR unsigned int* glbSpkCntR;
EXPORT_VAR unsigned int* glbSpkR;
EXPORT_VAR uint32_t* recordSpkR;
EXPORT_VAR scalar* VR;
EXPORT_VAR scalar* RefracTimeR;
#define spikeCount_R2 glbSpkCntR2[0]
#define spike_R2 glbSpkR2
#define glbSpkShiftR2 0

EXPORT_VAR unsigned int* glbSpkCntR2;
EXPORT_VAR unsigned int* glbSpkR2;
EXPORT_VAR uint32_t* recordSpkR2;
EXPORT_VAR float* sTR2;
EXPORT_VAR scalar* VR2;
EXPORT_VAR scalar* RefracTimeR2;
// current source variables
EXPORT_VAR scalar* magnitudeR2_input;
#define spikeCount_R4 glbSpkCntR4[0]
#define spike_R4 glbSpkR4
#define glbSpkShiftR4 0

EXPORT_VAR unsigned int* glbSpkCntR4;
EXPORT_VAR unsigned int* glbSpkR4;
EXPORT_VAR uint32_t* recordSpkR4;
EXPORT_VAR float* sTR4;
EXPORT_VAR scalar* VR4;
EXPORT_VAR scalar* RefracTimeR4;
// current source variables
EXPORT_VAR scalar* magnitudeR4_input;
#define spikeCount_Stim glbSpkCntStim[0]
#define spike_Stim glbSpkStim
#define glbSpkShiftStim 0

EXPORT_VAR unsigned int* glbSpkCntStim;
EXPORT_VAR unsigned int* glbSpkStim;
EXPORT_VAR unsigned int* startSpikeStim;
EXPORT_VAR unsigned int* endSpikeStim;
EXPORT_VAR scalar* spikeTimesStim;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR float* inSynEPG_D7;
EXPORT_VAR float* inSynR4_EPG;
EXPORT_VAR float* inSynR2_EPG;
EXPORT_VAR float* inSynD7_EPG;
EXPORT_VAR float* inSynR_EPG;
EXPORT_VAR float* inSynPEN_EPG;
EXPORT_VAR float* inSynEPG_EPG;
EXPORT_VAR float* inSynStim_EPG;
EXPORT_VAR float* inSynEPG_PEN;
EXPORT_VAR float* inSynEPG_R;
EXPORT_VAR float* inSynR2_R2;
EXPORT_VAR float* inSynR4_R4;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
EXPORT_VAR scalar* gD7_EPG;
EXPORT_VAR scalar* gEPG_D7;
EXPORT_VAR scalar* gEPG_EPG;
EXPORT_VAR scalar* gEPG_PEN;
EXPORT_VAR scalar* gEPG_R;
EXPORT_VAR scalar* gPEN_EPG;
EXPORT_VAR scalar* gR2_EPG;
EXPORT_VAR scalar* gR2_R2;
EXPORT_VAR scalar* gR4_EPG;
EXPORT_VAR scalar* gR4_R4;
EXPORT_VAR scalar* gR_EPG;
EXPORT_VAR scalar* gStim_EPG;

EXPORT_FUNC void pushD7SpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullD7SpikesFromDevice();
EXPORT_FUNC void pushD7CurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullD7CurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getD7CurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getD7CurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushVD7ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVD7FromDevice();
EXPORT_FUNC void pushCurrentVD7ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVD7FromDevice();
EXPORT_FUNC scalar* getCurrentVD7(unsigned int batch = 0); 
EXPORT_FUNC void pushRefracTimeD7ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullRefracTimeD7FromDevice();
EXPORT_FUNC void pushCurrentRefracTimeD7ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentRefracTimeD7FromDevice();
EXPORT_FUNC scalar* getCurrentRefracTimeD7(unsigned int batch = 0); 
EXPORT_FUNC void pushD7StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullD7StateFromDevice();
EXPORT_FUNC void pushEPGSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEPGSpikesFromDevice();
EXPORT_FUNC void pushEPGCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEPGCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getEPGCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getEPGCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushEPGSpikeTimesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEPGSpikeTimesFromDevice();
EXPORT_FUNC void pushVEPGToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVEPGFromDevice();
EXPORT_FUNC void pushCurrentVEPGToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVEPGFromDevice();
EXPORT_FUNC scalar* getCurrentVEPG(unsigned int batch = 0); 
EXPORT_FUNC void pushRefracTimeEPGToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullRefracTimeEPGFromDevice();
EXPORT_FUNC void pushCurrentRefracTimeEPGToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentRefracTimeEPGFromDevice();
EXPORT_FUNC scalar* getCurrentRefracTimeEPG(unsigned int batch = 0); 
EXPORT_FUNC void pushEPGStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEPGStateFromDevice();
EXPORT_FUNC void pushPENSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPENSpikesFromDevice();
EXPORT_FUNC void pushPENCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPENCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getPENCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getPENCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushVPENToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVPENFromDevice();
EXPORT_FUNC void pushCurrentVPENToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVPENFromDevice();
EXPORT_FUNC scalar* getCurrentVPEN(unsigned int batch = 0); 
EXPORT_FUNC void pushRefracTimePENToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullRefracTimePENFromDevice();
EXPORT_FUNC void pushCurrentRefracTimePENToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentRefracTimePENFromDevice();
EXPORT_FUNC scalar* getCurrentRefracTimePEN(unsigned int batch = 0); 
EXPORT_FUNC void pushPENStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPENStateFromDevice();
EXPORT_FUNC void pushmagnitudePEN_inputToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullmagnitudePEN_inputFromDevice();
EXPORT_FUNC void pushPEN_inputStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPEN_inputStateFromDevice();
EXPORT_FUNC void pushRSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullRSpikesFromDevice();
EXPORT_FUNC void pushRCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullRCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getRCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getRCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushVRToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVRFromDevice();
EXPORT_FUNC void pushCurrentVRToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVRFromDevice();
EXPORT_FUNC scalar* getCurrentVR(unsigned int batch = 0); 
EXPORT_FUNC void pushRefracTimeRToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullRefracTimeRFromDevice();
EXPORT_FUNC void pushCurrentRefracTimeRToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentRefracTimeRFromDevice();
EXPORT_FUNC scalar* getCurrentRefracTimeR(unsigned int batch = 0); 
EXPORT_FUNC void pushRStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullRStateFromDevice();
EXPORT_FUNC void pushR2SpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullR2SpikesFromDevice();
EXPORT_FUNC void pushR2CurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullR2CurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getR2CurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getR2CurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushR2SpikeTimesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullR2SpikeTimesFromDevice();
EXPORT_FUNC void pushVR2ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVR2FromDevice();
EXPORT_FUNC void pushCurrentVR2ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVR2FromDevice();
EXPORT_FUNC scalar* getCurrentVR2(unsigned int batch = 0); 
EXPORT_FUNC void pushRefracTimeR2ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullRefracTimeR2FromDevice();
EXPORT_FUNC void pushCurrentRefracTimeR2ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentRefracTimeR2FromDevice();
EXPORT_FUNC scalar* getCurrentRefracTimeR2(unsigned int batch = 0); 
EXPORT_FUNC void pushR2StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullR2StateFromDevice();
EXPORT_FUNC void pushmagnitudeR2_inputToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullmagnitudeR2_inputFromDevice();
EXPORT_FUNC void pushR2_inputStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullR2_inputStateFromDevice();
EXPORT_FUNC void pushR4SpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullR4SpikesFromDevice();
EXPORT_FUNC void pushR4CurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullR4CurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getR4CurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getR4CurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushR4SpikeTimesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullR4SpikeTimesFromDevice();
EXPORT_FUNC void pushVR4ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVR4FromDevice();
EXPORT_FUNC void pushCurrentVR4ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVR4FromDevice();
EXPORT_FUNC scalar* getCurrentVR4(unsigned int batch = 0); 
EXPORT_FUNC void pushRefracTimeR4ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullRefracTimeR4FromDevice();
EXPORT_FUNC void pushCurrentRefracTimeR4ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentRefracTimeR4FromDevice();
EXPORT_FUNC scalar* getCurrentRefracTimeR4(unsigned int batch = 0); 
EXPORT_FUNC void pushR4StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullR4StateFromDevice();
EXPORT_FUNC void pushmagnitudeR4_inputToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullmagnitudeR4_inputFromDevice();
EXPORT_FUNC void pushR4_inputStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullR4_inputStateFromDevice();
EXPORT_FUNC void pushStimSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullStimSpikesFromDevice();
EXPORT_FUNC void pushStimCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullStimCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getStimCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getStimCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushstartSpikeStimToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullstartSpikeStimFromDevice();
EXPORT_FUNC void pushCurrentstartSpikeStimToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentstartSpikeStimFromDevice();
EXPORT_FUNC unsigned int* getCurrentstartSpikeStim(unsigned int batch = 0); 
EXPORT_FUNC void pushendSpikeStimToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullendSpikeStimFromDevice();
EXPORT_FUNC void pushCurrentendSpikeStimToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentendSpikeStimFromDevice();
EXPORT_FUNC unsigned int* getCurrentendSpikeStim(unsigned int batch = 0); 
EXPORT_FUNC void pushStimStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullStimStateFromDevice();
EXPORT_FUNC void allocatespikeTimesStim(unsigned int count);
EXPORT_FUNC void freespikeTimesStim();
EXPORT_FUNC void pushspikeTimesStimToDevice(unsigned int count);
EXPORT_FUNC void pullspikeTimesStimFromDevice(unsigned int count);
EXPORT_FUNC void pushgD7_EPGToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgD7_EPGFromDevice();
EXPORT_FUNC void pushinSynD7_EPGToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynD7_EPGFromDevice();
EXPORT_FUNC void pushD7_EPGStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullD7_EPGStateFromDevice();
EXPORT_FUNC void pushgEPG_D7ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgEPG_D7FromDevice();
EXPORT_FUNC void pushinSynEPG_D7ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynEPG_D7FromDevice();
EXPORT_FUNC void pushEPG_D7StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEPG_D7StateFromDevice();
EXPORT_FUNC void pushgEPG_EPGToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgEPG_EPGFromDevice();
EXPORT_FUNC void pushinSynEPG_EPGToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynEPG_EPGFromDevice();
EXPORT_FUNC void pushEPG_EPGStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEPG_EPGStateFromDevice();
EXPORT_FUNC void pushgEPG_PENToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgEPG_PENFromDevice();
EXPORT_FUNC void pushinSynEPG_PENToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynEPG_PENFromDevice();
EXPORT_FUNC void pushEPG_PENStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEPG_PENStateFromDevice();
EXPORT_FUNC void pushgEPG_RToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgEPG_RFromDevice();
EXPORT_FUNC void pushinSynEPG_RToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynEPG_RFromDevice();
EXPORT_FUNC void pushEPG_RStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEPG_RStateFromDevice();
EXPORT_FUNC void pushgPEN_EPGToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgPEN_EPGFromDevice();
EXPORT_FUNC void pushinSynPEN_EPGToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynPEN_EPGFromDevice();
EXPORT_FUNC void pushPEN_EPGStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullPEN_EPGStateFromDevice();
EXPORT_FUNC void pushgR2_EPGToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgR2_EPGFromDevice();
EXPORT_FUNC void pushinSynR2_EPGToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynR2_EPGFromDevice();
EXPORT_FUNC void pushR2_EPGStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullR2_EPGStateFromDevice();
EXPORT_FUNC void pushgR2_R2ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgR2_R2FromDevice();
EXPORT_FUNC void pushinSynR2_R2ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynR2_R2FromDevice();
EXPORT_FUNC void pushR2_R2StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullR2_R2StateFromDevice();
EXPORT_FUNC void pushgR4_EPGToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgR4_EPGFromDevice();
EXPORT_FUNC void pushinSynR4_EPGToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynR4_EPGFromDevice();
EXPORT_FUNC void pushR4_EPGStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullR4_EPGStateFromDevice();
EXPORT_FUNC void pushgR4_R4ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgR4_R4FromDevice();
EXPORT_FUNC void pushinSynR4_R4ToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynR4_R4FromDevice();
EXPORT_FUNC void pushR4_R4StateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullR4_R4StateFromDevice();
EXPORT_FUNC void pushgR_EPGToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgR_EPGFromDevice();
EXPORT_FUNC void pushinSynR_EPGToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynR_EPGFromDevice();
EXPORT_FUNC void pushR_EPGStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullR_EPGStateFromDevice();
EXPORT_FUNC void pushgStim_EPGToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgStim_EPGFromDevice();
EXPORT_FUNC void pushinSynStim_EPGToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynStim_EPGFromDevice();
EXPORT_FUNC void pushStim_EPGStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullStim_EPGStateFromDevice();
// Runner functions
EXPORT_FUNC void copyStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyStateFromDevice();
EXPORT_FUNC void copyCurrentSpikesFromDevice();
EXPORT_FUNC void copyCurrentSpikeEventsFromDevice();
EXPORT_FUNC void allocateRecordingBuffers(unsigned int timesteps);
EXPORT_FUNC void pullRecordingBuffersFromDevice();
EXPORT_FUNC void allocateMem();
EXPORT_FUNC void freeMem();
EXPORT_FUNC size_t getFreeDeviceMemBytes();
EXPORT_FUNC void stepTime();

// Functions generated by backend
EXPORT_FUNC void updateNeurons(float t, unsigned int recordingTimestep); 
EXPORT_FUNC void updateSynapses(float t);
EXPORT_FUNC void initialize();
EXPORT_FUNC void initializeSparse();
}  // extern "C"
