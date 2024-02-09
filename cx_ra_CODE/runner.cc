#include "definitionsInternal.h"


extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
unsigned long long iT;
float t;
unsigned long long numRecordingTimesteps = 0;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
double initTime = 0.0;
double initSparseTime = 0.0;
double neuronUpdateTime = 0.0;
double presynapticUpdateTime = 0.0;
double postsynapticUpdateTime = 0.0;
double synapseDynamicsTime = 0.0;
// ------------------------------------------------------------------------
// merged group arrays
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
unsigned int* glbSpkCntD7;
unsigned int* glbSpkD7;
uint32_t* recordSpkD7;
scalar* VD7;
scalar* RefracTimeD7;
unsigned int* glbSpkCntEPG;
unsigned int* glbSpkEPG;
uint32_t* recordSpkEPG;
float* sTEPG;
scalar* VEPG;
scalar* RefracTimeEPG;
unsigned int* glbSpkCntPEN;
unsigned int* glbSpkPEN;
uint32_t* recordSpkPEN;
scalar* VPEN;
scalar* RefracTimePEN;
// current source variables
scalar* magnitudePEN_input;
unsigned int* glbSpkCntR;
unsigned int* glbSpkR;
uint32_t* recordSpkR;
scalar* VR;
scalar* RefracTimeR;
unsigned int* glbSpkCntR2;
unsigned int* glbSpkR2;
uint32_t* recordSpkR2;
float* sTR2;
scalar* VR2;
scalar* RefracTimeR2;
// current source variables
scalar* magnitudeR2_input;
unsigned int* glbSpkCntR4;
unsigned int* glbSpkR4;
uint32_t* recordSpkR4;
float* sTR4;
scalar* VR4;
scalar* RefracTimeR4;
// current source variables
scalar* magnitudeR4_input;
unsigned int* glbSpkCntStim;
unsigned int* glbSpkStim;
unsigned int* startSpikeStim;
unsigned int* endSpikeStim;
scalar* spikeTimesStim;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------
float* inSynEPG_D7;
float* inSynR4_EPG;
float* inSynR2_EPG;
float* inSynD7_EPG;
float* inSynR_EPG;
float* inSynPEN_EPG;
float* inSynEPG_EPG;
float* inSynStim_EPG;
float* inSynEPG_PEN;
float* inSynEPG_R;
float* inSynR2_R2;
float* inSynR4_R4;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
scalar* gD7_EPG;
scalar* gEPG_D7;
scalar* gEPG_EPG;
scalar* gEPG_PEN;
scalar* gEPG_R;
scalar* gPEN_EPG;
scalar* gR2_EPG;
scalar* gR2_R2;
scalar* gR4_EPG;
scalar* gR4_R4;
scalar* gR_EPG;
scalar* gStim_EPG;

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------
void allocatespikeTimesStim(unsigned int count) {
    spikeTimesStim = new scalar[count];
    pushMergedNeuronUpdate0spikeTimesToDevice(0, spikeTimesStim);
}
void freespikeTimesStim() {
    delete[] spikeTimesStim;
}
void pushspikeTimesStimToDevice(unsigned int count) {
}
void pullspikeTimesStimFromDevice(unsigned int count) {
}

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
void pushD7SpikesToDevice(bool uninitialisedOnly) {
}

void pushD7CurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushVD7ToDevice(bool uninitialisedOnly) {
}

void pushCurrentVD7ToDevice(bool uninitialisedOnly) {
}

void pushRefracTimeD7ToDevice(bool uninitialisedOnly) {
}

void pushCurrentRefracTimeD7ToDevice(bool uninitialisedOnly) {
}

void pushD7StateToDevice(bool uninitialisedOnly) {
    pushVD7ToDevice(uninitialisedOnly);
    pushRefracTimeD7ToDevice(uninitialisedOnly);
}

void pushEPGSpikesToDevice(bool uninitialisedOnly) {
}

void pushEPGCurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushEPGSpikeTimesToDevice(bool uninitialisedOnly) {
}

void pushVEPGToDevice(bool uninitialisedOnly) {
}

void pushCurrentVEPGToDevice(bool uninitialisedOnly) {
}

void pushRefracTimeEPGToDevice(bool uninitialisedOnly) {
}

void pushCurrentRefracTimeEPGToDevice(bool uninitialisedOnly) {
}

void pushEPGStateToDevice(bool uninitialisedOnly) {
    pushVEPGToDevice(uninitialisedOnly);
    pushRefracTimeEPGToDevice(uninitialisedOnly);
}

void pushPENSpikesToDevice(bool uninitialisedOnly) {
}

void pushPENCurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushVPENToDevice(bool uninitialisedOnly) {
}

void pushCurrentVPENToDevice(bool uninitialisedOnly) {
}

void pushRefracTimePENToDevice(bool uninitialisedOnly) {
}

void pushCurrentRefracTimePENToDevice(bool uninitialisedOnly) {
}

void pushPENStateToDevice(bool uninitialisedOnly) {
    pushVPENToDevice(uninitialisedOnly);
    pushRefracTimePENToDevice(uninitialisedOnly);
}

void pushmagnitudePEN_inputToDevice(bool uninitialisedOnly) {
}

void pushPEN_inputStateToDevice(bool uninitialisedOnly) {
    pushmagnitudePEN_inputToDevice(uninitialisedOnly);
}

void pushRSpikesToDevice(bool uninitialisedOnly) {
}

void pushRCurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushVRToDevice(bool uninitialisedOnly) {
}

void pushCurrentVRToDevice(bool uninitialisedOnly) {
}

void pushRefracTimeRToDevice(bool uninitialisedOnly) {
}

void pushCurrentRefracTimeRToDevice(bool uninitialisedOnly) {
}

void pushRStateToDevice(bool uninitialisedOnly) {
    pushVRToDevice(uninitialisedOnly);
    pushRefracTimeRToDevice(uninitialisedOnly);
}

void pushR2SpikesToDevice(bool uninitialisedOnly) {
}

void pushR2CurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushR2SpikeTimesToDevice(bool uninitialisedOnly) {
}

void pushVR2ToDevice(bool uninitialisedOnly) {
}

void pushCurrentVR2ToDevice(bool uninitialisedOnly) {
}

void pushRefracTimeR2ToDevice(bool uninitialisedOnly) {
}

void pushCurrentRefracTimeR2ToDevice(bool uninitialisedOnly) {
}

void pushR2StateToDevice(bool uninitialisedOnly) {
    pushVR2ToDevice(uninitialisedOnly);
    pushRefracTimeR2ToDevice(uninitialisedOnly);
}

void pushmagnitudeR2_inputToDevice(bool uninitialisedOnly) {
}

void pushR2_inputStateToDevice(bool uninitialisedOnly) {
    pushmagnitudeR2_inputToDevice(uninitialisedOnly);
}

void pushR4SpikesToDevice(bool uninitialisedOnly) {
}

void pushR4CurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushR4SpikeTimesToDevice(bool uninitialisedOnly) {
}

void pushVR4ToDevice(bool uninitialisedOnly) {
}

void pushCurrentVR4ToDevice(bool uninitialisedOnly) {
}

void pushRefracTimeR4ToDevice(bool uninitialisedOnly) {
}

void pushCurrentRefracTimeR4ToDevice(bool uninitialisedOnly) {
}

void pushR4StateToDevice(bool uninitialisedOnly) {
    pushVR4ToDevice(uninitialisedOnly);
    pushRefracTimeR4ToDevice(uninitialisedOnly);
}

void pushmagnitudeR4_inputToDevice(bool uninitialisedOnly) {
}

void pushR4_inputStateToDevice(bool uninitialisedOnly) {
    pushmagnitudeR4_inputToDevice(uninitialisedOnly);
}

void pushStimSpikesToDevice(bool uninitialisedOnly) {
}

void pushStimCurrentSpikesToDevice(bool uninitialisedOnly) {
}

void pushstartSpikeStimToDevice(bool uninitialisedOnly) {
}

void pushCurrentstartSpikeStimToDevice(bool uninitialisedOnly) {
}

void pushendSpikeStimToDevice(bool uninitialisedOnly) {
}

void pushCurrentendSpikeStimToDevice(bool uninitialisedOnly) {
}

void pushStimStateToDevice(bool uninitialisedOnly) {
    pushstartSpikeStimToDevice(uninitialisedOnly);
    pushendSpikeStimToDevice(uninitialisedOnly);
}

void pushgD7_EPGToDevice(bool uninitialisedOnly) {
}

void pushinSynD7_EPGToDevice(bool uninitialisedOnly) {
}

void pushD7_EPGStateToDevice(bool uninitialisedOnly) {
    pushgD7_EPGToDevice(uninitialisedOnly);
    pushinSynD7_EPGToDevice(uninitialisedOnly);
}

void pushgEPG_D7ToDevice(bool uninitialisedOnly) {
}

void pushinSynEPG_D7ToDevice(bool uninitialisedOnly) {
}

void pushEPG_D7StateToDevice(bool uninitialisedOnly) {
    pushgEPG_D7ToDevice(uninitialisedOnly);
    pushinSynEPG_D7ToDevice(uninitialisedOnly);
}

void pushgEPG_EPGToDevice(bool uninitialisedOnly) {
}

void pushinSynEPG_EPGToDevice(bool uninitialisedOnly) {
}

void pushEPG_EPGStateToDevice(bool uninitialisedOnly) {
    pushgEPG_EPGToDevice(uninitialisedOnly);
    pushinSynEPG_EPGToDevice(uninitialisedOnly);
}

void pushgEPG_PENToDevice(bool uninitialisedOnly) {
}

void pushinSynEPG_PENToDevice(bool uninitialisedOnly) {
}

void pushEPG_PENStateToDevice(bool uninitialisedOnly) {
    pushgEPG_PENToDevice(uninitialisedOnly);
    pushinSynEPG_PENToDevice(uninitialisedOnly);
}

void pushgEPG_RToDevice(bool uninitialisedOnly) {
}

void pushinSynEPG_RToDevice(bool uninitialisedOnly) {
}

void pushEPG_RStateToDevice(bool uninitialisedOnly) {
    pushgEPG_RToDevice(uninitialisedOnly);
    pushinSynEPG_RToDevice(uninitialisedOnly);
}

void pushgPEN_EPGToDevice(bool uninitialisedOnly) {
}

void pushinSynPEN_EPGToDevice(bool uninitialisedOnly) {
}

void pushPEN_EPGStateToDevice(bool uninitialisedOnly) {
    pushgPEN_EPGToDevice(uninitialisedOnly);
    pushinSynPEN_EPGToDevice(uninitialisedOnly);
}

void pushgR2_EPGToDevice(bool uninitialisedOnly) {
}

void pushinSynR2_EPGToDevice(bool uninitialisedOnly) {
}

void pushR2_EPGStateToDevice(bool uninitialisedOnly) {
    pushgR2_EPGToDevice(uninitialisedOnly);
    pushinSynR2_EPGToDevice(uninitialisedOnly);
}

void pushgR2_R2ToDevice(bool uninitialisedOnly) {
}

void pushinSynR2_R2ToDevice(bool uninitialisedOnly) {
}

void pushR2_R2StateToDevice(bool uninitialisedOnly) {
    pushgR2_R2ToDevice(uninitialisedOnly);
    pushinSynR2_R2ToDevice(uninitialisedOnly);
}

void pushgR4_EPGToDevice(bool uninitialisedOnly) {
}

void pushinSynR4_EPGToDevice(bool uninitialisedOnly) {
}

void pushR4_EPGStateToDevice(bool uninitialisedOnly) {
    pushgR4_EPGToDevice(uninitialisedOnly);
    pushinSynR4_EPGToDevice(uninitialisedOnly);
}

void pushgR4_R4ToDevice(bool uninitialisedOnly) {
}

void pushinSynR4_R4ToDevice(bool uninitialisedOnly) {
}

void pushR4_R4StateToDevice(bool uninitialisedOnly) {
    pushgR4_R4ToDevice(uninitialisedOnly);
    pushinSynR4_R4ToDevice(uninitialisedOnly);
}

void pushgR_EPGToDevice(bool uninitialisedOnly) {
}

void pushinSynR_EPGToDevice(bool uninitialisedOnly) {
}

void pushR_EPGStateToDevice(bool uninitialisedOnly) {
    pushgR_EPGToDevice(uninitialisedOnly);
    pushinSynR_EPGToDevice(uninitialisedOnly);
}

void pushgStim_EPGToDevice(bool uninitialisedOnly) {
}

void pushinSynStim_EPGToDevice(bool uninitialisedOnly) {
}

void pushStim_EPGStateToDevice(bool uninitialisedOnly) {
    pushgStim_EPGToDevice(uninitialisedOnly);
    pushinSynStim_EPGToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullD7SpikesFromDevice() {
}

void pullD7CurrentSpikesFromDevice() {
}

void pullVD7FromDevice() {
}

void pullCurrentVD7FromDevice() {
}

void pullRefracTimeD7FromDevice() {
}

void pullCurrentRefracTimeD7FromDevice() {
}

void pullD7StateFromDevice() {
    pullVD7FromDevice();
    pullRefracTimeD7FromDevice();
}

void pullEPGSpikesFromDevice() {
}

void pullEPGCurrentSpikesFromDevice() {
}

void pullEPGSpikeTimesFromDevice() {
}

void pullVEPGFromDevice() {
}

void pullCurrentVEPGFromDevice() {
}

void pullRefracTimeEPGFromDevice() {
}

void pullCurrentRefracTimeEPGFromDevice() {
}

void pullEPGStateFromDevice() {
    pullVEPGFromDevice();
    pullRefracTimeEPGFromDevice();
}

void pullPENSpikesFromDevice() {
}

void pullPENCurrentSpikesFromDevice() {
}

void pullVPENFromDevice() {
}

void pullCurrentVPENFromDevice() {
}

void pullRefracTimePENFromDevice() {
}

void pullCurrentRefracTimePENFromDevice() {
}

void pullPENStateFromDevice() {
    pullVPENFromDevice();
    pullRefracTimePENFromDevice();
}

void pullmagnitudePEN_inputFromDevice() {
}

void pullPEN_inputStateFromDevice() {
    pullmagnitudePEN_inputFromDevice();
}

void pullRSpikesFromDevice() {
}

void pullRCurrentSpikesFromDevice() {
}

void pullVRFromDevice() {
}

void pullCurrentVRFromDevice() {
}

void pullRefracTimeRFromDevice() {
}

void pullCurrentRefracTimeRFromDevice() {
}

void pullRStateFromDevice() {
    pullVRFromDevice();
    pullRefracTimeRFromDevice();
}

void pullR2SpikesFromDevice() {
}

void pullR2CurrentSpikesFromDevice() {
}

void pullR2SpikeTimesFromDevice() {
}

void pullVR2FromDevice() {
}

void pullCurrentVR2FromDevice() {
}

void pullRefracTimeR2FromDevice() {
}

void pullCurrentRefracTimeR2FromDevice() {
}

void pullR2StateFromDevice() {
    pullVR2FromDevice();
    pullRefracTimeR2FromDevice();
}

void pullmagnitudeR2_inputFromDevice() {
}

void pullR2_inputStateFromDevice() {
    pullmagnitudeR2_inputFromDevice();
}

void pullR4SpikesFromDevice() {
}

void pullR4CurrentSpikesFromDevice() {
}

void pullR4SpikeTimesFromDevice() {
}

void pullVR4FromDevice() {
}

void pullCurrentVR4FromDevice() {
}

void pullRefracTimeR4FromDevice() {
}

void pullCurrentRefracTimeR4FromDevice() {
}

void pullR4StateFromDevice() {
    pullVR4FromDevice();
    pullRefracTimeR4FromDevice();
}

void pullmagnitudeR4_inputFromDevice() {
}

void pullR4_inputStateFromDevice() {
    pullmagnitudeR4_inputFromDevice();
}

void pullStimSpikesFromDevice() {
}

void pullStimCurrentSpikesFromDevice() {
}

void pullstartSpikeStimFromDevice() {
}

void pullCurrentstartSpikeStimFromDevice() {
}

void pullendSpikeStimFromDevice() {
}

void pullCurrentendSpikeStimFromDevice() {
}

void pullStimStateFromDevice() {
    pullstartSpikeStimFromDevice();
    pullendSpikeStimFromDevice();
}

void pullgD7_EPGFromDevice() {
}

void pullinSynD7_EPGFromDevice() {
}

void pullD7_EPGStateFromDevice() {
    pullgD7_EPGFromDevice();
    pullinSynD7_EPGFromDevice();
}

void pullgEPG_D7FromDevice() {
}

void pullinSynEPG_D7FromDevice() {
}

void pullEPG_D7StateFromDevice() {
    pullgEPG_D7FromDevice();
    pullinSynEPG_D7FromDevice();
}

void pullgEPG_EPGFromDevice() {
}

void pullinSynEPG_EPGFromDevice() {
}

void pullEPG_EPGStateFromDevice() {
    pullgEPG_EPGFromDevice();
    pullinSynEPG_EPGFromDevice();
}

void pullgEPG_PENFromDevice() {
}

void pullinSynEPG_PENFromDevice() {
}

void pullEPG_PENStateFromDevice() {
    pullgEPG_PENFromDevice();
    pullinSynEPG_PENFromDevice();
}

void pullgEPG_RFromDevice() {
}

void pullinSynEPG_RFromDevice() {
}

void pullEPG_RStateFromDevice() {
    pullgEPG_RFromDevice();
    pullinSynEPG_RFromDevice();
}

void pullgPEN_EPGFromDevice() {
}

void pullinSynPEN_EPGFromDevice() {
}

void pullPEN_EPGStateFromDevice() {
    pullgPEN_EPGFromDevice();
    pullinSynPEN_EPGFromDevice();
}

void pullgR2_EPGFromDevice() {
}

void pullinSynR2_EPGFromDevice() {
}

void pullR2_EPGStateFromDevice() {
    pullgR2_EPGFromDevice();
    pullinSynR2_EPGFromDevice();
}

void pullgR2_R2FromDevice() {
}

void pullinSynR2_R2FromDevice() {
}

void pullR2_R2StateFromDevice() {
    pullgR2_R2FromDevice();
    pullinSynR2_R2FromDevice();
}

void pullgR4_EPGFromDevice() {
}

void pullinSynR4_EPGFromDevice() {
}

void pullR4_EPGStateFromDevice() {
    pullgR4_EPGFromDevice();
    pullinSynR4_EPGFromDevice();
}

void pullgR4_R4FromDevice() {
}

void pullinSynR4_R4FromDevice() {
}

void pullR4_R4StateFromDevice() {
    pullgR4_R4FromDevice();
    pullinSynR4_R4FromDevice();
}

void pullgR_EPGFromDevice() {
}

void pullinSynR_EPGFromDevice() {
}

void pullR_EPGStateFromDevice() {
    pullgR_EPGFromDevice();
    pullinSynR_EPGFromDevice();
}

void pullgStim_EPGFromDevice() {
}

void pullinSynStim_EPGFromDevice() {
}

void pullStim_EPGStateFromDevice() {
    pullgStim_EPGFromDevice();
    pullinSynStim_EPGFromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getD7CurrentSpikes(unsigned int batch) {
    return (glbSpkD7);
}

unsigned int& getD7CurrentSpikeCount(unsigned int batch) {
    return glbSpkCntD7[0];
}

scalar* getCurrentVD7(unsigned int batch) {
    return VD7;
}

scalar* getCurrentRefracTimeD7(unsigned int batch) {
    return RefracTimeD7;
}

unsigned int* getEPGCurrentSpikes(unsigned int batch) {
    return (glbSpkEPG);
}

unsigned int& getEPGCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntEPG[0];
}

scalar* getCurrentVEPG(unsigned int batch) {
    return VEPG;
}

scalar* getCurrentRefracTimeEPG(unsigned int batch) {
    return RefracTimeEPG;
}

unsigned int* getPENCurrentSpikes(unsigned int batch) {
    return (glbSpkPEN);
}

unsigned int& getPENCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntPEN[0];
}

scalar* getCurrentVPEN(unsigned int batch) {
    return VPEN;
}

scalar* getCurrentRefracTimePEN(unsigned int batch) {
    return RefracTimePEN;
}

unsigned int* getRCurrentSpikes(unsigned int batch) {
    return (glbSpkR);
}

unsigned int& getRCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntR[0];
}

scalar* getCurrentVR(unsigned int batch) {
    return VR;
}

scalar* getCurrentRefracTimeR(unsigned int batch) {
    return RefracTimeR;
}

unsigned int* getR2CurrentSpikes(unsigned int batch) {
    return (glbSpkR2);
}

unsigned int& getR2CurrentSpikeCount(unsigned int batch) {
    return glbSpkCntR2[0];
}

scalar* getCurrentVR2(unsigned int batch) {
    return VR2;
}

scalar* getCurrentRefracTimeR2(unsigned int batch) {
    return RefracTimeR2;
}

unsigned int* getR4CurrentSpikes(unsigned int batch) {
    return (glbSpkR4);
}

unsigned int& getR4CurrentSpikeCount(unsigned int batch) {
    return glbSpkCntR4[0];
}

scalar* getCurrentVR4(unsigned int batch) {
    return VR4;
}

scalar* getCurrentRefracTimeR4(unsigned int batch) {
    return RefracTimeR4;
}

unsigned int* getStimCurrentSpikes(unsigned int batch) {
    return (glbSpkStim);
}

unsigned int& getStimCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntStim[0];
}

unsigned int* getCurrentstartSpikeStim(unsigned int batch) {
    return startSpikeStim;
}

unsigned int* getCurrentendSpikeStim(unsigned int batch) {
    return endSpikeStim;
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushD7StateToDevice(uninitialisedOnly);
    pushEPGStateToDevice(uninitialisedOnly);
    pushPENStateToDevice(uninitialisedOnly);
    pushPEN_inputStateToDevice(uninitialisedOnly);
    pushRStateToDevice(uninitialisedOnly);
    pushR2StateToDevice(uninitialisedOnly);
    pushR2_inputStateToDevice(uninitialisedOnly);
    pushR4StateToDevice(uninitialisedOnly);
    pushR4_inputStateToDevice(uninitialisedOnly);
    pushStimStateToDevice(uninitialisedOnly);
    pushD7_EPGStateToDevice(uninitialisedOnly);
    pushEPG_D7StateToDevice(uninitialisedOnly);
    pushEPG_EPGStateToDevice(uninitialisedOnly);
    pushEPG_PENStateToDevice(uninitialisedOnly);
    pushEPG_RStateToDevice(uninitialisedOnly);
    pushPEN_EPGStateToDevice(uninitialisedOnly);
    pushR2_EPGStateToDevice(uninitialisedOnly);
    pushR2_R2StateToDevice(uninitialisedOnly);
    pushR4_EPGStateToDevice(uninitialisedOnly);
    pushR4_R4StateToDevice(uninitialisedOnly);
    pushR_EPGStateToDevice(uninitialisedOnly);
    pushStim_EPGStateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
}

void copyStateFromDevice() {
    pullD7StateFromDevice();
    pullEPGStateFromDevice();
    pullPENStateFromDevice();
    pullPEN_inputStateFromDevice();
    pullRStateFromDevice();
    pullR2StateFromDevice();
    pullR2_inputStateFromDevice();
    pullR4StateFromDevice();
    pullR4_inputStateFromDevice();
    pullStimStateFromDevice();
    pullD7_EPGStateFromDevice();
    pullEPG_D7StateFromDevice();
    pullEPG_EPGStateFromDevice();
    pullEPG_PENStateFromDevice();
    pullEPG_RStateFromDevice();
    pullPEN_EPGStateFromDevice();
    pullR2_EPGStateFromDevice();
    pullR2_R2StateFromDevice();
    pullR4_EPGStateFromDevice();
    pullR4_R4StateFromDevice();
    pullR_EPGStateFromDevice();
    pullStim_EPGStateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pullD7CurrentSpikesFromDevice();
    pullEPGCurrentSpikesFromDevice();
    pullPENCurrentSpikesFromDevice();
    pullRCurrentSpikesFromDevice();
    pullR2CurrentSpikesFromDevice();
    pullR4CurrentSpikesFromDevice();
    pullStimCurrentSpikesFromDevice();
}

void copyCurrentSpikeEventsFromDevice() {
}

void allocateRecordingBuffers(unsigned int timesteps) {
    numRecordingTimesteps = timesteps;
     {
        const unsigned int numWords = 1 * timesteps;
         {
            recordSpkD7 = new uint32_t[numWords];
            pushMergedNeuronUpdate4recordSpkToDevice(0, recordSpkD7);
        }
    }
     {
        const unsigned int numWords = 1 * timesteps;
         {
            recordSpkEPG = new uint32_t[numWords];
            pushMergedNeuronUpdate3recordSpkToDevice(0, recordSpkEPG);
        }
    }
     {
        const unsigned int numWords = 1 * timesteps;
         {
            recordSpkPEN = new uint32_t[numWords];
            pushMergedNeuronUpdate2recordSpkToDevice(0, recordSpkPEN);
        }
    }
     {
        const unsigned int numWords = 1 * timesteps;
         {
            recordSpkR = new uint32_t[numWords];
            pushMergedNeuronUpdate4recordSpkToDevice(1, recordSpkR);
        }
    }
     {
        const unsigned int numWords = 2 * timesteps;
         {
            recordSpkR2 = new uint32_t[numWords];
            pushMergedNeuronUpdate1recordSpkToDevice(0, recordSpkR2);
        }
    }
     {
        const unsigned int numWords = 1 * timesteps;
         {
            recordSpkR4 = new uint32_t[numWords];
            pushMergedNeuronUpdate1recordSpkToDevice(1, recordSpkR4);
        }
    }
     {
    }
}

void pullRecordingBuffersFromDevice() {
    if(numRecordingTimesteps == 0) {
        throw std::runtime_error("Recording buffer not allocated - cannot pull from device");
    }
     {
        const unsigned int numWords = 1 * numRecordingTimesteps;
         {
        }
    }
     {
        const unsigned int numWords = 1 * numRecordingTimesteps;
         {
        }
    }
     {
        const unsigned int numWords = 1 * numRecordingTimesteps;
         {
        }
    }
     {
        const unsigned int numWords = 1 * numRecordingTimesteps;
         {
        }
    }
     {
        const unsigned int numWords = 2 * numRecordingTimesteps;
         {
        }
    }
     {
        const unsigned int numWords = 1 * numRecordingTimesteps;
         {
        }
    }
     {
    }
}

void allocateMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    glbSpkCntD7 = new unsigned int[1];
    glbSpkD7 = new unsigned int[8];
    VD7 = new scalar[8];
    RefracTimeD7 = new scalar[8];
    glbSpkCntEPG = new unsigned int[1];
    glbSpkEPG = new unsigned int[16];
    sTEPG = new float[16];
    VEPG = new scalar[16];
    RefracTimeEPG = new scalar[16];
    glbSpkCntPEN = new unsigned int[1];
    glbSpkPEN = new unsigned int[16];
    VPEN = new scalar[16];
    RefracTimePEN = new scalar[16];
    // current source variables
    magnitudePEN_input = new scalar[16];
    glbSpkCntR = new unsigned int[1];
    glbSpkR = new unsigned int[1];
    VR = new scalar[1];
    RefracTimeR = new scalar[1];
    glbSpkCntR2 = new unsigned int[1];
    glbSpkR2 = new unsigned int[42];
    sTR2 = new float[42];
    VR2 = new scalar[42];
    RefracTimeR2 = new scalar[42];
    // current source variables
    magnitudeR2_input = new scalar[42];
    glbSpkCntR4 = new unsigned int[1];
    glbSpkR4 = new unsigned int[26];
    sTR4 = new float[26];
    VR4 = new scalar[26];
    RefracTimeR4 = new scalar[26];
    // current source variables
    magnitudeR4_input = new scalar[26];
    glbSpkCntStim = new unsigned int[1];
    glbSpkStim = new unsigned int[1];
    startSpikeStim = new unsigned int[1];
    endSpikeStim = new unsigned int[1];
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    inSynEPG_D7 = new float[8];
    inSynR4_EPG = new float[16];
    inSynR2_EPG = new float[16];
    inSynD7_EPG = new float[16];
    inSynR_EPG = new float[16];
    inSynPEN_EPG = new float[16];
    inSynEPG_EPG = new float[16];
    inSynStim_EPG = new float[16];
    inSynEPG_PEN = new float[16];
    inSynEPG_R = new float[1];
    inSynR2_R2 = new float[42];
    inSynR4_R4 = new float[26];
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    gD7_EPG = new scalar[128];
    gEPG_D7 = new scalar[128];
    gEPG_EPG = new scalar[256];
    gEPG_PEN = new scalar[256];
    gEPG_R = new scalar[16];
    gPEN_EPG = new scalar[256];
    gR2_EPG = new scalar[672];
    gR2_R2 = new scalar[1764];
    gR4_EPG = new scalar[416];
    gR4_R4 = new scalar[676];
    gR_EPG = new scalar[16];
    gStim_EPG = new scalar[16];
    
    pushMergedNeuronInitGroup0ToDevice(0, glbSpkCntStim, glbSpkStim, 1);
    pushMergedNeuronInitGroup1ToDevice(0, glbSpkCntR2, glbSpkR2, sTR2, VR2, RefracTimeR2, inSynR2_R2, magnitudeR2_input, 42);
    pushMergedNeuronInitGroup1ToDevice(1, glbSpkCntR4, glbSpkR4, sTR4, VR4, RefracTimeR4, inSynR4_R4, magnitudeR4_input, 26);
    pushMergedNeuronInitGroup2ToDevice(0, glbSpkCntPEN, glbSpkPEN, VPEN, RefracTimePEN, inSynEPG_PEN, magnitudePEN_input, 16);
    pushMergedNeuronInitGroup3ToDevice(0, glbSpkCntEPG, glbSpkEPG, sTEPG, VEPG, RefracTimeEPG, inSynR4_EPG, inSynR2_EPG, inSynD7_EPG, inSynR_EPG, inSynPEN_EPG, inSynEPG_EPG, inSynStim_EPG, 16);
    pushMergedNeuronInitGroup4ToDevice(0, glbSpkCntD7, glbSpkD7, VD7, RefracTimeD7, inSynEPG_D7, 8);
    pushMergedNeuronInitGroup4ToDevice(1, glbSpkCntR, glbSpkR, VR, RefracTimeR, inSynEPG_R, 1);
    pushMergedNeuronUpdateGroup0ToDevice(0, glbSpkCntStim, glbSpkStim, startSpikeStim, endSpikeStim, spikeTimesStim, 1);
    pushMergedNeuronUpdateGroup1ToDevice(0, glbSpkCntR2, glbSpkR2, sTR2, VR2, RefracTimeR2, inSynR2_R2, magnitudeR2_input, recordSpkR2, 42);
    pushMergedNeuronUpdateGroup1ToDevice(1, glbSpkCntR4, glbSpkR4, sTR4, VR4, RefracTimeR4, inSynR4_R4, magnitudeR4_input, recordSpkR4, 26);
    pushMergedNeuronUpdateGroup2ToDevice(0, glbSpkCntPEN, glbSpkPEN, VPEN, RefracTimePEN, inSynEPG_PEN, magnitudePEN_input, recordSpkPEN, 16);
    pushMergedNeuronUpdateGroup3ToDevice(0, glbSpkCntEPG, glbSpkEPG, sTEPG, VEPG, RefracTimeEPG, inSynR4_EPG, inSynR2_EPG, inSynD7_EPG, inSynR_EPG, inSynPEN_EPG, inSynEPG_EPG, inSynStim_EPG, recordSpkEPG, 16);
    pushMergedNeuronUpdateGroup4ToDevice(0, glbSpkCntD7, glbSpkD7, VD7, RefracTimeD7, inSynEPG_D7, recordSpkD7, 8);
    pushMergedNeuronUpdateGroup4ToDevice(1, glbSpkCntR, glbSpkR, VR, RefracTimeR, inSynEPG_R, recordSpkR, 1);
    pushMergedPresynapticUpdateGroup0ToDevice(0, inSynR2_EPG, glbSpkCntR2, glbSpkR2, sTR2, sTEPG, gR2_EPG, 42, 16, 16);
    pushMergedPresynapticUpdateGroup0ToDevice(1, inSynR4_EPG, glbSpkCntR4, glbSpkR4, sTR4, sTEPG, gR4_EPG, 26, 16, 16);
    pushMergedPresynapticUpdateGroup1ToDevice(0, inSynD7_EPG, glbSpkCntD7, glbSpkD7, gD7_EPG, 8, 16, 16);
    pushMergedPresynapticUpdateGroup1ToDevice(1, inSynEPG_D7, glbSpkCntEPG, glbSpkEPG, gEPG_D7, 16, 8, 8);
    pushMergedPresynapticUpdateGroup1ToDevice(2, inSynEPG_EPG, glbSpkCntEPG, glbSpkEPG, gEPG_EPG, 16, 16, 16);
    pushMergedPresynapticUpdateGroup1ToDevice(3, inSynEPG_PEN, glbSpkCntEPG, glbSpkEPG, gEPG_PEN, 16, 16, 16);
    pushMergedPresynapticUpdateGroup1ToDevice(4, inSynEPG_R, glbSpkCntEPG, glbSpkEPG, gEPG_R, 16, 1, 1);
    pushMergedPresynapticUpdateGroup1ToDevice(5, inSynPEN_EPG, glbSpkCntPEN, glbSpkPEN, gPEN_EPG, 16, 16, 16);
    pushMergedPresynapticUpdateGroup1ToDevice(6, inSynR2_R2, glbSpkCntR2, glbSpkR2, gR2_R2, 42, 42, 42);
    pushMergedPresynapticUpdateGroup1ToDevice(7, inSynR4_R4, glbSpkCntR4, glbSpkR4, gR4_R4, 26, 26, 26);
    pushMergedPresynapticUpdateGroup1ToDevice(8, inSynR_EPG, glbSpkCntR, glbSpkR, gR_EPG, 1, 16, 16);
    pushMergedPresynapticUpdateGroup1ToDevice(9, inSynStim_EPG, glbSpkCntStim, glbSpkStim, gStim_EPG, 1, 16, 16);
    pushMergedPostsynapticUpdateGroup0ToDevice(0, glbSpkCntEPG, glbSpkEPG, sTR2, sTEPG, gR2_EPG, 42, 16, 16, 42);
    pushMergedPostsynapticUpdateGroup0ToDevice(1, glbSpkCntEPG, glbSpkEPG, sTR4, sTEPG, gR4_EPG, 26, 16, 16, 26);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(0, glbSpkCntD7);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(1, glbSpkCntEPG);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(2, glbSpkCntPEN);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(3, glbSpkCntR);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(4, glbSpkCntR2);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(5, glbSpkCntR4);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(6, glbSpkCntStim);
}

void freeMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    delete[] glbSpkCntD7;
    delete[] glbSpkD7;
    delete[] recordSpkD7;
    delete[] VD7;
    delete[] RefracTimeD7;
    delete[] glbSpkCntEPG;
    delete[] glbSpkEPG;
    delete[] recordSpkEPG;
    delete[] sTEPG;
    delete[] VEPG;
    delete[] RefracTimeEPG;
    delete[] glbSpkCntPEN;
    delete[] glbSpkPEN;
    delete[] recordSpkPEN;
    delete[] VPEN;
    delete[] RefracTimePEN;
    // current source variables
    delete[] magnitudePEN_input;
    delete[] glbSpkCntR;
    delete[] glbSpkR;
    delete[] recordSpkR;
    delete[] VR;
    delete[] RefracTimeR;
    delete[] glbSpkCntR2;
    delete[] glbSpkR2;
    delete[] recordSpkR2;
    delete[] sTR2;
    delete[] VR2;
    delete[] RefracTimeR2;
    // current source variables
    delete[] magnitudeR2_input;
    delete[] glbSpkCntR4;
    delete[] glbSpkR4;
    delete[] recordSpkR4;
    delete[] sTR4;
    delete[] VR4;
    delete[] RefracTimeR4;
    // current source variables
    delete[] magnitudeR4_input;
    delete[] glbSpkCntStim;
    delete[] glbSpkStim;
    delete[] startSpikeStim;
    delete[] endSpikeStim;
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    delete[] inSynEPG_D7;
    delete[] inSynR4_EPG;
    delete[] inSynR2_EPG;
    delete[] inSynD7_EPG;
    delete[] inSynR_EPG;
    delete[] inSynPEN_EPG;
    delete[] inSynEPG_EPG;
    delete[] inSynStim_EPG;
    delete[] inSynEPG_PEN;
    delete[] inSynEPG_R;
    delete[] inSynR2_R2;
    delete[] inSynR4_R4;
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    delete[] gD7_EPG;
    delete[] gEPG_D7;
    delete[] gEPG_EPG;
    delete[] gEPG_PEN;
    delete[] gEPG_R;
    delete[] gPEN_EPG;
    delete[] gR2_EPG;
    delete[] gR2_R2;
    delete[] gR4_EPG;
    delete[] gR4_R4;
    delete[] gR_EPG;
    delete[] gStim_EPG;
    
}

size_t getFreeDeviceMemBytes() {
    return 0;
}

void stepTime() {
    updateSynapses(t);
    updateNeurons(t, (unsigned int)(iT % numRecordingTimesteps)); 
    iT++;
    t = iT*DT;
}

