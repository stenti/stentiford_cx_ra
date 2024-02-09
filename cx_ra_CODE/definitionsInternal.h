#pragma once
#include "definitions.h"

#define SUPPORT_CODE_FUNC inline
using std::min;
using std::max;
#define gennCLZ __builtin_clz

// ------------------------------------------------------------------------
// merged group structures
// ------------------------------------------------------------------------
extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged group arrays for host initialisation
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
// current source variables
// current source variables
// current source variables

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// copying merged group structures to device
// ------------------------------------------------------------------------
EXPORT_FUNC void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronInitGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* sT, scalar* V, scalar* RefracTime, float* inSynInSyn0, scalar* magnitudeCS0, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronInitGroup2ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, scalar* V, scalar* RefracTime, float* inSynInSyn0, scalar* magnitudeCS0, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronInitGroup3ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* sT, scalar* V, scalar* RefracTime, float* inSynInSyn0, float* inSynInSyn1, float* inSynInSyn2, float* inSynInSyn3, float* inSynInSyn4, float* inSynInSyn5, float* inSynInSyn6, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronInitGroup4ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, scalar* V, scalar* RefracTime, float* inSynInSyn0, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, unsigned int* startSpike, unsigned int* endSpike, scalar* spikeTimes, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronUpdate0spikeTimesToDevice(unsigned int idx, scalar* value);
EXPORT_FUNC void pushMergedNeuronUpdateGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* sT, scalar* V, scalar* RefracTime, float* inSynInSyn0, scalar* magnitudeCS0, uint32_t* recordSpk, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronUpdate1recordSpkToDevice(unsigned int idx, uint32_t* value);
EXPORT_FUNC void pushMergedNeuronUpdateGroup2ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, scalar* V, scalar* RefracTime, float* inSynInSyn0, scalar* magnitudeCS0, uint32_t* recordSpk, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronUpdate2recordSpkToDevice(unsigned int idx, uint32_t* value);
EXPORT_FUNC void pushMergedNeuronUpdateGroup3ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* sT, scalar* V, scalar* RefracTime, float* inSynInSyn0, float* inSynInSyn1, float* inSynInSyn2, float* inSynInSyn3, float* inSynInSyn4, float* inSynInSyn5, float* inSynInSyn6, uint32_t* recordSpk, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronUpdate3recordSpkToDevice(unsigned int idx, uint32_t* value);
EXPORT_FUNC void pushMergedNeuronUpdateGroup4ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, scalar* V, scalar* RefracTime, float* inSynInSyn0, uint32_t* recordSpk, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronUpdate4recordSpkToDevice(unsigned int idx, uint32_t* value);
EXPORT_FUNC void pushMergedPresynapticUpdateGroup0ToDevice(unsigned int idx, float* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, float* sTPre, float* sTPost, scalar* g, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride);
EXPORT_FUNC void pushMergedPresynapticUpdateGroup1ToDevice(unsigned int idx, float* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, scalar* g, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride);
EXPORT_FUNC void pushMergedPostsynapticUpdateGroup0ToDevice(unsigned int idx, unsigned int* trgSpkCnt, unsigned int* trgSpk, float* sTPre, float* sTPost, scalar* g, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride, unsigned int colStride);
EXPORT_FUNC void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt);
}  // extern "C"
