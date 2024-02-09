#include "definitionsInternal.h"
#include "supportCode.h"

struct MergedNeuronUpdateGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    unsigned int* startSpike;
    unsigned int* endSpike;
    scalar* spikeTimes;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup1
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    float* sT;
    scalar* V;
    scalar* RefracTime;
    float* inSynInSyn0;
    scalar* magnitudeCS0;
    uint32_t* recordSpk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup2
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    scalar* V;
    scalar* RefracTime;
    float* inSynInSyn0;
    scalar* magnitudeCS0;
    uint32_t* recordSpk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup3
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    float* sT;
    scalar* V;
    scalar* RefracTime;
    float* inSynInSyn0;
    float* inSynInSyn1;
    float* inSynInSyn2;
    float* inSynInSyn3;
    float* inSynInSyn4;
    float* inSynInSyn5;
    float* inSynInSyn6;
    uint32_t* recordSpk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup4
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    scalar* V;
    scalar* RefracTime;
    float* inSynInSyn0;
    uint32_t* recordSpk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronSpikeQueueUpdateGroup0
 {
    unsigned int* spkCnt;
    
}
;
static MergedNeuronUpdateGroup0 mergedNeuronUpdateGroup0[1];
void pushMergedNeuronUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, unsigned int* startSpike, unsigned int* endSpike, scalar* spikeTimes, unsigned int numNeurons) {
    mergedNeuronUpdateGroup0[idx].spkCnt = spkCnt;
    mergedNeuronUpdateGroup0[idx].spk = spk;
    mergedNeuronUpdateGroup0[idx].startSpike = startSpike;
    mergedNeuronUpdateGroup0[idx].endSpike = endSpike;
    mergedNeuronUpdateGroup0[idx].spikeTimes = spikeTimes;
    mergedNeuronUpdateGroup0[idx].numNeurons = numNeurons;
}
static MergedNeuronUpdateGroup1 mergedNeuronUpdateGroup1[2];
void pushMergedNeuronUpdateGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* sT, scalar* V, scalar* RefracTime, float* inSynInSyn0, scalar* magnitudeCS0, uint32_t* recordSpk, unsigned int numNeurons) {
    mergedNeuronUpdateGroup1[idx].spkCnt = spkCnt;
    mergedNeuronUpdateGroup1[idx].spk = spk;
    mergedNeuronUpdateGroup1[idx].sT = sT;
    mergedNeuronUpdateGroup1[idx].V = V;
    mergedNeuronUpdateGroup1[idx].RefracTime = RefracTime;
    mergedNeuronUpdateGroup1[idx].inSynInSyn0 = inSynInSyn0;
    mergedNeuronUpdateGroup1[idx].magnitudeCS0 = magnitudeCS0;
    mergedNeuronUpdateGroup1[idx].recordSpk = recordSpk;
    mergedNeuronUpdateGroup1[idx].numNeurons = numNeurons;
}
static MergedNeuronUpdateGroup2 mergedNeuronUpdateGroup2[1];
void pushMergedNeuronUpdateGroup2ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, scalar* V, scalar* RefracTime, float* inSynInSyn0, scalar* magnitudeCS0, uint32_t* recordSpk, unsigned int numNeurons) {
    mergedNeuronUpdateGroup2[idx].spkCnt = spkCnt;
    mergedNeuronUpdateGroup2[idx].spk = spk;
    mergedNeuronUpdateGroup2[idx].V = V;
    mergedNeuronUpdateGroup2[idx].RefracTime = RefracTime;
    mergedNeuronUpdateGroup2[idx].inSynInSyn0 = inSynInSyn0;
    mergedNeuronUpdateGroup2[idx].magnitudeCS0 = magnitudeCS0;
    mergedNeuronUpdateGroup2[idx].recordSpk = recordSpk;
    mergedNeuronUpdateGroup2[idx].numNeurons = numNeurons;
}
static MergedNeuronUpdateGroup3 mergedNeuronUpdateGroup3[1];
void pushMergedNeuronUpdateGroup3ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* sT, scalar* V, scalar* RefracTime, float* inSynInSyn0, float* inSynInSyn1, float* inSynInSyn2, float* inSynInSyn3, float* inSynInSyn4, float* inSynInSyn5, float* inSynInSyn6, uint32_t* recordSpk, unsigned int numNeurons) {
    mergedNeuronUpdateGroup3[idx].spkCnt = spkCnt;
    mergedNeuronUpdateGroup3[idx].spk = spk;
    mergedNeuronUpdateGroup3[idx].sT = sT;
    mergedNeuronUpdateGroup3[idx].V = V;
    mergedNeuronUpdateGroup3[idx].RefracTime = RefracTime;
    mergedNeuronUpdateGroup3[idx].inSynInSyn0 = inSynInSyn0;
    mergedNeuronUpdateGroup3[idx].inSynInSyn1 = inSynInSyn1;
    mergedNeuronUpdateGroup3[idx].inSynInSyn2 = inSynInSyn2;
    mergedNeuronUpdateGroup3[idx].inSynInSyn3 = inSynInSyn3;
    mergedNeuronUpdateGroup3[idx].inSynInSyn4 = inSynInSyn4;
    mergedNeuronUpdateGroup3[idx].inSynInSyn5 = inSynInSyn5;
    mergedNeuronUpdateGroup3[idx].inSynInSyn6 = inSynInSyn6;
    mergedNeuronUpdateGroup3[idx].recordSpk = recordSpk;
    mergedNeuronUpdateGroup3[idx].numNeurons = numNeurons;
}
static MergedNeuronUpdateGroup4 mergedNeuronUpdateGroup4[2];
void pushMergedNeuronUpdateGroup4ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, scalar* V, scalar* RefracTime, float* inSynInSyn0, uint32_t* recordSpk, unsigned int numNeurons) {
    mergedNeuronUpdateGroup4[idx].spkCnt = spkCnt;
    mergedNeuronUpdateGroup4[idx].spk = spk;
    mergedNeuronUpdateGroup4[idx].V = V;
    mergedNeuronUpdateGroup4[idx].RefracTime = RefracTime;
    mergedNeuronUpdateGroup4[idx].inSynInSyn0 = inSynInSyn0;
    mergedNeuronUpdateGroup4[idx].recordSpk = recordSpk;
    mergedNeuronUpdateGroup4[idx].numNeurons = numNeurons;
}
static MergedNeuronSpikeQueueUpdateGroup0 mergedNeuronSpikeQueueUpdateGroup0[7];
void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt) {
    mergedNeuronSpikeQueueUpdateGroup0[idx].spkCnt = spkCnt;
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
void pushMergedNeuronUpdate0spikeTimesToDevice(unsigned int idx, scalar* value) {
    mergedNeuronUpdateGroup0[idx].spikeTimes = value;
}

void pushMergedNeuronUpdate1recordSpkToDevice(unsigned int idx, uint32_t* value) {
    mergedNeuronUpdateGroup1[idx].recordSpk = value;
}

void pushMergedNeuronUpdate2recordSpkToDevice(unsigned int idx, uint32_t* value) {
    mergedNeuronUpdateGroup2[idx].recordSpk = value;
}

void pushMergedNeuronUpdate3recordSpkToDevice(unsigned int idx, uint32_t* value) {
    mergedNeuronUpdateGroup3[idx].recordSpk = value;
}

void pushMergedNeuronUpdate4recordSpkToDevice(unsigned int idx, uint32_t* value) {
    mergedNeuronUpdateGroup4[idx].recordSpk = value;
}

void updateNeurons(float t, unsigned int recordingTimestep) {
     {
        // merged neuron spike queue update group 0
        for(unsigned int g = 0; g < 7; g++) {
            const auto *group = &mergedNeuronSpikeQueueUpdateGroup0[g]; 
            group->spkCnt[0] = 0;
        }
    }
     {
        // merged neuron update group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronUpdateGroup0[g]; 
            
            for(unsigned int i = 0; i < group->numNeurons; i++) {
                unsigned int lstartSpike = group->startSpike[i];
                const unsigned int lendSpike = group->endSpike[i];
                
                // test whether spike condition was fulfilled previously
                // calculate membrane potential
                
                // test for and register a true spike
                if (lstartSpike != lendSpike && t >= group->spikeTimes[lstartSpike]) {
                    group->spk[group->spkCnt[0]++] = i;
                    // spike reset code
                    lstartSpike++;
                    
                }
                group->startSpike[i] = lstartSpike;
            }
        }
    }
     {
        // merged neuron update group 1
        for(unsigned int g = 0; g < 2; g++) {
            const auto *group = &mergedNeuronUpdateGroup1[g]; 
            const unsigned int numRecordingWords = (group->numNeurons + 31) / 32;
            std::fill_n(&group->recordSpk[recordingTimestep * numRecordingWords], numRecordingWords, 0);
            
            for(unsigned int i = 0; i < group->numNeurons; i++) {
                scalar lV = group->V[i];
                scalar lRefracTime = group->RefracTime[i];
                const float lsT = group->sT[i];
                
                float Isyn = 0;
                 {
                    // pull inSyn values in a coalesced access
                    float linSyn = group->inSynInSyn0[i];
                    Isyn += (9.90066334662237368e-01f) * linSyn;
                    linSyn *= (9.80198673306755253e-01f);
                    group->inSynInSyn0[i] = linSyn;
                }
                // current source 0
                 {
                    scalar lcsmagnitude = group->magnitudeCS0[i];
                    Isyn += lcsmagnitude;
                    group->magnitudeCS0[i] = lcsmagnitude;
                }
                // test whether spike condition was fulfilled previously
                // calculate membrane potential
                if (lRefracTime <= 0.0f) {
                  scalar alpha = ((Isyn + (0.00000000000000000e+00f)) * (1.00000000000000000e+02f)) + (-7.00000000000000000e+01f);
                  lV = alpha - ((9.51229424500714016e-01f) * (alpha - lV));
                }
                else {
                  lRefracTime -= DT;
                }
                
                // test for and register a true spike
                if (lRefracTime <= 0.0f && lV >= (-4.50000000000000000e+01f)) {
                    group->spk[group->spkCnt[0]++] = i;
                    group->sT[i] = t;
                    group->recordSpk[(recordingTimestep * numRecordingWords) + (i / 32)] |= (1 << (i % 32));
                    // spike reset code
                    lV = (-7.00000000000000000e+01f);
                    lRefracTime = (2.00000000000000000e+00f);
                    
                }
                group->V[i] = lV;
                group->RefracTime[i] = lRefracTime;
            }
        }
    }
     {
        // merged neuron update group 2
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronUpdateGroup2[g]; 
            const unsigned int numRecordingWords = (group->numNeurons + 31) / 32;
            std::fill_n(&group->recordSpk[recordingTimestep * numRecordingWords], numRecordingWords, 0);
            
            for(unsigned int i = 0; i < group->numNeurons; i++) {
                scalar lV = group->V[i];
                scalar lRefracTime = group->RefracTime[i];
                
                float Isyn = 0;
                 {
                    // pull inSyn values in a coalesced access
                    float linSyn = group->inSynInSyn0[i];
                    Isyn += (9.95016625083189332e-01f) * linSyn;
                    linSyn *= (9.90049833749168107e-01f);
                    group->inSynInSyn0[i] = linSyn;
                }
                // current source 0
                 {
                    scalar lcsmagnitude = group->magnitudeCS0[i];
                    Isyn += lcsmagnitude;
                    group->magnitudeCS0[i] = lcsmagnitude;
                }
                // test whether spike condition was fulfilled previously
                // calculate membrane potential
                if (lRefracTime <= 0.0f) {
                  scalar alpha = ((Isyn + (0.00000000000000000e+00f)) * (1.00000000000000000e+02f)) + (-7.00000000000000000e+01f);
                  lV = alpha - ((9.51229424500714016e-01f) * (alpha - lV));
                }
                else {
                  lRefracTime -= DT;
                }
                
                // test for and register a true spike
                if (lRefracTime <= 0.0f && lV >= (-4.50000000000000000e+01f)) {
                    group->spk[group->spkCnt[0]++] = i;
                    group->recordSpk[(recordingTimestep * numRecordingWords) + (i / 32)] |= (1 << (i % 32));
                    // spike reset code
                    lV = (-7.00000000000000000e+01f);
                    lRefracTime = (2.00000000000000000e+00f);
                    
                }
                group->V[i] = lV;
                group->RefracTime[i] = lRefracTime;
            }
        }
    }
     {
        // merged neuron update group 3
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronUpdateGroup3[g]; 
            const unsigned int numRecordingWords = (group->numNeurons + 31) / 32;
            std::fill_n(&group->recordSpk[recordingTimestep * numRecordingWords], numRecordingWords, 0);
            
            for(unsigned int i = 0; i < group->numNeurons; i++) {
                scalar lV = group->V[i];
                scalar lRefracTime = group->RefracTime[i];
                const float lsT = group->sT[i];
                
                float Isyn = 0;
                 {
                    // pull inSyn values in a coalesced access
                    float linSyn = group->inSynInSyn0[i];
                    Isyn += (9.90066334662237368e-01f) * linSyn;
                    linSyn *= (9.80198673306755253e-01f);
                    group->inSynInSyn0[i] = linSyn;
                }
                 {
                    // pull inSyn values in a coalesced access
                    float linSyn = group->inSynInSyn1[i];
                    Isyn += (9.90066334662237368e-01f) * linSyn;
                    linSyn *= (9.80198673306755253e-01f);
                    group->inSynInSyn1[i] = linSyn;
                }
                 {
                    // pull inSyn values in a coalesced access
                    float linSyn = group->inSynInSyn2[i];
                    Isyn += (9.90066334662237368e-01f) * linSyn;
                    linSyn *= (9.80198673306755253e-01f);
                    group->inSynInSyn2[i] = linSyn;
                }
                 {
                    // pull inSyn values in a coalesced access
                    float linSyn = group->inSynInSyn3[i];
                    Isyn += (9.90066334662237368e-01f) * linSyn;
                    linSyn *= (9.80198673306755253e-01f);
                    group->inSynInSyn3[i] = linSyn;
                }
                 {
                    // pull inSyn values in a coalesced access
                    float linSyn = group->inSynInSyn4[i];
                    Isyn += (9.95016625083189332e-01f) * linSyn;
                    linSyn *= (9.90049833749168107e-01f);
                    group->inSynInSyn4[i] = linSyn;
                }
                 {
                    // pull inSyn values in a coalesced access
                    float linSyn = group->inSynInSyn5[i];
                    Isyn += (9.95016625083189332e-01f) * linSyn;
                    linSyn *= (9.90049833749168107e-01f);
                    group->inSynInSyn5[i] = linSyn;
                }
                 {
                    // pull inSyn values in a coalesced access
                    float linSyn = group->inSynInSyn6[i];
                    Isyn += (6.32120558828557666e-01f) * linSyn;
                    linSyn *= (3.67879441171442334e-01f);
                    group->inSynInSyn6[i] = linSyn;
                }
                // test whether spike condition was fulfilled previously
                // calculate membrane potential
                if (lRefracTime <= 0.0f) {
                  scalar alpha = ((Isyn + (0.00000000000000000e+00f)) * (1.00000000000000000e+02f)) + (-7.00000000000000000e+01f);
                  lV = alpha - ((9.51229424500714016e-01f) * (alpha - lV));
                }
                else {
                  lRefracTime -= DT;
                }
                
                // test for and register a true spike
                if (lRefracTime <= 0.0f && lV >= (-4.50000000000000000e+01f)) {
                    group->spk[group->spkCnt[0]++] = i;
                    group->sT[i] = t;
                    group->recordSpk[(recordingTimestep * numRecordingWords) + (i / 32)] |= (1 << (i % 32));
                    // spike reset code
                    lV = (-7.00000000000000000e+01f);
                    lRefracTime = (2.00000000000000000e+00f);
                    
                }
                group->V[i] = lV;
                group->RefracTime[i] = lRefracTime;
            }
        }
    }
     {
        // merged neuron update group 4
        for(unsigned int g = 0; g < 2; g++) {
            const auto *group = &mergedNeuronUpdateGroup4[g]; 
            const unsigned int numRecordingWords = (group->numNeurons + 31) / 32;
            std::fill_n(&group->recordSpk[recordingTimestep * numRecordingWords], numRecordingWords, 0);
            
            for(unsigned int i = 0; i < group->numNeurons; i++) {
                scalar lV = group->V[i];
                scalar lRefracTime = group->RefracTime[i];
                
                float Isyn = 0;
                 {
                    // pull inSyn values in a coalesced access
                    float linSyn = group->inSynInSyn0[i];
                    Isyn += (9.95016625083189332e-01f) * linSyn;
                    linSyn *= (9.90049833749168107e-01f);
                    group->inSynInSyn0[i] = linSyn;
                }
                // test whether spike condition was fulfilled previously
                // calculate membrane potential
                if (lRefracTime <= 0.0f) {
                  scalar alpha = ((Isyn + (0.00000000000000000e+00f)) * (1.00000000000000000e+02f)) + (-7.00000000000000000e+01f);
                  lV = alpha - ((9.51229424500714016e-01f) * (alpha - lV));
                }
                else {
                  lRefracTime -= DT;
                }
                
                // test for and register a true spike
                if (lRefracTime <= 0.0f && lV >= (-4.50000000000000000e+01f)) {
                    group->spk[group->spkCnt[0]++] = i;
                    group->recordSpk[(recordingTimestep * numRecordingWords) + (i / 32)] |= (1 << (i % 32));
                    // spike reset code
                    lV = (-7.00000000000000000e+01f);
                    lRefracTime = (2.00000000000000000e+00f);
                    
                }
                group->V[i] = lV;
                group->RefracTime[i] = lRefracTime;
            }
        }
    }
}
