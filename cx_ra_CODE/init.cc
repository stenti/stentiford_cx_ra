#include "definitionsInternal.h"
struct MergedNeuronInitGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronInitGroup1
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    float* sT;
    scalar* V;
    scalar* RefracTime;
    float* inSynInSyn0;
    scalar* magnitudeCS0;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronInitGroup2
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    scalar* V;
    scalar* RefracTime;
    float* inSynInSyn0;
    scalar* magnitudeCS0;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronInitGroup3
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
    unsigned int numNeurons;
    
}
;
struct MergedNeuronInitGroup4
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    scalar* V;
    scalar* RefracTime;
    float* inSynInSyn0;
    unsigned int numNeurons;
    
}
;
static MergedNeuronInitGroup0 mergedNeuronInitGroup0[1];
void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, unsigned int numNeurons) {
    mergedNeuronInitGroup0[idx].spkCnt = spkCnt;
    mergedNeuronInitGroup0[idx].spk = spk;
    mergedNeuronInitGroup0[idx].numNeurons = numNeurons;
}
static MergedNeuronInitGroup1 mergedNeuronInitGroup1[2];
void pushMergedNeuronInitGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* sT, scalar* V, scalar* RefracTime, float* inSynInSyn0, scalar* magnitudeCS0, unsigned int numNeurons) {
    mergedNeuronInitGroup1[idx].spkCnt = spkCnt;
    mergedNeuronInitGroup1[idx].spk = spk;
    mergedNeuronInitGroup1[idx].sT = sT;
    mergedNeuronInitGroup1[idx].V = V;
    mergedNeuronInitGroup1[idx].RefracTime = RefracTime;
    mergedNeuronInitGroup1[idx].inSynInSyn0 = inSynInSyn0;
    mergedNeuronInitGroup1[idx].magnitudeCS0 = magnitudeCS0;
    mergedNeuronInitGroup1[idx].numNeurons = numNeurons;
}
static MergedNeuronInitGroup2 mergedNeuronInitGroup2[1];
void pushMergedNeuronInitGroup2ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, scalar* V, scalar* RefracTime, float* inSynInSyn0, scalar* magnitudeCS0, unsigned int numNeurons) {
    mergedNeuronInitGroup2[idx].spkCnt = spkCnt;
    mergedNeuronInitGroup2[idx].spk = spk;
    mergedNeuronInitGroup2[idx].V = V;
    mergedNeuronInitGroup2[idx].RefracTime = RefracTime;
    mergedNeuronInitGroup2[idx].inSynInSyn0 = inSynInSyn0;
    mergedNeuronInitGroup2[idx].magnitudeCS0 = magnitudeCS0;
    mergedNeuronInitGroup2[idx].numNeurons = numNeurons;
}
static MergedNeuronInitGroup3 mergedNeuronInitGroup3[1];
void pushMergedNeuronInitGroup3ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* sT, scalar* V, scalar* RefracTime, float* inSynInSyn0, float* inSynInSyn1, float* inSynInSyn2, float* inSynInSyn3, float* inSynInSyn4, float* inSynInSyn5, float* inSynInSyn6, unsigned int numNeurons) {
    mergedNeuronInitGroup3[idx].spkCnt = spkCnt;
    mergedNeuronInitGroup3[idx].spk = spk;
    mergedNeuronInitGroup3[idx].sT = sT;
    mergedNeuronInitGroup3[idx].V = V;
    mergedNeuronInitGroup3[idx].RefracTime = RefracTime;
    mergedNeuronInitGroup3[idx].inSynInSyn0 = inSynInSyn0;
    mergedNeuronInitGroup3[idx].inSynInSyn1 = inSynInSyn1;
    mergedNeuronInitGroup3[idx].inSynInSyn2 = inSynInSyn2;
    mergedNeuronInitGroup3[idx].inSynInSyn3 = inSynInSyn3;
    mergedNeuronInitGroup3[idx].inSynInSyn4 = inSynInSyn4;
    mergedNeuronInitGroup3[idx].inSynInSyn5 = inSynInSyn5;
    mergedNeuronInitGroup3[idx].inSynInSyn6 = inSynInSyn6;
    mergedNeuronInitGroup3[idx].numNeurons = numNeurons;
}
static MergedNeuronInitGroup4 mergedNeuronInitGroup4[2];
void pushMergedNeuronInitGroup4ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, scalar* V, scalar* RefracTime, float* inSynInSyn0, unsigned int numNeurons) {
    mergedNeuronInitGroup4[idx].spkCnt = spkCnt;
    mergedNeuronInitGroup4[idx].spk = spk;
    mergedNeuronInitGroup4[idx].V = V;
    mergedNeuronInitGroup4[idx].RefracTime = RefracTime;
    mergedNeuronInitGroup4[idx].inSynInSyn0 = inSynInSyn0;
    mergedNeuronInitGroup4[idx].numNeurons = numNeurons;
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
void initialize() {
    // ------------------------------------------------------------------------
    // Neuron groups
     {
        // merged neuron init group 0
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronInitGroup0[g]; 
            group->spkCnt[0] = 0;
            for (unsigned i = 0; i < (group->numNeurons); i++) {
                group->spk[i] = 0;
            }
            // current source variables
        }
    }
     {
        // merged neuron init group 1
        for(unsigned int g = 0; g < 2; g++) {
            const auto *group = &mergedNeuronInitGroup1[g]; 
            group->spkCnt[0] = 0;
            for (unsigned i = 0; i < (group->numNeurons); i++) {
                group->spk[i] = 0;
            }
            for (unsigned i = 0; i < (group->numNeurons); i++) {
                group->sT[i] = -TIME_MAX;
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    scalar initVal;
                    initVal = (-5.20000000000000000e+01f);
                    group->V[i] = initVal;
                }
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00f);
                    group->RefracTime[i] = initVal;
                }
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    group->inSynInSyn0[i] = 0.000000000e+00f;
                }
            }
            // current source variables
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00f);
                    group->magnitudeCS0[i] = initVal;
                }
            }
        }
    }
     {
        // merged neuron init group 2
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronInitGroup2[g]; 
            group->spkCnt[0] = 0;
            for (unsigned i = 0; i < (group->numNeurons); i++) {
                group->spk[i] = 0;
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    scalar initVal;
                    initVal = (-5.20000000000000000e+01f);
                    group->V[i] = initVal;
                }
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00f);
                    group->RefracTime[i] = initVal;
                }
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    group->inSynInSyn0[i] = 0.000000000e+00f;
                }
            }
            // current source variables
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00f);
                    group->magnitudeCS0[i] = initVal;
                }
            }
        }
    }
     {
        // merged neuron init group 3
        for(unsigned int g = 0; g < 1; g++) {
            const auto *group = &mergedNeuronInitGroup3[g]; 
            group->spkCnt[0] = 0;
            for (unsigned i = 0; i < (group->numNeurons); i++) {
                group->spk[i] = 0;
            }
            for (unsigned i = 0; i < (group->numNeurons); i++) {
                group->sT[i] = -TIME_MAX;
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    scalar initVal;
                    initVal = (-5.20000000000000000e+01f);
                    group->V[i] = initVal;
                }
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00f);
                    group->RefracTime[i] = initVal;
                }
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    group->inSynInSyn0[i] = 0.000000000e+00f;
                }
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    group->inSynInSyn1[i] = 0.000000000e+00f;
                }
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    group->inSynInSyn2[i] = 0.000000000e+00f;
                }
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    group->inSynInSyn3[i] = 0.000000000e+00f;
                }
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    group->inSynInSyn4[i] = 0.000000000e+00f;
                }
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    group->inSynInSyn5[i] = 0.000000000e+00f;
                }
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    group->inSynInSyn6[i] = 0.000000000e+00f;
                }
            }
            // current source variables
        }
    }
     {
        // merged neuron init group 4
        for(unsigned int g = 0; g < 2; g++) {
            const auto *group = &mergedNeuronInitGroup4[g]; 
            group->spkCnt[0] = 0;
            for (unsigned i = 0; i < (group->numNeurons); i++) {
                group->spk[i] = 0;
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    scalar initVal;
                    initVal = (-5.20000000000000000e+01f);
                    group->V[i] = initVal;
                }
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00f);
                    group->RefracTime[i] = initVal;
                }
            }
             {
                for (unsigned i = 0; i < (group->numNeurons); i++) {
                    group->inSynInSyn0[i] = 0.000000000e+00f;
                }
            }
            // current source variables
        }
    }
    // ------------------------------------------------------------------------
    // Synapse groups
    // ------------------------------------------------------------------------
    // Custom update groups
    // ------------------------------------------------------------------------
    // Custom WU update groups
    // ------------------------------------------------------------------------
    // Synapse sparse connectivity
}

void initializeSparse() {
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
    // ------------------------------------------------------------------------
    // Custom sparse WU update groups
}
