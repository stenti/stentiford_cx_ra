#include "definitionsInternal.h"
#include "supportCode.h"

struct MergedPresynapticUpdateGroup0
 {
    float* inSyn;
    unsigned int* srcSpkCnt;
    unsigned int* srcSpk;
    float* sTPre;
    float* sTPost;
    scalar* g;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    
}
;
struct MergedPresynapticUpdateGroup1
 {
    float* inSyn;
    unsigned int* srcSpkCnt;
    unsigned int* srcSpk;
    scalar* g;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    
}
;
struct MergedPostsynapticUpdateGroup0
 {
    unsigned int* trgSpkCnt;
    unsigned int* trgSpk;
    float* sTPre;
    float* sTPost;
    scalar* g;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    unsigned int colStride;
    
}
;
static MergedPresynapticUpdateGroup0 mergedPresynapticUpdateGroup0[2];
void pushMergedPresynapticUpdateGroup0ToDevice(unsigned int idx, float* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, float* sTPre, float* sTPost, scalar* g, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride) {
    mergedPresynapticUpdateGroup0[idx].inSyn = inSyn;
    mergedPresynapticUpdateGroup0[idx].srcSpkCnt = srcSpkCnt;
    mergedPresynapticUpdateGroup0[idx].srcSpk = srcSpk;
    mergedPresynapticUpdateGroup0[idx].sTPre = sTPre;
    mergedPresynapticUpdateGroup0[idx].sTPost = sTPost;
    mergedPresynapticUpdateGroup0[idx].g = g;
    mergedPresynapticUpdateGroup0[idx].numSrcNeurons = numSrcNeurons;
    mergedPresynapticUpdateGroup0[idx].numTrgNeurons = numTrgNeurons;
    mergedPresynapticUpdateGroup0[idx].rowStride = rowStride;
}
static MergedPresynapticUpdateGroup1 mergedPresynapticUpdateGroup1[10];
void pushMergedPresynapticUpdateGroup1ToDevice(unsigned int idx, float* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, scalar* g, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride) {
    mergedPresynapticUpdateGroup1[idx].inSyn = inSyn;
    mergedPresynapticUpdateGroup1[idx].srcSpkCnt = srcSpkCnt;
    mergedPresynapticUpdateGroup1[idx].srcSpk = srcSpk;
    mergedPresynapticUpdateGroup1[idx].g = g;
    mergedPresynapticUpdateGroup1[idx].numSrcNeurons = numSrcNeurons;
    mergedPresynapticUpdateGroup1[idx].numTrgNeurons = numTrgNeurons;
    mergedPresynapticUpdateGroup1[idx].rowStride = rowStride;
}
static MergedPostsynapticUpdateGroup0 mergedPostsynapticUpdateGroup0[2];
void pushMergedPostsynapticUpdateGroup0ToDevice(unsigned int idx, unsigned int* trgSpkCnt, unsigned int* trgSpk, float* sTPre, float* sTPost, scalar* g, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride, unsigned int colStride) {
    mergedPostsynapticUpdateGroup0[idx].trgSpkCnt = trgSpkCnt;
    mergedPostsynapticUpdateGroup0[idx].trgSpk = trgSpk;
    mergedPostsynapticUpdateGroup0[idx].sTPre = sTPre;
    mergedPostsynapticUpdateGroup0[idx].sTPost = sTPost;
    mergedPostsynapticUpdateGroup0[idx].g = g;
    mergedPostsynapticUpdateGroup0[idx].numSrcNeurons = numSrcNeurons;
    mergedPostsynapticUpdateGroup0[idx].numTrgNeurons = numTrgNeurons;
    mergedPostsynapticUpdateGroup0[idx].rowStride = rowStride;
    mergedPostsynapticUpdateGroup0[idx].colStride = colStride;
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
void updateSynapses(float t) {
     {
        // merged presynaptic update group 0
        for(unsigned int g = 0; g < 2; g++) {
            const auto *group = &mergedPresynapticUpdateGroup0[g]; 
            // process presynaptic events: True Spikes
            for (unsigned int i = 0; i < group->srcSpkCnt[0]; i++) {
                const unsigned int ipre = group->srcSpk[i];
                for (unsigned int ipost = 0; ipost < group->numTrgNeurons; ipost++) {
                    const unsigned int synAddress = (ipre * group->numTrgNeurons) + ipost;
                    
                    group->inSyn[ipost] += group->g[synAddress];
                    if(t > (0.00000000000000000e+00f) && t < (1.94330000000000000e+04f)){
                        scalar dt = t - (1.00000000000000000e+00f + group->sTPost[ipost]);
                        scalar timing = exp(-dt / (5.00000000000000000e+01f)) - (5.99999999999999978e-02f);
                        scalar newWeight = group->g[synAddress] + ((1.00000000000000002e-02f) * timing);
                        group->g[synAddress] = fmin((0.00000000000000000e+00f), fmax((-2.00000000000000011e-01f), newWeight));
                    }
                }
            }
            
        }
    }
     {
        // merged presynaptic update group 1
        for(unsigned int g = 0; g < 10; g++) {
            const auto *group = &mergedPresynapticUpdateGroup1[g]; 
            // process presynaptic events: True Spikes
            for (unsigned int i = 0; i < group->srcSpkCnt[0]; i++) {
                const unsigned int ipre = group->srcSpk[i];
                for (unsigned int ipost = 0; ipost < group->numTrgNeurons; ipost++) {
                    const unsigned int synAddress = (ipre * group->numTrgNeurons) + ipost;
                    group->inSyn[ipost] += group->g[synAddress];
                }
            }
            
        }
    }
     {
        // merged postsynaptic update group 0
        for(unsigned int g = 0; g < 2; g++) {
            const auto *group = &mergedPostsynapticUpdateGroup0[g]; 
            const unsigned int numSpikes = group->trgSpkCnt[0];
            for (unsigned int j = 0; j < numSpikes; j++) {
                const unsigned int spike = group->trgSpk[j];
                for (unsigned int i = 0; i < group->numSrcNeurons; i++) {
                    
                    if(t > (0.00000000000000000e+00f) && t < (1.94330000000000000e+04f)){
                        scalar dt = t - (1.00000000000000000e+00f + group->sTPre[i]);
                        scalar timing = exp(-dt / (5.00000000000000000e+01f)) - (5.99999999999999978e-02f);
                        scalar newWeight = group->g[((group->numTrgNeurons * i) + spike)] + ((1.00000000000000002e-02f) * timing);
                        group->g[((group->numTrgNeurons * i) + spike)] = fmin((0.00000000000000000e+00f), fmax((-2.00000000000000011e-01f), newWeight));
                    }
                }
            }
            
        }
    }
}
