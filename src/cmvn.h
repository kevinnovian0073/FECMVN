#pragma once
#include "Arduino.h"

// ===== MFCC params (match your Octave) =====
#define FRAME_LEN_MS 25  // 25 ms
#define FRAME_STEP_MS 10 // 10 ms
#define NFFT 512
#define NUM_FILTERS 26
#define NUM_MFCC_COEFFS 13
#define TARGET_FRAMES 55 // fixed output length

typedef struct
{
    int num_frames; // will always be TARGET_FRAMES after pad/truncate

    // MFCC only (output of extract_mfcc)
    float mfcc[TARGET_FRAMES][NUM_MFCC_COEFFS];

    // Final features after delta + delta-delta + CMVN:
    // feat[t][k][0] = MFCC (CMVN)
    // feat[t][k][1] = Delta (CMVN)
    // feat[t][k][2] = Delta-Delta (CMVN)
    float feat[TARGET_FRAMES][NUM_MFCC_COEFFS][3];

} MFCC_Result;

// Step 1: compute MFCC (55x13) into result->mfcc
void extract_mfcc(const int16_t *signal, int length, int sample_rate, MFCC_Result *result);

// Step 2: compute Delta + Delta-Delta + CMVN from result->mfcc into result->feat
void extract_cmvn_features(MFCC_Result *result);
