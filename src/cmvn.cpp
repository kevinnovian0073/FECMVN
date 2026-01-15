#include "cmvn.h"
#include <math.h>
#include <string.h>

#ifndef EPS_CMVN
#define EPS_CMVN 1e-8f
#endif

#define T TARGET_FRAMES
#define K NUM_MFCC_COEFFS

// Move big buffers to static storage (NOT stack)
static float d1[T][K];
static float d2[T][K];
static float mfcc_tmp[T][K];
static float d1_tmp[T][K];
static float d2_tmp[T][K];

// --- Delta regression with edge clamp ---
static void compute_delta_clamp(const float in[T][K], float out[T][K], int N)
{
    float denom = 0.0f;
    for (int n = 1; n <= N; ++n) denom += (float)(n * n);
    denom *= 2.0f;

    for (int t = 0; t < T; ++t)
    {
        for (int k = 0; k < K; ++k)
        {
            float num = 0.0f;
            for (int n = 1; n <= N; ++n)
            {
                int tp = t + n; if (tp >= T) tp = T - 1;
                int tm = t - n; if (tm < 0)  tm = 0;
                num += (float)n * (in[tp][k] - in[tm][k]);
            }
            out[t][k] = num / denom;
        }
    }
}

// --- CMVN per coefficient, in-place (sample std, T-1) ---
static void cmvn_sample_inplace(float X[T][K])
{
    float mean[K];
    float var[K];

    for (int k = 0; k < K; ++k) { mean[k] = 0.0f; var[k] = 0.0f; }

    for (int k = 0; k < K; ++k)
    {
        float s = 0.0f;
        for (int t = 0; t < T; ++t) s += X[t][k];
        mean[k] = s / (float)T;
    }

    const float denom = (T > 1) ? (float)(T - 1) : 1.0f;
    for (int k = 0; k < K; ++k)
    {
        float s = 0.0f;
        for (int t = 0; t < T; ++t)
        {
            float d = X[t][k] - mean[k];
            s += d * d;
        }
        var[k] = s / denom;
    }

    for (int k = 0; k < K; ++k)
    {
        float stdv = sqrtf(var[k]) + EPS_CMVN;
        for (int t = 0; t < T; ++t)
            X[t][k] = (X[t][k] - mean[k]) / stdv;
    }
}

void extract_cmvn_features(MFCC_Result *result)
{
    if (!result) return;

    // 1) Delta and Delta-Delta
    compute_delta_clamp(result->mfcc, d1, 2);
    compute_delta_clamp(d1, d2, 2);

    // 2) Copy into temporaries
    memcpy(mfcc_tmp, result->mfcc, sizeof(mfcc_tmp));
    memcpy(d1_tmp,   d1,          sizeof(d1_tmp));
    memcpy(d2_tmp,   d2,          sizeof(d2_tmp));

    // 3) CMVN per channel
    cmvn_sample_inplace(mfcc_tmp);
    cmvn_sample_inplace(d1_tmp);
    cmvn_sample_inplace(d2_tmp);

    // 4) Pack into result->feat
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k)
        {
            result->feat[t][k][0] = mfcc_tmp[t][k];
            result->feat[t][k][1] = d1_tmp[t][k];
            result->feat[t][k][2] = d2_tmp[t][k];
        }
}
