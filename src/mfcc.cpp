// --- Code 1 (aligned to your Octave) + PAD/TRUNCATE TO 55 FRAMES ---
#include "cmvn.h"
#include <math.h>
#include <string.h>
#include <arduinoFFT.h>

#define FFT_SIZE NFFT
#define SAMPLE_RATE 16000
#define NORMALIZE_INT16 1 // 1: scale int16 to [-1,1] to match audioread


// Global buffers
static float filterbank[NUM_FILTERS][NFFT / 2 + 1];
static double vReal[NFFT];
static double vImag[NFFT];
static float pow_spec[NFFT / 2 + 1];
static float mel_energies[NUM_FILTERS];



// Mel filterbank (ln/1127 form is equivalent to log10/2595 in Octave)
static void mel_filterbank(float fb[NUM_FILTERS][NFFT / 2 + 1], int fs)
{
    float low_mel = 0.0f;
    float high_mel = 1127.0f * logf(1.0f + (fs * 0.5f) / 700.0f);
    float mel_points[NUM_FILTERS + 2];
    int bins[NUM_FILTERS + 2];

    for (int i = 0; i < NUM_FILTERS + 2; ++i)
        mel_points[i] = low_mel + i * (high_mel - low_mel) / (NUM_FILTERS + 1);

    for (int i = 0; i < NUM_FILTERS + 2; ++i)
    {
        float hz = 700.0f * (expf(mel_points[i] / 1127.0f) - 1.0f); // O'Shaughnessy's mel filterbak variant
        int b = (int)floorf((NFFT + 1) * hz / fs);
        // (Optional light guards)
        if (b < 0)
            b = 0;
        if (b > NFFT / 2)
            b = NFFT / 2;
        bins[i] = b;
    }

    memset(fb, 0, sizeof(float) * NUM_FILTERS * (NFFT / 2 + 1));
    for (int i = 1; i <= NUM_FILTERS; ++i)
    {
        int b0 = bins[i - 1], b1 = bins[i], b2 = bins[i + 1];
        if (b1 <= b0)
            b1 = b0 + 1;
        if (b2 <= b1)
            b2 = b1 + 1;

        for (int j = b0; j < b1; ++j)
            fb[i - 1][j] = (float)(j - b0) / (float)(b1 - b0);
        for (int j = b1; j < b2; ++j)
            fb[i - 1][j] = (float)(b2 - j) / (float)(b2 - b1);
    }
}

// DCT-II (orthonormal) — matches Octave's dct()
static void dct_ortho(const float *in, float *out)
{
    for (int k = 0; k < NUM_MFCC_COEFFS; ++k)
    {
        float sum = 0.0f;
        for (int n = 0; n < NUM_FILTERS; ++n)
        {
            sum += in[n] * cosf((float)M_PI * k * (2.0f * n + 1.0f) / (2.0f * NUM_FILTERS)); // Discrete cosine transform
        }
        float norm = (k == 0) ? sqrtf(1.0f / NUM_FILTERS) : sqrtf(2.0f / NUM_FILTERS);
        out[k] = sum * norm;
    }
}

void extract_mfcc(const int16_t *signal, int length, int sample_rate, MFCC_Result *result)
{
    const int frame_len = sample_rate * FRAME_LEN_MS / 1000;   // 25 ms
    const int frame_step = sample_rate * FRAME_STEP_MS / 1000; // 10 ms

    // Always report exactly 55 frames (pad with zeros or truncate)
    result->num_frames = TARGET_FRAMES;

    // Zero the entire output [55 x NUM_MFCC_COEFFS] up-front (padding baseline)
    for (int f = 0; f < TARGET_FRAMES; ++f)
    {
        memset(result->mfcc[f], 0, sizeof(float) * NUM_MFCC_COEFFS);
    }

    // Nothing to compute? keep zeros
    if (!signal || length <= 0)
        return;

    // Drop partial last frame to match octave
    int raw_frames;
    if (length < frame_len)
    {
        raw_frames = 0; // no full frames
    }
    else
    {
        raw_frames = 1 + ((length - frame_len) / frame_step); // floor
    }

    int frames_to_compute = raw_frames;
    if (frames_to_compute > TARGET_FRAMES)
        frames_to_compute = TARGET_FRAMES;

    // result->mfcc already zero-initialized, so anything >= frames_to_compute stays 0

    mel_filterbank(filterbank, sample_rate);

    // Hamming window for frame_len only
    static float hamming_win[NFFT];
    for (int i = 0; i < frame_len; ++i)
    {
        hamming_win[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / (frame_len - 1));
    }
    for (int i = frame_len; i < NFFT; ++i)
        hamming_win[i] = 0.0f;

    for (int f = 0; f < frames_to_compute; ++f)
    {
        const int start = f * frame_step;
        const int remaining = length - start;

        // Pre-emphasis across frames + manual Hamming, zero-pad to NFFT
        for (int i = 0; i < frame_len; ++i)
        {
            const int idx = start + i;

            float sx = 0.0f;
            float prev = 0.0f;

            if (idx >= 0 && idx < length)
            {
                sx = (float)signal[idx];
            }
            if (idx - 1 >= 0 && idx - 1 < length)
            {
                prev = (float)signal[idx - 1];
            }

#if NORMALIZE_INT16
            sx *= (1.0f / 32768.0f);
            prev *= (1.0f / 32768.0f);
#endif

            const float x = (sx - 0.97f * prev) * hamming_win[i];
            vReal[i] = (double)x;
            vImag[i] = 0.0;
        }
        for (int i = frame_len; i < NFFT; ++i)
        {
            vReal[i] = 0.0;
            vImag[i] = 0.0;
        }

        // FFT — no extra windowing here (already applied)
        ArduinoFFT<double> fft(vReal, vImag, NFFT, sample_rate);
        // fft.windowing(...)  // keep disabled
        fft.compute(vReal, vImag, NFFT, FFT_FORWARD);
        fft.complexToMagnitude(vReal, vImag, NFFT);

        // Power spectrum (one-sided)
        for (int i = 0; i <= NFFT / 2; ++i)
            pow_spec[i] = (float)(vReal[i] * vReal[i]);

        // Mel energies
        memset(mel_energies, 0, sizeof(mel_energies));
        for (int i = 0; i < NUM_FILTERS; ++i)
            for (int j = 0; j <= NFFT / 2; ++j)
                mel_energies[i] += filterbank[i][j] * pow_spec[j];

        // Log compression (natural log)
        for (int i = 0; i < NUM_FILTERS; ++i)
            mel_energies[i] = logf(fmaxf(mel_energies[i], 1e-10f));

        // DCT → MFCCs (includes C0 at index 0)
        dct_ortho(mel_energies, result->mfcc[f]);
    }

    // If raw_frames < 55 → trailing rows remain zero (already padded)
    // If raw_frames > 55 → extra frames implicitly truncated
}
