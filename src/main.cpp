#include <Arduino.h>
#include <SD.h>
#include <SPI.h>
#include "driver/adc.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>
#include "VoiceActivityDetector.h"
#include "cmvn.h"

const int SAMPLE_RATE = 16000;
const int BUFFER_SIZE = 512;
const int TOTAL_SAMPLES = 24000;       // 1.5 seconds @ 16kHz
const int CALIBRATION_SAMPLES = 48000; // 3 seconds @ 16kHz
const int MAX_VAD_OUTPUT = 24000;

#define BUTTON_PIN 40
#define ADC_MIC_PIN 5

#if !defined(CONFIG_IDF_TARGET_ESP32)
#define VSPI FSPI
#endif

// TFT pins
#define TFT_CS 9
#define TFT_RST 10
#define TFT_DC 11
#define TFT_MOSI 12
#define TFT_SCK 13

// SD pins
#define SD_MISO 20
#define SD_MOSI 47
#define SD_SCK 21
#define SD_CS 48

// for debugging cmvn only
unsigned long recordingStartTime = 0;
unsigned long recordingDuration = 0;
unsigned long recordingEndTime = 0;

unsigned long CMVNDuration = 0;
unsigned long MFCCDuration = 0;

int16_t *fullBuffer = nullptr; // Copy of filtered recording (TOTAL_SAMPLES)
int16_t *vad_output = nullptr; // Output after VAD (MAX_VAD_OUTPUT)

// HPF for 100Hz
float hpf_b[3] = {0.9726, -1.9452, 0.9726};
float hpf_a[3] = {1.0000, -1.9445, 0.9460};
float hpf_x[3] = {0, 0, 0}, hpf_y[3] = {0, 0, 0};

float b[3] = {0.4107, 0, -0.4107};
float a[3] = {1.0000, -1.0739, 0.1786};
float x[3] = {0, 0, 0}, y[3] = {0, 0, 0};

volatile bool recording = false;
volatile bool samplingFlag = false;

int16_t buffer[BUFFER_SIZE];
volatile size_t bufferIndex = 0;
volatile size_t totalSamplesRecorded = 0;
volatile bool bufferReadyToWrite = false;

portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;
esp_timer_handle_t periodic_timer;

SPIClass spiSD(VSPI);
SPIClass spiTFT(HSPI);
Adafruit_ST7735 tft = Adafruit_ST7735(&spiTFT, TFT_CS, TFT_DC, TFT_RST);

File audioFile;
String filename;

VoiceActivityDetector vad;
bool isCalibrating = false;

// MFCC result (includes mfcc + feat[][][3])
MFCC_Result mfcc_result;

void onTimer(void *arg);
void startRecording();
void stopRecording();
void handleAudioProcessing();
String getUniqueFilename();
void writeWavHeader(File file, uint32_t dataSize);
void showStatus(const char *msg);

String makeVadFilename(const String &origWav);
bool saveVadWav(const String &path, const int16_t *samples, int sampleCount);

int getLastFileIndex()
{
  File indexFile = SD.open("/index_validate.txt", FILE_READ);
  int index = 1;
  if (indexFile)
  {
    index = indexFile.parseInt();
    indexFile.close();
  }
  return index;
}

void saveLastFileIndex(int index)
{
  File indexFile = SD.open("/index_validate.txt", FILE_WRITE);
  if (indexFile)
  {
    indexFile.println(index);
    indexFile.close();
  }
}

String makeVadFilename(const String &origWav)
{
  String out = origWav;
  out.replace(".wav", "_vad.wav");
  return out;
}

bool saveVadWav(const String &path, const int16_t *samples, int sampleCount)
{
  if (!samples || sampleCount <= 0)
    return false;

  File f = SD.open(path, FILE_WRITE);
  if (!f)
  {
    Serial.printf("Failed to open VAD WAV: %s\n", path.c_str());
    return false;
  }

  // data size in bytes (16-bit mono)
  uint32_t dataSize = (uint32_t)sampleCount * sizeof(int16_t);

  // Write header then data
  writeWavHeader(f, dataSize);
  size_t written = f.write((const uint8_t *)samples, dataSize);
  f.flush();
  f.close();

  if (written != dataSize)
  {
    Serial.printf("VAD WAV short write: %u/%u bytes\n", (unsigned)written, (unsigned)dataSize);
    return false;
  }

  Serial.println("Saved VAD WAV: " + path);
  return true;
}

static bool save_cmvn_csv_55x39(const char *path, const MFCC_Result *r)
{
  File f = SD.open(path, FILE_WRITE);
  if (!f)
  {
    Serial.printf("Failed to open CSV: %s\n", path);
    return false;
  }

  // 55 rows x 39 cols: [MFCC(13), D1(13), D2(13)]
  for (int t = 0; t < TARGET_FRAMES; ++t)
  {
    for (int k = 0; k < NUM_MFCC_COEFFS; ++k)
    {
      f.print(r->feat[t][k][0], 6);
      f.print(',');
    }
    for (int k = 0; k < NUM_MFCC_COEFFS; ++k)
    {
      f.print(r->feat[t][k][1], 6);
      f.print(',');
    }
    for (int k = 0; k < NUM_MFCC_COEFFS; ++k)
    {
      f.print(r->feat[t][k][2], 6);
      if (k < NUM_MFCC_COEFFS - 1)
        f.print(',');
    }
    f.print('\n');
  }

  f.flush();
  f.close();
  Serial.printf("Saved CSV: %s\n", path);
  return true;
}

void onTimer(void *arg)
{
  samplingFlag = true; // keep ISR tiny
}

void setup()
{
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  // TFT
  spiTFT.begin(TFT_SCK, -1, TFT_MOSI, TFT_CS);
  tft.initR(INITR_BLACKTAB);
  tft.setRotation(3);
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(2);

  // SD
  spiSD.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  if (!SD.begin(SD_CS, spiSD))
  {
    Serial.println("SD init failed!");
    showStatus("SD Failed!");
    while (1)
      ;
  }

  // ADC
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(ADC1_CHANNEL_4, ADC_ATTEN_DB_11);

  // Timer
  const esp_timer_create_args_t timer_args = {
      .callback = &onTimer,
      .name = "sampling_timer"};
  esp_timer_create(&timer_args, &periodic_timer);

  // PSRAM check (recommended)
  if (!psramFound())
  {
    Serial.println("PSRAM not found! (Buffers may crash)");
    showStatus("No PSRAM!");
    // You can still continue, but you asked PSRAM to avoid restarts.
    while (1)
      ;
  }
  Serial.println("PSRAM found.");

  // VAD init (calibration mode)
  vad_init(&vad, 0.020f, SAMPLE_RATE / 50, 5);
  vad_reset_calibration(&vad);

  // Start calibration capture
  isCalibrating = true;
  totalSamplesRecorded = 0;

  showStatus("Calibrating");
  Serial.println("Calibrating...");

  esp_timer_start_periodic(periodic_timer, 1000000 / SAMPLE_RATE);
}

void loop()
{
  // button triggers recording (not during calibration)
  static bool lastButtonState = HIGH;
  static unsigned long lastPress = 0;

  bool current = digitalRead(BUTTON_PIN);
  if (!isCalibrating &&
      current == LOW && lastButtonState == HIGH &&
      millis() - lastPress > 200)
  {
    lastPress = millis();
    if (!recording)
      startRecording();
  }
  lastButtonState = current;

  // main audio work
  handleAudioProcessing();
}

void startRecording()
{
  recordingStartTime = micros(); // <<< START timing
  Serial.println("Recording started.");
  showStatus("Recording");

  // Allocate fullBuffer in PSRAM (once per recording)
  if (fullBuffer)
  {
    heap_caps_free(fullBuffer);
    fullBuffer = nullptr;
  }
  fullBuffer = (int16_t *)heap_caps_malloc(
      TOTAL_SAMPLES * sizeof(int16_t),
      MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!fullBuffer)
  {
    Serial.println("Failed to allocate fullBuffer in PSRAM!");
    showStatus("PSRAM fail");
    recording = false;
    return;
  }
  memset(fullBuffer, 0, TOTAL_SAMPLES * sizeof(int16_t));

  // Reset counters/flags
  totalSamplesRecorded = 0;
  bufferIndex = 0;
  bufferReadyToWrite = false;

  // Prepare WAV file
  filename = getUniqueFilename();
  audioFile = SD.open(filename, FILE_WRITE);
  if (!audioFile)
  {
    Serial.println("Failed to open WAV file!");
    showStatus("WAV open fail");
    recording = false;
    return;
  }

  // reserve header
  for (int i = 0; i < 44; i++)
    audioFile.write((uint8_t)0);

  recording = true;

  // Arm timer
  esp_timer_start_periodic(periodic_timer, 1000000 / SAMPLE_RATE);
}

void stopRecording()
{
  recordingEndTime = micros(); //
  Serial.println("Recording stopped.");
  showStatus("Processing");

  // Finish WAV header
  if (audioFile)
  {
    uint32_t dataSize = audioFile.size() - 44;
    audioFile.seek(0);
    writeWavHeader(audioFile, dataSize);
    audioFile.close();
    Serial.println("Saved WAV: " + filename);
  }

  // Allocate vad_output in PSRAM
  if (vad_output)
  {
    heap_caps_free(vad_output);
    vad_output = nullptr;
  }
  vad_output = (int16_t *)heap_caps_malloc(
      MAX_VAD_OUTPUT * sizeof(int16_t),
      MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!vad_output)
  {
    Serial.println("Failed to allocate vad_output in PSRAM!");
  }

  // Run VAD
  int vad_sample_count = 0;
  if (vad_output)
  {
    vad_sample_count = vad_extract(&vad, fullBuffer, TOTAL_SAMPLES,
                                   vad_output, MAX_VAD_OUTPUT);
  }

  Serial.print("VAD Output Samples: ");
  Serial.println(vad_sample_count);

  // Save VAD-trimmed WAV
  if (vad_sample_count > 0 && vad_output)
  {
    String vadWavName = makeVadFilename(filename);
    saveVadWav(vadWavName, vad_output, vad_sample_count);
  }

  if (vad_sample_count > 0)
  {
    vad_update_threshold_after_speech(&vad);

    // ===== MFCC timing =====
    unsigned long t0 = micros();
    extract_mfcc(vad_output, vad_sample_count, SAMPLE_RATE, &mfcc_result);
    unsigned long t1 = micros();
    MFCCDuration = t1 - t0;

    // ===== CMVN timing =====
    unsigned long t2 = micros();
    extract_cmvn_features(&mfcc_result);
    unsigned long t3 = micros();
    CMVNDuration = t3 - t2;

    // Save CSV
    String csv_name = filename;
    csv_name.replace(".wav", "_cmvn.csv");
    save_cmvn_csv_55x39(csv_name.c_str(), &mfcc_result);
  }

  else
  {
    Serial.println("No voiced audio detected. Skipping MFCC/CMVN/CSV.");
  }

  // Serial.printf("Total samples recorded: %u\n", (unsigned)TOTAL_SAMPLES);

  // Free PSRAM buffers for next run
  if (fullBuffer)
  {
    heap_caps_free(fullBuffer);
    fullBuffer = nullptr;
  }
  if (vad_output)
  {
    heap_caps_free(vad_output);
    vad_output = nullptr;
  }

  showStatus("Done");
  recordingDuration = recordingEndTime - recordingStartTime;
  Serial.printf("Recording time : %.2f ms\n", recordingDuration / 1000.0f);
  Serial.printf("MFCC time      : %.2f ms\n", MFCCDuration / 1000.0f);
  Serial.printf("CMVN time      : %.2f ms\n", CMVNDuration / 1000.0f);

  Serial.flush();
}

void handleAudioProcessing()
{
  // If calibrating, we are also sampling here
  // If recording, we are sampling too
  if (!samplingFlag)
    return;

  // (simple approach; good enough for many cases)
  while (samplingFlag)
  {
    samplingFlag = false;

    // --- read ADC ---
    int16_t raw = (int16_t)((2048 - adc1_get_raw(ADC1_CHANNEL_4)) * 16);

    // --- filtering (HPF -> BPF) ---
    hpf_x[0] = (float)raw;
    hpf_y[0] = hpf_b[0] * hpf_x[0] + hpf_b[1] * hpf_x[1] + hpf_b[2] * hpf_x[2] - hpf_a[1] * hpf_y[1] - hpf_a[2] * hpf_y[2];

    hpf_x[2] = hpf_x[1];
    hpf_x[1] = hpf_x[0];
    hpf_y[2] = hpf_y[1];
    hpf_y[1] = hpf_y[0];

    x[0] = hpf_y[0];
    y[0] = b[0] * x[0] + b[1] * x[1] + b[2] * x[2] - a[1] * y[1] - a[2] * y[2];

    x[2] = x[1];
    x[1] = x[0];
    y[2] = y[1];
    y[1] = y[0];

    int16_t filtered = (int16_t)y[0];

    // --- calibration collection ---
    if (isCalibrating)
    {
      vad_collect_calibration_sample(&vad, filtered);
      totalSamplesRecorded++;

      if (totalSamplesRecorded >= (size_t)CALIBRATION_SAMPLES)
      {
        esp_timer_stop(periodic_timer);
        isCalibrating = false;
        vad_compute_calibration_threshold(&vad);

        Serial.printf("Calibration Done. Threshold: %.3f\n", vad.threshold);
        showStatus("Ready");

        // Reset counters after calibration
        totalSamplesRecorded = 0;
      }
      return;
    }

    // --- if not recording, ignore samples ---
    if (!recording)
      return;

    // --- buffering + SD write ---
    portENTER_CRITICAL(&timerMux);

    buffer[bufferIndex++] = filtered;

    if (totalSamplesRecorded < (size_t)TOTAL_SAMPLES && fullBuffer)
    {
      fullBuffer[totalSamplesRecorded] = filtered;
    }
    totalSamplesRecorded++;

    if (bufferIndex >= (size_t)BUFFER_SIZE)
    {
      bufferReadyToWrite = true;
      bufferIndex = 0;
    }

    portEXIT_CRITICAL(&timerMux);

    // write chunks to SD (outside critical)
    if (bufferReadyToWrite && audioFile)
    {
      portENTER_CRITICAL(&timerMux);
      bufferReadyToWrite = false;
      portEXIT_CRITICAL(&timerMux);

      audioFile.write((uint8_t *)buffer, BUFFER_SIZE * sizeof(int16_t));
    }

    // --- stop condition ---
    if (totalSamplesRecorded >= (size_t)TOTAL_SAMPLES)
    {
      esp_timer_stop(periodic_timer);
      recording = false;

      // flush last partial buffer if any (bufferIndex already reset to 0 on full blocks)

      stopRecording();
      return;
    }
  }
}

void showStatus(const char *msg)
{
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(2);
  tft.setCursor(10, 50);
  tft.println(msg);
}

String getUniqueFilename()
{
  int fileIndex = getLastFileIndex();
  String name = "/Test190126Module_" + String(fileIndex) + ".wav";
  saveLastFileIndex(fileIndex + 1);
  return name;
}

void writeWavHeader(File file, uint32_t dataSize)
{
  uint32_t fileSize = dataSize + 44 - 8;
  uint32_t fmtChunkSize = 16;
  uint16_t audioFormat = 1;
  uint16_t numChannels = 1;
  uint32_t sampleRate = SAMPLE_RATE;
  uint32_t byteRate = sampleRate * numChannels * 2;
  uint16_t blockAlign = numChannels * 2;
  uint16_t bitsPerSample = 16;

  file.write((const uint8_t *)"RIFF", 4);
  file.write((uint8_t *)&fileSize, 4);
  file.write((const uint8_t *)"WAVE", 4);
  file.write((const uint8_t *)"fmt ", 4);
  file.write((uint8_t *)&fmtChunkSize, 4);
  file.write((uint8_t *)&audioFormat, 2);
  file.write((uint8_t *)&numChannels, 2);
  file.write((uint8_t *)&sampleRate, 4);
  file.write((uint8_t *)&byteRate, 4);
  file.write((uint8_t *)&blockAlign, 2);
  file.write((uint8_t *)&bitsPerSample, 2);
  file.write((const uint8_t *)"data", 4);
  file.write((uint8_t *)&dataSize, 4);
  file.flush();
}