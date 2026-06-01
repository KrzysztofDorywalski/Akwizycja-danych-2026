#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>

// OLED SH1106
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SH1106G display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// Czujnik tętna
const int heartPin = A0;
int threshold = 512;
bool pulseDetected = false;

unsigned long lastBeatTime = 0;
int bpm = 0;

// Historia BPM
const int maxHistory = 60;
int bpmHistory[maxHistory] = {0};
int bpmIndex = 0;
int historyCount = 0;

void setup() {
  Serial.begin(9600);

  if (!display.begin(0x3C)) {
    Serial.println("Nie wykryto SH1106!");
    while (1);
  }

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SH110X_WHITE);
  display.setCursor(0, 0);
  display.println("Pomiar tetna...");
  display.display();
  delay(1000);
}

void loop() {
  int signal = analogRead(heartPin);

  if (signal > threshold && !pulseDetected) {
    pulseDetected = true;
    unsigned long now = millis();
    unsigned long delta = now - lastBeatTime;

    if (delta > 300) {
      bpm = 60000 / delta;
      lastBeatTime = now;

      addToHistory(bpm);

      Serial.print("Tętno: ");
      Serial.print(bpm);
      Serial.println(" BPM");

      drawDisplay();
    }
  }

  if (signal < threshold) {
    pulseDetected = false;
  }

  delay(10);
}

void addToHistory(int value) {
  bpmHistory[bpmIndex] = value;
  bpmIndex = (bpmIndex + 1) % maxHistory;
  if (historyCount < maxHistory) historyCount++;
}

int getMinBPM() {
  int minVal = 999;
  for (int i = 0; i < historyCount; i++) {
    if (bpmHistory[i] < minVal) minVal = bpmHistory[i];
  }
  return minVal;
}

int getMaxBPM() {
  int maxVal = 0;
  for (int i = 0; i < historyCount; i++) {
    if (bpmHistory[i] > maxVal) maxVal = bpmHistory[i];
  }
  return maxVal;
}

void drawDisplay() {
  display.clearDisplay();

  // Aktualne BPM
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("BPM: ");
  display.print(bpm);

  // Min / Max
  display.setCursor(70, 0);
  display.print("MIN:");
  display.print(getMinBPM());

  display.setCursor(70, 10);
  display.print("MAX:");
  display.print(getMaxBPM());

  // Wykres
  int start = (bpmIndex + maxHistory - SCREEN_WIDTH) % maxHistory;
  for (int x = 0; x < SCREEN_WIDTH && x < historyCount; x++) {
    int index = (start + x) % maxHistory;
    int barHeight = map(bpmHistory[index], 40, 180, 0, 30);
    barHeight = constrain(barHeight, 0, 30);
    display.drawPixel(x, SCREEN_HEIGHT - barHeight - 1, SH110X_WHITE);
  }

  display.display();
}
