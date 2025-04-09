#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include "Wire.h"

/*
Master (DOIT ESP32 - 1) MAC Address -> 94:54:C5:B5:DC:25
Slave (DOIT ESP32 - 2) MAC Address -> 08:A6:F7:B0:7C:91
*/

#define BUFFER_SIZE 10

typedef struct SlaveRX
{
    uint8_t samples;
    float temp_buffer[BUFFER_SIZE];
    float hum_buffer[BUFFER_SIZE];
    // float pressure_buffer[BUFFER_SIZE];
    // float blood_oxygen_buffer[BUFFER_SIZE];
    // float heart_rate_buffer[BUFFER_SIZE];
    int16_t audio_buffer[BUFFER_SIZE];
};
struct SlaveRX RXMsg;

bool ESP_RX_Complete = false;

#define CHANNEL 3

// Init ESP Now with fallback
void InitESPNow()
{
  WiFi.disconnect();
  if (esp_now_init() == ESP_OK)
  {
    Serial.println("ESPNow Init Success");
  }
  else
  {
    Serial.println("ESPNow Init Failed");
    ESP.restart();
  }
}

// config AP SSID
void configDeviceAP()
{
  const char *SSID = "Slave_1";
  bool result = WiFi.softAP(SSID, "Slave_1_Password", CHANNEL, 0);
  if (!result)
  {
    Serial.println("AP Config failed.");
  }
  else
  {
    Serial.println("AP Config Success. Broadcasting with AP: " + String(SSID));
    Serial.print("AP CHANNEL ");
    Serial.println(WiFi.channel());
  }
}

// callback when data is recv from Master
void OnDataRecv(const esp_now_recv_info_t *recv_info, const uint8_t *incomingData, int len)
{
  memcpy(&RXMsg, incomingData, sizeof(RXMsg));

  // Print each sample as a separate line
  for (int i = 0; i < RXMsg.samples; i++)
  {
    Serial.print("A:"); // Audio
    Serial.print(RXMsg.audio_buffer[i]);
    Serial.print(",T:"); // Temperature
    Serial.print(RXMsg.temp_buffer[i], 2);
    Serial.print(",H:"); // Humidity
    Serial.print(RXMsg.hum_buffer[i], 2);
    Serial.print(",P:"); // Pressure
    Serial.print(RXMsg.pressure_buffer[i], 2);
    Serial.print(",O:"); // Oxygen
    Serial.print(RXMsg.blood_oxygen_buffer[i], 2);
    Serial.print(",HR:"); // Heart Rate
    Serial.print(RXMsg.heart_rate_buffer[i], 2);
    Serial.println();
  }

  /* Check the sender's MAC address to check the identity of the sender*/ // debug use
  // char macStr[18];
  // snprintf(macStr, sizeof(macStr), "%02x:%02x:%02x:%02x:%02x:%02x",
  //          recv_info->src_addr[0], recv_info->src_addr[1], recv_info->src_addr[2],
  //          recv_info->src_addr[3], recv_info->src_addr[4], recv_info->src_addr[5]);
  // Serial.print("Last Packet Recv from: "); Serial.println(macStr);

  // Serial.print("Last Packet Recv Data: ");
  // Serial.println(RXMsg);

  // Serial.println("");
}

void setup()
{
  // intialize the serial port
  Serial.begin(115200);
  delay(10);

  // initialize the ESP-NOW communication
  // Set device in AP mode to begin with
  WiFi.mode(WIFI_AP);

  // configure device AP mode
  configDeviceAP();

  // This is the mac address of the Slave in AP Mode
  Serial.print("AP MAC: ");
  Serial.println(WiFi.softAPmacAddress());

  // Init ESPNow with a fallback logic
  InitESPNow();

  // Once ESPNow is successfully Init, we will register for recv CB to
  // get recv packer info.
  esp_now_register_recv_cb(OnDataRecv);
}

void loop()
{
  // delay(1000);
}
