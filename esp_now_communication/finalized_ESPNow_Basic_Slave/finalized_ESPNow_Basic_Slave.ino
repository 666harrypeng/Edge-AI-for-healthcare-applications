#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include "Wire.h"

/*
Master (DOIT ESP32 - 1) MAC Address -> 94:54:C5:B5:DC:25
Slave (DOIT ESP32 - 2) MAC Address -> de:54:75:d1:fa:d4
*/


//ESP-NOW data definition
// String RXMsg = "";
typedef struct SlaveRX
{
  float audio, temper, hum, blood;
};

struct SlaveRX RXMsg;;



bool ESP_RX_Complete = false;


#define CHANNEL 13

// Init ESP Now with fallback
void InitESPNow() {
  WiFi.disconnect();
  if (esp_now_init() == ESP_OK) {
    Serial.println("ESPNow Init Success");
  }
  else {
    Serial.println("ESPNow Init Failed");
    ESP.restart();
  }
}

// config AP SSID
void configDeviceAP() {
  const char *SSID = "Slave_1";
  bool result = WiFi.softAP(SSID, "Slave_1_Password", CHANNEL, 0);
  if (!result) {
    Serial.println("AP Config failed.");
  } else {
    Serial.println("AP Config Success. Broadcasting with AP: " + String(SSID));
    Serial.print("AP CHANNEL "); Serial.println(WiFi.channel());
  }
}


// callback when data is recv from Master
void OnDataRecv(const uint8_t *mac_addr, const uint8_t *incommingData, int data_len) {
  memcpy(&RXMsg, incommingData, sizeof(RXMsg));

  Serial.print("A:");
  Serial.print(RXMsg.audio);
  Serial.print(",T:");
  Serial.print(RXMsg.temper);
  Serial.print(",H:");
  Serial.print(RXMsg.hum);
  Serial.print(",B:");
  Serial.print(RXMsg.blood);
  Serial.println();

  char macStr[18];
  snprintf(macStr, sizeof(macStr), "%02x:%02x:%02x:%02x:%02x:%02x",
           mac_addr[0], mac_addr[1], mac_addr[2], mac_addr[3], mac_addr[4], mac_addr[5]);
  // Serial.print("Last Packet Recv from: "); Serial.println(macStr);

  // Serial.print("Last Packet Recv Data: "); 
  // Serial.println(RXMsg);

  // Serial.println("");
}



void setup() {
  // intialize the serial port
  Serial.begin(115200);
  delay(10);

  // initialize the ESP-NOW communication
  //Set device in AP mode to begin with
  WiFi.mode(WIFI_AP);

  // configure device AP mode
  configDeviceAP();

  // This is the mac address of the Slave in AP Mode
  Serial.print("AP MAC: "); Serial.println(WiFi.softAPmacAddress());

  // Init ESPNow with a fallback logic
  InitESPNow();
  
  // Once ESPNow is successfully Init, we will register for recv CB to
  // get recv packer info.
  esp_now_register_recv_cb(OnDataRecv);
}


void loop() {

  // delay(1000);
}


