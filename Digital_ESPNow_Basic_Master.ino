#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_AHTX0.h>
#include <Adafruit_BMP085.h>
#include "String.h"
#include "MAX30105.h"
#include "heartRate.h"


Adafruit_AHTX0 aht10;
sensors_event_t aht10Temp, aht10Hum;
Adafruit_BMP085 bmp;
MAX30105 particleSensor;

//計算心跳用變數
const byte RATE_SIZE = 4; //多少平均數量
byte rates[RATE_SIZE]; //心跳陣列
byte rateSpot = 0;
long lastBeat = 0; //Time at which the last beat occurred
float beatsPerMinute;
int beatAvg;

//計算血氧用變數
double avered = 0;
double aveir = 0;
double sumirrms = 0;
double sumredrms = 0;

double SpO2 = 0;
double ESpO2 = 60.0;//初始值
double FSpO2 = 0.7; //filter factor for estimated SpO2
double frate = 0.95; //low pass filter for IR/red LED value to eliminate AC component
int i = 0;
int Num = 30;//取樣30次才計算1次
#define FINGER_ON 7000    //紅外線最小量（判斷手指有沒有上）
#define MINIMUM_SPO2 60.0 //血氧最小量

bool deviceConnected = false;
// ----------------------------------------------------------------

unsigned long currentMillis;
unsigned long previousMillisDisplay = 0;
//ESP-NOW data definition
typedef struct MasterTX {
  float pressure, temper, hum, blood, heartRate;
};
struct MasterTX TXMsg;

// Global copy of slave
esp_now_peer_info_t slave;
#define CHANNEL 13  // 0-20

#define PRINTSCANRESULTS 0
#define DELETEBEFOREPAIR 0

// Status of Port for communication
String Serial0_RX_String = "";
bool Serial0_RX_String_Complete = false;
uint32_t time_got_RX_String = 0;

unsigned long ScanForSlaveTimeStamp;
bool isPaired;


// Init ESP Now with fallback
void InitESPNow() {
  WiFi.disconnect();
  if (esp_now_init() == ESP_OK) {
    Serial.println("ESPNow Init Success");
  } else {
    Serial.println("ESPNow Init Failed");
    ESP.restart();
  }
}

// Scan for slaves in AP mode
void ScanForSlave() {
  int16_t scanResults = WiFi.scanNetworks(false, false, false, 300, CHANNEL);  // Scan only on one channel
  // reset on each scan
  bool slaveFound = 0;
  memset(&slave, 0, sizeof(slave));

  Serial.println("");
  if (scanResults == 0) {
    Serial.println("No WiFi devices in AP Mode found");
  } else {
    Serial.print("Found ");
    Serial.print(scanResults);
    Serial.println(" devices ");
    for (int i = 0; i < scanResults; ++i) {
      // Print SSID and RSSI for each device found
      String SSID = WiFi.SSID(i);
      int32_t RSSI = WiFi.RSSI(i);
      String BSSIDstr = WiFi.BSSIDstr(i);

      if (PRINTSCANRESULTS) {
        Serial.print(i + 1);
        Serial.print(": ");
        Serial.print(SSID);
        Serial.print(" (");
        Serial.print(RSSI);
        Serial.print(")");
        Serial.println("");
      }
      delay(10);
      // Check if the current device starts with `Slave`
      if (SSID.indexOf("Slave") == 0) {
        // SSID of interest
        Serial.println("Found a Slave.");
        Serial.print(i + 1);
        Serial.print(": ");
        Serial.print(SSID);
        Serial.print(" [");
        Serial.print(BSSIDstr);
        Serial.print("]");
        Serial.print(" (");
        Serial.print(RSSI);
        Serial.print(")");
        Serial.println("");
        // Get BSSID => Mac Address of the Slave
        int mac[6];
        if (6 == sscanf(BSSIDstr.c_str(), "%x:%x:%x:%x:%x:%x", &mac[0], &mac[1], &mac[2], &mac[3], &mac[4], &mac[5])) {
          for (int ii = 0; ii < 6; ++ii) {
            slave.peer_addr[ii] = (uint8_t)mac[ii];
          }
        }

        slave.channel = CHANNEL;  // pick a channel
        slave.encrypt = 0;        // no encryption

        slaveFound = 1;
        // we are planning to have only one slave in this example;
        // Hence, break after we find one, to be a bit efficient
        break;
      }
    }
  }

  if (slaveFound) {
    Serial.println("Slave Found, processing..");
  } else {
    Serial.println("Slave Not Found, trying again.");
  }

  // clean up ram
  WiFi.scanDelete();
}

// Check if the slave is already paired with the master.
// If not, pair the slave with master
bool manageSlave() {
  if (slave.channel == CHANNEL) {
    if (DELETEBEFOREPAIR) {
      deletePeer();
    }

    Serial.print("Slave Status: ");
    // check if the peer exists
    bool exists = esp_now_is_peer_exist(slave.peer_addr);
    if (exists) {
      // Slave already paired.
      Serial.println("Already Paired");
      return true;
    } else {
      // Slave not paired, attempt pair
      esp_err_t addStatus = esp_now_add_peer(&slave);
      if (addStatus == ESP_OK) {
        // Pair success
        Serial.println("Pair success");
        return true;
      } else if (addStatus == ESP_ERR_ESPNOW_NOT_INIT) {
        Serial.println("ESPNOW Not Init");
        return false;
      } else if (addStatus == ESP_ERR_ESPNOW_ARG) {
        Serial.println("Invalid Argument");
        return false;
      } else if (addStatus == ESP_ERR_ESPNOW_FULL) {
        Serial.println("Peer list full");
        return false;
      } else if (addStatus == ESP_ERR_ESPNOW_NO_MEM) {
        Serial.println("Out of memory");
        return false;
      } else if (addStatus == ESP_ERR_ESPNOW_EXIST) {
        Serial.println("Peer Exists");
        return true;
      } else {
        Serial.println("Not sure what happened");
        return false;
      }
    }
  } else {
    // No slave found to process
    Serial.println("No Slave found to process");
    return false;
  }
}

void deletePeer() {
  esp_err_t delStatus = esp_now_del_peer(slave.peer_addr);
  Serial.print("Slave Delete Status: ");
  if (delStatus == ESP_OK) {
    // Delete success
    Serial.println("Delete Peer Success");
  } else if (delStatus == ESP_ERR_ESPNOW_NOT_INIT) {
    // How did we get so far!!
    Serial.println("ESPNOW Not Init");
  } else if (delStatus == ESP_ERR_ESPNOW_ARG) {
    Serial.println("Invalid Argument");
  } else if (delStatus == ESP_ERR_ESPNOW_NOT_FOUND) {
    Serial.println("Peer not found.");
  } else {
    Serial.println("Not sure what happened");
  }
}


void sendData() {
  const uint8_t *peer_addr = slave.peer_addr;
  esp_err_t result = esp_now_send(peer_addr, (uint8_t *)&TXMsg, sizeof(TXMsg));
}


// callback when data is sent from Master to Slave
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  char macStr[18];
  snprintf(macStr, sizeof(macStr), "%02x:%02x:%02x:%02x:%02x:%02x",
           mac_addr[0], mac_addr[1], mac_addr[2], mac_addr[3], mac_addr[4], mac_addr[5]);
  Serial.print("Packet Sent to: ");
  Serial.println(macStr);
}



void setup() {
  Serial.begin(115200);
  delay(10);

  //Set device in STA mode to begin with
  WiFi.mode(WIFI_STA);
  esp_wifi_set_channel(CHANNEL, WIFI_SECOND_CHAN_NONE);
  Serial.println("ESPNow/Basic/Master Example");

  // This is the mac address of the Master in Station Mode
  Serial.print("STA MAC: ");
  Serial.println(WiFi.macAddress());
  Serial.print("STA CHANNEL ");
  Serial.println(WiFi.channel());

  // Init ESPNow with a fallback logic
  InitESPNow();

  // Once ESPNow is successfully Init, we will register for Send CB to
  // get the status of Trasnmitted packet
  esp_now_register_send_cb(OnDataSent);

  // scan for slave
  ScanForSlave();

  // If Slave is found, it would be populate in `slave` variable
  // We will check if `slave` is defined and then we proceed further
  if (slave.channel == CHANNEL) {  // check if slave channel is defined
    // slave is defined
    // Add slave as peer if it has not been added already
    isPaired = manageSlave();
    Serial.print("Whether Slave has been paired: ");
    Serial.println(isPaired);
  }
  if (!aht10.begin()) {
    Serial.println(F("Could not find AHT? Check wiring"));
    while (1) delay(10);
  }
  Serial.println(F("AHT10 or AHT20 found"));
  if (!bmp.begin()) {
    Serial.println("Could not find a valid BMP085/BMP180 sensor, check wiring!");
    while (1) {}
  }
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) //Use default I2C port, 400kHz speed
  {
    Serial.println("找不到MAX30102");
    while (1);
  }
  //以下選項可自行調整
  byte ledBrightness = 60; //亮度建議=127, Options: 0=Off to 255=50mA
  byte sampleAverage = 4; //Options: 1, 2, 4, 8, 16, 32
  byte ledMode = 2; //Options: 1 = Red only(心跳), 2 = Red + IR(血氧)
  int sampleRate = 800; //Options: 50, 100, 200, 400, 800, 1000, 1600, 3200
  int pulseWidth = 215; //Options: 69, 118, 215, 411
  int adcRange = 16384; //Options: 2048, 4096, 8192, 16384
  // Set up the wanted parameters
  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange); //Configure sensor with these settings
  particleSensor.enableDIETEMPRDY();

  particleSensor.setPulseAmplitudeRed(0x0A); //Turn Red LED to low to indicate sensor is running
  particleSensor.setPulseAmplitudeGreen(0); //Turn off Green LED
}



void loop() {
  currentMillis = millis();
  if (currentMillis - previousMillisDisplay > 100) {
    getAht10Values();
    TXMsg.pressure = bmp.readPressure();
    // printValues();
    previousMillisDisplay = currentMillis;
  }
  // TXMsg.audio = 500.0 + random(100,200);
  // TXMsg.temper = 19 + random(1,3);
  // TXMsg.hum = 63+ random(3,6);
  // TXMsg.blood = 98+random(1,2);
  TXMsg.blood = ESpO2;
  TXMsg.heartRate = beatAvg;
  long irValue = particleSensor.getIR();    //Reading the IR value it will permit us to know if there's a finger on the sensor or not
  //是否有放手指
  if (irValue > FINGER_ON ) {
    //檢查是否有心跳，測量心跳
    if (checkForBeat(irValue) == true) {
      //Serial.print("Bpm="); Serial.println(beatAvg);//將心跳顯示到序列
      long delta = millis() - lastBeat;//計算心跳差
      lastBeat = millis();
      beatsPerMinute = 60 / (delta / 1000.0);//計算平均心跳
      if (beatsPerMinute < 255 && beatsPerMinute > 20) {
        //心跳必須再20-255之間
        rates[rateSpot++] = (byte)beatsPerMinute; //儲存心跳數值陣列
        rateSpot %= RATE_SIZE;
        beatAvg = 0;//計算平均值
        for (byte x = 0 ; x < RATE_SIZE ; x++) beatAvg += rates[x];
        beatAvg /= RATE_SIZE;
      }
    }

    //測量血氧
    uint32_t ir, red;
    double fred, fir;
    particleSensor.check(); //Check the sensor, read up to 3 samples
    if (particleSensor.available()) {
      i++;
      ir = particleSensor.getFIFOIR(); //讀取紅外線
      red = particleSensor.getFIFORed(); //讀取紅光
      //Serial.println("red=" + String(red) + ",IR=" + String(ir) + ",i=" + String(i));
      fir = (double)ir;//轉double
      fred = (double)red;//轉double
      aveir = aveir * frate + (double)ir * (1.0 - frate); //average IR level by low pass filter
      avered = avered * frate + (double)red * (1.0 - frate);//average red level by low pass filter
      sumirrms += (fir - aveir) * (fir - aveir);//square sum of alternate component of IR level
      sumredrms += (fred - avered) * (fred - avered); //square sum of alternate component of red level

      if ((i % Num) == 0) {
        double R = (sqrt(sumirrms) / aveir) / (sqrt(sumredrms) / avered);
        SpO2 = -23.3 * (R - 0.4) + 120;
        ESpO2 = FSpO2 * ESpO2 + (1.0 - FSpO2) * SpO2;//low pass filter
        if (ESpO2 <= MINIMUM_SPO2) ESpO2 = MINIMUM_SPO2; //indicator for finger detached
        if (ESpO2 > 100) ESpO2 = 99.9;
        //Serial.print(",SPO2="); Serial.println(ESpO2);
        sumredrms = 0.0; sumirrms = 0.0; SpO2 = 0;
        i = 0;
      }
      particleSensor.nextSample(); //We're finished with this sample so move to next sample
    }
    
    // //將數據顯示到序列
    // Serial.print("Bpm:" + String(beatAvg));
    // //顯示血氧數值，避免誤測，規定心跳超過30才能顯示血氧
    // if (beatAvg > 30)  Serial.println(",SPO2:" + String(ESpO2));
    // else Serial.println(",SPO2:" + String(ESpO2));
  }
  //沒偵測到手指，清除所有數據及螢幕內容顯示"Finger Please"
  else {
    //清除心跳數據
    for (byte rx = 0 ; rx < RATE_SIZE ; rx++) rates[rx] = 0;
    beatAvg = 0; rateSpot = 0; lastBeat = 0;
    //清除血氧數據
    avered = 0; aveir = 0; sumirrms = 0; sumredrms = 0;
    SpO2 = 0; ESpO2 = 90.0;
  }
  sendData();
  // delay(100);
}

void getAht10Values() {
  aht10.getEvent(&aht10Hum, &aht10Temp);  // populate temp and humidity objects with fresh data
  TXMsg.temper = aht10Temp.temperature;
  TXMsg.hum = aht10Hum.relative_humidity;
  /*
  Serial.println(F("AHT10 Values"));
  Serial.print(F("Temperature: "));
  Serial.print(temperature);
  Serial.println(F(" *C"));

  Serial.print(F("Humidity: "));
  Serial.print(humidity);
  Serial.println(F(" %"));
  Serial.println();
  */
}
// void printValues() {
//   Serial.print("Temperature = ");
//   Serial.print(temperature);
//   Serial.println(" *C");
//   static char temperatureCTemp[6];
//   dtostrf(temperature, 6, 2, temperatureCTemp);
//   //Set temperature Characteristic value and notify connected client
//   bmeTemperatureCelsiusCharacteristics.setValue(temperatureCTemp);
//   bmeTemperatureCelsiusCharacteristics.notify();

//   // Convert temperature to Fahrenheit
//   /*Serial.print("Temperature = ");
//   Serial.print(1.8 * bme.readTemperature() + 32);
//   Serial.println(" *F");*/

//   Serial.print("Humidity = ");
//   Serial.print(humidity);
//   Serial.println(" %");
//   static char humidityTemp[6];
//   dtostrf(humidity, 6, 2, humidityTemp);
//   //Set humidity Characteristic value and notify connected client
//   bmeHumidityCharacteristics.setValue(humidityTemp);
//   bmeHumidityCharacteristics.notify();
//   Serial.println();
// }