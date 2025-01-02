#include <Wire.h>
#include <Adafruit_Sensor.h>  // https://github.com/adafruit/Adafruit_Sensor
// sensor data from AHT10 module
#include <Adafruit_AHTX0.h> // https://github.com/adafruit/Adafruit_AHTX0
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
Adafruit_AHTX0 aht10;
sensors_event_t aht10Temp, aht10Hum;
#define bleServerName "AHT10_ESP32"
const String SENSOR_NAME = "AHT10 T & H";  // don't extend the string with more than 14 characters !

float temperature = -99;
float humidity = -99;
bool deviceConnected = false;
// ----------------------------------------------------------------
// vars for "delay" without blocking the loop
const unsigned long period1s = 1000;    //the value is a number of milliseconds, ie 1 second
const unsigned long period3s = 3000;    //the value is a number of milliseconds, ie 3 seconds
const unsigned long period10s = 10000;  //the value is a number of milliseconds, ie 10 seconds
const unsigned long period30s = 30000;  //the value is a number of milliseconds, ie 30 seconds
unsigned long currentMillis;
unsigned long previousMillisDisplay = 0;

#define SERVICE_UUID "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
BLECharacteristic bmeTemperatureCelsiusCharacteristics("beb5483e-36e1-4688-b7f5-ea07361b26a8", BLECharacteristic::PROPERTY_NOTIFY);
BLEDescriptor bmeTemperatureCelsiusDescriptor(BLEUUID((uint16_t)0x2902));
BLECharacteristic bmeHumidityCharacteristics("ca73b3ba-39f6-4ab3-91ae-186dc9577d99", BLECharacteristic::PROPERTY_NOTIFY);
BLEDescriptor bmeHumidityDescriptor(BLEUUID((uint16_t)0x2903));
//Setup callbacks onConnect and onDisconnect
class MyServerCallbacks: public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
  };
  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
  }
};
// ----------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  Serial.println(F("Environment Sensor AHT10"));

  // ----------------------------------------------------------------
  
  // ----------------------------------------------------------------
  // setup the sensor
  // AHT10 Temp + Humidity sensor
  if (! aht10.begin()) {
    Serial.println(F("Could not find AHT? Check wiring"));
    while (1) delay(10);
  }
  Serial.println(F("AHT10 or AHT20 found"));
  // Create the BLE Device
  BLEDevice::init(bleServerName);
  String macAddress = BLEDevice::getAddress().toString().c_str();  // Get BLE address
  Serial.println("ESP32 BLE MAC Address: " + macAddress);  // Print the address
  // Create the BLE Server
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());
  // Create the BLE Service
  BLEService *bmeService = pServer->createService(SERVICE_UUID);
  bmeService->addCharacteristic(&bmeTemperatureCelsiusCharacteristics);
  bmeTemperatureCelsiusDescriptor.setValue("BME temperature Celsius");
  bmeTemperatureCelsiusCharacteristics.addDescriptor(&bmeTemperatureCelsiusDescriptor);
  bmeService->addCharacteristic(&bmeHumidityCharacteristics);
  bmeHumidityDescriptor.setValue("BME humidity");
  bmeHumidityCharacteristics.addDescriptor(new BLE2902());
  // Start the service
  bmeService->start();

  // Start advertising
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pServer->getAdvertising()->start();
  Serial.println("Waiting a client connection to notify...");
}


void loop() {

  currentMillis = millis();
  if (currentMillis - previousMillisDisplay > 1000) {
    getAht10Values();
    printValues();
    previousMillisDisplay = currentMillis;
  }
}

void getAht10Values() {
  aht10.getEvent(&aht10Hum, &aht10Temp);// populate temp and humidity objects with fresh data
  temperature = aht10Temp.temperature;
  humidity = aht10Hum.relative_humidity;
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

void printValues() {
  Serial.print("Temperature = ");
  Serial.print(temperature);
  Serial.println(" *C");
  static char temperatureCTemp[6];
  dtostrf(temperature, 6, 2, temperatureCTemp);
  //Set temperature Characteristic value and notify connected client
  bmeTemperatureCelsiusCharacteristics.setValue(temperatureCTemp);
  bmeTemperatureCelsiusCharacteristics.notify();
  
  // Convert temperature to Fahrenheit
  /*Serial.print("Temperature = ");
  Serial.print(1.8 * bme.readTemperature() + 32);
  Serial.println(" *F");*/

  Serial.print("Humidity = ");
  Serial.print(humidity);
  Serial.println(" %");
  static char humidityTemp[6];
  dtostrf(humidity, 6, 2, humidityTemp);
  //Set humidity Characteristic value and notify connected client
  bmeHumidityCharacteristics.setValue(humidityTemp);
  bmeHumidityCharacteristics.notify();   
  Serial.println();
}

// String processor(const String &var) {
//   //Serial.println(var);
//   if (var == "TEMPERATURE") {
//     return String(temperature);
//   } else if (var == "HUMIDITY") {
//     return String(humidity);
//   } else if (var == "PRESSURE") {
//     return "n/a";
//   } else if (var == "SENSORNAME") {
//     return SENSOR_NAME;
//   }
// }
