//#include <Wire.h>
#include <Adafruit_Sensor.h>  // https://github.com/adafruit/Adafruit_Sensor
// sensor data from AHT10 module
#include <Adafruit_AHTX0.h> // https://github.com/adafruit/Adafruit_AHTX0
Adafruit_AHTX0 aht10;
sensors_event_t aht10Temp, aht10Hum;

const String SENSOR_NAME = "AHT10 T & H";  // don't extend the string with more than 14 characters !

float temperature = -99;
float humidity = -99;

// ----------------------------------------------------------------
// vars for "delay" without blocking the loop
const unsigned long period1s = 1000;    //the value is a number of milliseconds, ie 1 second
const unsigned long period3s = 3000;    //the value is a number of milliseconds, ie 3 seconds
const unsigned long period10s = 10000;  //the value is a number of milliseconds, ie 10 seconds
const unsigned long period30s = 30000;  //the value is a number of milliseconds, ie 30 seconds
unsigned long currentMillis;
unsigned long previousMillisDisplay = 0;


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

  // Convert temperature to Fahrenheit
  /*Serial.print("Temperature = ");
  Serial.print(1.8 * bme.readTemperature() + 32);
  Serial.println(" *F");*/

  Serial.print("Humidity = ");
  Serial.print(humidity);
  Serial.println(" %");
  Serial.println();
}

String processor(const String &var) {
  //Serial.println(var);
  if (var == "TEMPERATURE") {
    return String(temperature);
  } else if (var == "HUMIDITY") {
    return String(humidity);
  } else if (var == "PRESSURE") {
    return "n/a";
  } else if (var == "SENSORNAME") {
    return SENSOR_NAME;
  }
}
