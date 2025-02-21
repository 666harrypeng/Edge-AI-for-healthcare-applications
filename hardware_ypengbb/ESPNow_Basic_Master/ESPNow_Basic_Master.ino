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

/* Sensor List
blood oxygen 	-> MAX30102
heart rate 		-> MAX30102
temperature 	-> AHT10
humidity 		-> AHT10
pressure 		-> BMP180
normal audio 	-> KY-038
*/
//////////////////////////////////// Global Variables ////////////////////////////////////

// ESP-NOW transmit data definition
#define BUFFER_SIZE 10 // cannot be too large, esp_now_send() has a limit on the payload size
typedef struct MasterTX
{
	// Arrays to store multiple readings
	// const int buffer_size = 32;
	float temp_buffer[BUFFER_SIZE];
	float hum_buffer[BUFFER_SIZE];
	float pressure_buffer[BUFFER_SIZE];
	float blood_oxygen_buffer[BUFFER_SIZE];
	float heart_rate_buffer[BUFFER_SIZE];
	int16_t audio_buffer[BUFFER_SIZE];
	uint8_t samples; // Number of samples in each buffer

	// float temp_buffer[10];
	// float hum_buffer[10];
	// float pressure_buffer[10];
	// float blood_oxygen_buffer[10];
	// float heart_rate_buffer[10];
	// int16_t audio_buffer[10];
	// uint8_t samples; // Number of samples in each buffer

	void initialize_tx_msg_data()
	{
		samples = 0;
		memset(temp_buffer, 0, sizeof(temp_buffer));
		memset(hum_buffer, 0, sizeof(hum_buffer));
		memset(pressure_buffer, 0, sizeof(pressure_buffer));
		memset(blood_oxygen_buffer, 0, sizeof(blood_oxygen_buffer));
		memset(heart_rate_buffer, 0, sizeof(heart_rate_buffer));
		memset(audio_buffer, 0, sizeof(audio_buffer));
	}
};
struct MasterTX TXMsg;

// Temperature/Humidity Sensor (AHT10)
Adafruit_AHTX0 temp_hum_aht10;
sensors_event_t aht10Temp, aht10Hum;

// Pressure Sensor (BMP180)
Adafruit_BMP085 pressure_bmp;

// Blood Oxygen Sensor (MAX30102)
MAX30105 particleSensor;
/* Followings are the default values, feel free to finetune */
byte ledBrightness = 180; // suggested=127, Options: 0=Off to 255=50mA
byte sampleAverage = 2;	  // Options: 1, 2, 4, 8, 16, 32 (after exp: 1 or 2)
byte ledMode = 2;		  // Options: 1 = Red only(heart beat), 2 = Red + IR(blood oxygen)
int sampleRate = 1600;	  // Options: 50, 100, 200, 400, 800, 1000, 1600, 3200
int pulseWidth = 411;	  // Options: 69, 118, 215, 411
int adcRange = 16384;	  // Options: 2048, 4096, 8192, 16384

/* variables for heart beat */
const byte RATE_SIZE = 4; // use <RATE_SIZE> samples for averaging
byte rates[RATE_SIZE];	  // list of heart beat rates records
byte rateSpot = 0;
long lastBeat = 0; // Time at which the last beat occurred
long thisBeat = 0; // Time at which the current beat occurred
long beat_gap = 0; // Time gap between this beat and last beat
float beatsPerMinute = 0.0;
int beatAvg = 0;

/* variables for blood oxygen */
double avered = 0.0;
double aveir = 0.0;
double sumirrms = 0.0;
double sumredrms = 0.0;
double SpO2 = 0.0;
double ESpO2 = 0.0;	 // initial value
double FSpO2 = 0.7;	 // filter factor for estimated SpO2
double frate = 0.95; // low pass filter for IR/red LED value to eliminate AC component
int sample_count = 0;
int total_samples_batch = 10; // sample <total_samples_batch> (default 30) times to calculate blood oxygen 1 time
#define FINGER_ON 12000		  // IR_LED MIN value -> to determine whether the finger is on the sensor
#define MINIMUM_SPO2 60.0	  // blood oxygen minimum
#define A 1.6				  // Quadratic Approximation Constants (for blood oxygen)
#define B -34.66			  // Quadratic Approximation Constants (for blood oxygen)
#define C 112.7				  // Quadratic Approximation Constants (for blood oxygen)

// Global copy of slave
esp_now_peer_info_t slave;
#define CHANNEL 3 // 1-11
#define PRINTSCANRESULTS 0
#define DELETEBEFOREPAIR 0
bool isPaired;

// Add these variables for signal filtering
#define FILTER_SAMPLES 5
long irBuffer[FILTER_SAMPLES];
byte filterIndex = 0;

// Add these definitions for audio sampling
#define AUDIO_PIN 34 // GPIO34/ADC1_CH6
// #define AUDIO_BUFFER_SIZE 32
#define SAMPLE_INTERVAL_MS 1 // 1ms between samples (approx 1kHz)

// Remove timer-related variables
unsigned long lastSampleTime = 0; // For tracking sample timing

////////////////////////////////////////////////////////////////////////

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

// Scan for slaves in AP mode
void ScanForSlave()
{
	int16_t scanResults = WiFi.scanNetworks(false, false, false, 300, CHANNEL); // Scan only on one channel
	// reset on each scan
	bool slaveFound = 0;
	memset(&slave, 0, sizeof(slave));

	Serial.println("");
	if (scanResults == 0)
	{
		Serial.println("No WiFi devices in AP Mode found");
	}
	else
	{
		Serial.print("Found ");
		Serial.print(scanResults);
		Serial.println(" devices ");
		for (int i = 0; i < scanResults; ++i)
		{
			// Print SSID and RSSI for each device found
			String SSID = WiFi.SSID(i);
			int32_t RSSI = WiFi.RSSI(i);
			String BSSIDstr = WiFi.BSSIDstr(i);

			if (PRINTSCANRESULTS)
			{
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
			if (SSID.indexOf("Slave") == 0)
			{
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
				if (6 == sscanf(BSSIDstr.c_str(), "%x:%x:%x:%x:%x:%x", &mac[0], &mac[1], &mac[2], &mac[3], &mac[4], &mac[5]))
				{
					for (int ii = 0; ii < 6; ++ii)
					{
						slave.peer_addr[ii] = (uint8_t)mac[ii];
					}
				}

				slave.channel = CHANNEL; // pick a channel
				slave.encrypt = 0;		 // no encryption

				slaveFound = 1;
				// we are planning to have only one slave in this example;
				// Hence, break after we find one, to be a bit efficient
				break;
			}
		}
	}

	if (slaveFound)
	{
		Serial.println("Slave Found, processing..");
	}
	else
	{
		Serial.println("Slave Not Found, trying again.");
	}

	// clean up ram
	WiFi.scanDelete();
}

// Check if the slave is already paired with the master.
// If not, pair the slave with master
bool manageSlave()
{
	if (slave.channel == CHANNEL)
	{
		if (DELETEBEFOREPAIR)
		{
			deletePeer();
		}

		Serial.print("Slave Status: ");
		// check if the peer exists
		bool exists = esp_now_is_peer_exist(slave.peer_addr);
		if (exists)
		{
			// Slave already paired.
			Serial.println("Already Paired");
			return true;
		}
		else
		{
			// Slave not paired, attempt pair
			esp_err_t addStatus = esp_now_add_peer(&slave);
			if (addStatus == ESP_OK)
			{
				// Pair success
				Serial.println("Pair success");
				return true;
			}
			else if (addStatus == ESP_ERR_ESPNOW_NOT_INIT)
			{
				Serial.println("ESPNOW Not Init");
				return false;
			}
			else if (addStatus == ESP_ERR_ESPNOW_ARG)
			{
				Serial.print("addStatus: ");
				Serial.println("Invalid Argument");
				return false;
			}
			else if (addStatus == ESP_ERR_ESPNOW_FULL)
			{
				Serial.println("Peer list full");
				return false;
			}
			else if (addStatus == ESP_ERR_ESPNOW_NO_MEM)
			{
				Serial.println("Out of memory");
				return false;
			}
			else if (addStatus == ESP_ERR_ESPNOW_EXIST)
			{
				Serial.println("Peer Exists");
				return true;
			}
			else
			{
				Serial.println("Not sure what happened");
				return false;
			}
		}
	}
	else
	{
		// No slave found to process
		Serial.println("No Slave found to process");
		return false;
	}
}

void deletePeer()
{
	esp_err_t delStatus = esp_now_del_peer(slave.peer_addr);
	Serial.print("Slave Delete Status: ");
	if (delStatus == ESP_OK)
	{
		// Delete success
		Serial.println("Delete Peer Success");
	}
	else if (delStatus == ESP_ERR_ESPNOW_NOT_INIT)
	{
		// How did we get so far!!
		Serial.println("ESPNOW Not Init");
	}
	else if (delStatus == ESP_ERR_ESPNOW_ARG)
	{
		Serial.println("Invalid Argument");
	}
	else if (delStatus == ESP_ERR_ESPNOW_NOT_FOUND)
	{
		Serial.println("Peer not found.");
	}
	else
	{
		Serial.println("Not sure what happened");
	}
}

void sendData()
{
	const uint8_t *peer_addr = slave.peer_addr;
	esp_err_t result = esp_now_send(peer_addr, (uint8_t *)&TXMsg, sizeof(TXMsg));
}

// callback when data is sent from Master to Slave
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status)
{
	char macStr[18];
	snprintf(macStr, sizeof(macStr), "%02x:%02x:%02x:%02x:%02x:%02x",
			 mac_addr[0], mac_addr[1], mac_addr[2], mac_addr[3], mac_addr[4], mac_addr[5]);
	Serial.print("Packet Sent to: ");
	Serial.println(macStr);
}

/////////////////////////////////////// setup ///////////////////////////////////////////////////////
void setup()
{
	Serial.begin(115200);
	delay(10);

	/////////////////////////////////////// Master WIFI Setup ///////////////////////////////////////////////////////
	// Set device in STA (station) mode to begin with
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

	// Once ESPNow is successfully Init,
	// We will register for Send CB to get the status of Trasnmitted packet
	esp_now_register_send_cb(OnDataSent);

	// Initialize TXMsg values
	TXMsg.initialize_tx_msg_data();

	// scan for slave
	ScanForSlave();

	/////////////////////////////////////// Check Slave Connection ///////////////////////////////////////////////////////
	// If Slave is found, it would be populate in `slave` variable. We will check if `slave` is defined and then we proceed further
	if (slave.channel == CHANNEL)
	{ // check if slave channel is defined
		// slave is defined
		// Add slave as peer if it has not been added already
		isPaired = manageSlave();
		Serial.print("Whether Slave has been paired: ");
		Serial.println(isPaired);
	}

	/////////////////////////////////////// Check Temp/Hum Sensor (AHT10) ///////////////////////////////////////////////////////
	if (!temp_hum_aht10.begin())
	{
		Serial.println(F("Temp/Hum Sensor (AHT10) could NOT find. Check wiring ------ "));
		while (1)
			delay(10);
	}
	else
	{
		Serial.println(F("Temp/Hum Sensor (AHT10) found"));
	}

	/////////////////////////////////////// Check Pressure Sensor (BMP180) ///////////////////////////////////////////////////////
	if (!pressure_bmp.begin())
	{
		Serial.println(F("Pressure Sensor (BMP180) could NOT find. Check wiring ------ "));
		while (1)
			delay(10);
	}
	else
	{
		Serial.println(F("Pressure Sensor (BMP180) found"));
	}

	/////////////////////////////////////// Check Blood Oxygen Sensor (MAX30102) ///////////////////////////////////////////////////////
	if (!particleSensor.begin(Wire, I2C_SPEED_FAST))
	{
		Serial.println("Blood Oxygen Sensor (MAX30102) could NOT find. Check wiring ------ ");
		while (1)
			delay(10);
	}
	else
	{
		Serial.println("Blood Oxygen Sensor (MAX30102) found");

		// Configure particle sensor with pre-defined settings (global variables)
		particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
		particleSensor.enableDIETEMPRDY();
		particleSensor.setPulseAmplitudeRed(0x0A); // Turn Red LED to low to indicate sensor is running
		particleSensor.setPulseAmplitudeGreen(0);  // Turn off Green LED

		init_heart_rate_records();

		// Initialize filter buffer
		for (int i = 0; i < FILTER_SAMPLES; i++)
		{
			irBuffer[i] = 0;
		}

		Serial.println("Blood Oxygen Sensor (MAX30102) configured");
	}

	/////////////////////////////////////// Setup ADC for audio (KY-038) ///////////////////////////////////////////////////////
	analogReadResolution(12);		// Sets the ADC resolution to 12 bits (0–4095).
	analogSetAttenuation(ADC_11db); // Sets the attenuation to 11dB, which is the maximum attenuation for the ADC, allowing it to measure voltages up to approximately 3.3V.
}

void init_heart_rate_records()
{
	for (int i = 0; i < RATE_SIZE; i++)
	{
		rates[i] = 0;
	}

	rateSpot = 0;
	lastBeat = 0;
	thisBeat = 0;
	beat_gap = 0;
	beatsPerMinute = 0.0;
	beatAvg = 0;
}

void init_blood_oxygen_records()
{
	avered = 0.0;
	aveir = 0.0;
	sumirrms = 0.0;
	sumredrms = 0.0;
	SpO2 = 0.0;
	ESpO2 = 0.0;
	// sample_count = 0;
}

void get_temp_hum_values()
{
	temp_hum_aht10.getEvent(&aht10Hum, &aht10Temp);
	TXMsg.temp_buffer[TXMsg.samples] = aht10Temp.temperature;
	TXMsg.hum_buffer[TXMsg.samples] = aht10Hum.relative_humidity;
	// TXMsg.samples++;

	// // for debug
	// Serial.print(F("Temp: "));
	// Serial.print(TXMsg.temper);
	// Serial.print(F(" C, Hum: "));
	// Serial.print(TXMsg.hum);
	// Serial.println(F(" %"));
}

void get_pressure_values()
{
	TXMsg.pressure_buffer[TXMsg.samples] = pressure_bmp.readPressure();
	// TXMsg.samples++;

	// // for debug
	// Serial.print(F("Pressure: "));
	// Serial.print(TXMsg.pressure);
	// Serial.println(F(" Pa"));
}

void get_heart_rate_values()
{
	long irValue = particleSensor.getIR();
	// Serial.println("irValue: " + String(irValue));

	// // Apply moving average filter
	// irBuffer[filterIndex] = irValue;
	// filterIndex = (filterIndex + 1) % FILTER_SAMPLES;

	// long filteredIR = 0;
	// for(int i = 0; i < FILTER_SAMPLES; i++) {
	// 	filteredIR += irBuffer[i];
	// }
	// filteredIR /= FILTER_SAMPLES;

	// Serial.println("Raw IR=" + String(irValue) + ", Filtered IR=" + String(filteredIR));

	// if (filteredIR > FINGER_ON)
	if (irValue > FINGER_ON)
	{
		// Serial.println("Finger on sensor >>> checkForBeat(irValue)=" + String(checkForBeat(filteredIR)));
		// Serial.println("Finger on sensor >>> checkForBeat(irValue)=" + String(checkForBeat(irValue)));

		// if (checkForBeat(filteredIR) == true) // Returns true if a beat is detected
		if (checkForBeat(irValue) == true) // Returns true if a beat is detected
		{
			beat_gap = millis() - lastBeat;
			lastBeat = millis();
			// Serial.println("beat_gap=" + String(beat_gap));
			beatsPerMinute = 60 / (beat_gap / 1000.0); // get the average heart rate
			// Serial.println("beatsPerMinute=" + String(beatsPerMinute));
			if (beatsPerMinute < 255 && beatsPerMinute > 20) // constrain the heart rate to 20-255
			{
				// Serial.println("beatsPerMinute=" + String(beatsPerMinute) + " is in the range of 20-255");
				rates[rateSpot++] = (byte)beatsPerMinute;
				rateSpot %= RATE_SIZE;
				beatAvg = 0;
				for (byte x = 0; x < RATE_SIZE; x++)
					beatAvg += rates[x];
				beatAvg /= RATE_SIZE;

				// Serial.println("after update beatAvg(heart rate) = " + String(beatAvg));
			}
		}

		TXMsg.heart_rate_buffer[TXMsg.samples] = beatAvg;
		// TXMsg.samples++;

		// // for debug
		// Serial.print(F("Current HeartRate: "));
		// Serial.println(TXMsg.heart_rate);
	}
	else // no finger on the sensor
	{
		init_heart_rate_records();
		// TXMsg.heart_rate = 0.0f;

		// Serial.println("no finger on the sensor");
	}
}

void get_blood_oxygen_values()
{
	long irValue = particleSensor.getIR(); // Reading the most recent IR value -> it will permit us to know if there's a finger on the sensor or not

	// Serial.println("irValue=" + String(irValue));

	if (irValue > FINGER_ON)
	{
		uint32_t ir, red;
		double fred, fir;
		particleSensor.check(); // Check the sensor, read up to 3 samples
		if (particleSensor.available())
		{
			// Serial.println("particleSensor.available()");
			sample_count++;
			ir = particleSensor.getFIFOIR();   // get the IR value
			red = particleSensor.getFIFORed(); // get the Red value

			// Serial.println("Red=" + String(red) + ", IR=" + String(ir) + ", Sample Count=" + String(sample_count));

			fir = (double)ir;
			fred = (double)red;

			aveir = aveir * frate + fir * (1.0 - frate);	// average IR level by low pass filter
			avered = avered * frate + fred * (1.0 - frate); // average red level by low pass filter
			sumirrms += (fir - aveir) * (fir - aveir);		// square sum of alternate component of IR level
			sumredrms += (fred - avered) * (fred - avered); // square sum of alternate component of red level

			// Serial.println("avered=" + String(avered) + ", aveir=" + String(aveir));
			// Serial.println("sumirrms=" + String(sumirrms) + ", sumredrms=" + String(sumredrms));

			if ((sample_count % total_samples_batch) == 0)
			{
				/* Method 1: Original Linear Approximation & Artificial Clipping */
				// double R = (sqrt(sumirrms) / aveir) / (sqrt(sumredrms) / avered);
				// SpO2 = -23.3 * (R - 0.4) + 120;
				// ESpO2 = FSpO2 * ESpO2 + (1.0 - FSpO2) * SpO2; // low pass filter
				// if (ESpO2 <= MINIMUM_SPO2)
				// 	ESpO2 = MINIMUM_SPO2; // indicator for finger detached
				// if (ESpO2 > 100)
				// 	ESpO2 = 99.9;

				// Serial.print("\tSpO2=");
				// Serial.println(ESpO2);

				// /* Method 2: Quadratic Approximation (without artificial clipping) */
				// /* The relationship between R and SpO2 is typically inverse (higher R = lower SpO2) */
				double R = (sqrt(sumirrms) / aveir) / (sqrt(sumredrms) / avered);
				// Second-order relationship better matches actual SpO2 curve
				SpO2 = A * R * R + B * R + C;
				// Apply moving average filter instead of hard clipping
				ESpO2 = FSpO2 * ESpO2 + (1.0 - FSpO2) * SpO2;
				// if (ESpO2 <= MINIMUM_SPO2)
				// 	ESpO2 = MINIMUM_SPO2; // indicator for finger detached
				// if (ESpO2 > 100)
				// 	ESpO2 = 99.9;

				// // Serial.print("\tESpO2=");
				// // Serial.println(ESpO2);

				// Serial.println("R=" + String(R) + ", SpO2=" + String(SpO2) + ", ESpO2=" + String(ESpO2));
				// Reset accumulators
				sumirrms = 0.0;
				sumredrms = 0.0;
				sample_count = 0;

				TXMsg.blood_oxygen_buffer[TXMsg.samples] = ESpO2;
				// TXMsg.samples++;

				// Serial.println("Current Blood Oxygen: " + String(TXMsg.blood_oxygen) + " %");
			}

			particleSensor.nextSample();
		}
	}
	else // no finger on the sensor
	{
		init_blood_oxygen_records();
		TXMsg.blood_oxygen_buffer[TXMsg.samples] = 0.0f;
		// TXMsg.samples++;

		// Serial.println("no finger on the sensor");
	}
}

void get_heartrate_and_bloodoxygen_values()
{
	long irValue = particleSensor.getIR();
	// Serial.println("irValue: " + String(irValue));

	// // Apply moving average filter
	// irBuffer[filterIndex] = irValue;
	// filterIndex = (filterIndex + 1) % FILTER_SAMPLES;

	// long filteredIR = 0;
	// for(int i = 0; i < FILTER_SAMPLES; i++) {
	// 	filteredIR += irBuffer[i];
	// }
	// filteredIR /= FILTER_SAMPLES;

	// Serial.println("Raw IR=" + String(irValue) + ", Filtered IR=" + String(filteredIR));

	// if (filteredIR > FINGER_ON)
	if (irValue > FINGER_ON)
	{
		// Serial.println("Finger on sensor >>> checkForBeat(irValue)=" + String(checkForBeat(filteredIR)));
		// Serial.println("Finger on sensor >>> checkForBeat(irValue)=" + String(checkForBeat(irValue)));

		// if (checkForBeat(filteredIR) == true) // Returns true if a beat is detected
		if (checkForBeat(irValue) == true) // Returns true if a beat is detected
		{
			beat_gap = millis() - lastBeat;
			lastBeat = millis();
			// Serial.println("beat_gap=" + String(beat_gap));
			beatsPerMinute = 60 / (beat_gap / 1000.0); // get the average heart rate
			// Serial.println("beatsPerMinute=" + String(beatsPerMinute));
			if (beatsPerMinute < 255 && beatsPerMinute > 20) // constrain the heart rate to 20-255
			{
				// Serial.println("beatsPerMinute=" + String(beatsPerMinute) + " is in the range of 20-255");
				rates[rateSpot++] = (byte)beatsPerMinute;
				rateSpot %= RATE_SIZE;
				beatAvg = 0;
				for (byte x = 0; x < RATE_SIZE; x++)
					beatAvg += rates[x];
				beatAvg /= RATE_SIZE;

				// Serial.println("after update beatAvg(heart rate) = " + String(beatAvg));
			}
		}

		TXMsg.heart_rate_buffer[TXMsg.samples] = beatAvg;
		// TXMsg.samples++;

		// // for debug
		// Serial.print(F("Current HeartRate: "));
		// Serial.println(TXMsg.heart_rate);

		// ------------------------------------- Integrate Blood Oxygen Sensor Detection ------------------------
		uint32_t ir, red;
		double fred, fir;
		particleSensor.check(); // Check the sensor, read up to 3 samples
		if (particleSensor.available())
		{
			// Serial.println("particleSensor.available()");
			sample_count++;
			ir = particleSensor.getFIFOIR();   // get the IR value
			red = particleSensor.getFIFORed(); // get the Red value

			// Serial.println("Red=" + String(red) + ", IR=" + String(ir) + ", Sample Count=" + String(sample_count));

			fir = (double)ir;
			fred = (double)red;

			aveir = aveir * frate + fir * (1.0 - frate);	// average IR level by low pass filter
			avered = avered * frate + fred * (1.0 - frate); // average red level by low pass filter
			sumirrms += (fir - aveir) * (fir - aveir);		// square sum of alternate component of IR level
			sumredrms += (fred - avered) * (fred - avered); // square sum of alternate component of red level

			// Serial.println("avered=" + String(avered) + ", aveir=" + String(aveir));
			// Serial.println("sumirrms=" + String(sumirrms) + ", sumredrms=" + String(sumredrms));

			if ((sample_count % total_samples_batch) == 0)
			{
				/* Method 1: Original Linear Approximation & Artificial Clipping */
				// double R = (sqrt(sumirrms) / aveir) / (sqrt(sumredrms) / avered);
				// SpO2 = -23.3 * (R - 0.4) + 120;
				// ESpO2 = FSpO2 * ESpO2 + (1.0 - FSpO2) * SpO2; // low pass filter
				// if (ESpO2 <= MINIMUM_SPO2)
				// 	ESpO2 = MINIMUM_SPO2; // indicator for finger detached
				// if (ESpO2 > 100)
				// 	ESpO2 = 99.9;

				// Serial.print("\tSpO2=");
				// Serial.println(ESpO2);

				// /* Method 2: Quadratic Approximation (without artificial clipping) */
				// /* The relationship between R and SpO2 is typically inverse (higher R = lower SpO2) */
				double R = (sqrt(sumirrms) / aveir) / (sqrt(sumredrms) / avered);
				// Second-order relationship better matches actual SpO2 curve
				SpO2 = A * R * R + B * R + C;
				// Apply moving average filter instead of hard clipping
				ESpO2 = FSpO2 * ESpO2 + (1.0 - FSpO2) * SpO2;

				// // Serial.print("\tESpO2=");
				// // Serial.println(ESpO2);

				// Serial.println("R=" + String(R) + ", SpO2=" + String(SpO2) + ", ESpO2=" + String(ESpO2));
				// Reset accumulators
				sumirrms = 0.0;
				sumredrms = 0.0;
				sample_count = 0;

				TXMsg.blood_oxygen_buffer[TXMsg.samples] = ESpO2;
				// TXMsg.samples++;

				// Serial.println("Current Blood Oxygen: " + String(TXMsg.blood_oxygen) + " %");
			}

			particleSensor.nextSample();
		}
	}
	else // no finger on the sensor
	{
		init_heart_rate_records();
		// TXMsg.heart_rate = 0.0f;

		// Serial.println("no finger on the sensor");

		init_blood_oxygen_records();
		// TXMsg.blood_oxygen = 0.0f;
	}
}

void get_audio_samples()
{
	// int16_t audio_sample = analogRead(AUDIO_PIN) - 2048;
	int16_t audio_sample = analogRead(AUDIO_PIN);
	TXMsg.audio_buffer[TXMsg.samples] = audio_sample;
	// TXMsg.samples++;
}

void loop()
{
	unsigned long currentTime = millis();

	// Collect samples at regular intervals
	if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS)
	{
		if (TXMsg.samples < BUFFER_SIZE)
		{	
			Serial.println("Collecting samples");
			// Get all sensor readings
			get_temp_hum_values();
			// Serial.println("get_temp_hum_values");
			// get_pressure_values();
			// Serial.println("get_pressure_values");
			// get_heartrate_and_bloodoxygen_values();
			// Serial.println("get_heartrate_and_bloodoxygen_values");
			// Get and store audio sample
			get_audio_samples();
			// Serial.println("get_audio_samples");
			TXMsg.samples++;
			lastSampleTime = currentTime;
		}

		// Send when buffer is full
		if (TXMsg.samples >= BUFFER_SIZE)
		{
			Serial.println("---------------- Sending data ----------------");
			sendData();
			// Serial.println("sendData finished");
			// Debug print
			Serial.print("T[0]: " + String(TXMsg.temp_buffer[0]));
			Serial.print(", H[0]: " + String(TXMsg.hum_buffer[0]));
			Serial.print(", P[0]: " + String(TXMsg.pressure_buffer[0]));
			Serial.print(", B[0]: " + String(TXMsg.blood_oxygen_buffer[0]));
			Serial.print(", HR[0]: " + String(TXMsg.heart_rate_buffer[0]));
			Serial.print(", Audio[0]: " + String(TXMsg.audio_buffer[0]));
			Serial.println();

			// Reset audio buffer
			TXMsg.samples = 0;
			// Serial.println("TXMsg.samples = 0");
		}
	}
}
