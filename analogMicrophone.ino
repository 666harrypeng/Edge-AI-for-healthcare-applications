#include <AudioConfig.h>
#include <AudioLogger.h>
#include <AudioTools.h>
#include <WiFi.h>
#include <PubSubClient.h>
/**
 * @file streams-analog-serial.ino
 * @author Phil Schatzmann
 * @brief see https://github.com/pschatzmann/arduino-audio-tools/blob/main/examples/examples-stream/streams-adc-serial/README.md
 * @copyright GPLv3
 * #TODO retest is outstanding
 */

#include "Arduino.h"
#include "AudioTools.h"

#define SIZE 1024
#define N 100

AudioInfo info(96000, 1, 16);
// WhiteNoiseGenerator<int16_t> noise(32000);                        // subclass of SoundGenerator with max amplitude of 32000
AnalogAudioStream in;
// GeneratedSoundStream<int16_t> in_stream(noise);                   // Stream generated from noise
FilteredStream<int16_t, float> filtered(in, info.channels);  // Defiles the filter as BaseConverter
CsvOutput<int16_t> out(Serial); // ASCII output stream 
StreamCopy copier(out, filtered); 

float coef[] = {
    0.000000000000000000,
    0.000001071988036901,
    0.000003761072074216,
    0.000007102582583561,
    0.000009904268076117,
    0.000010756486529060,
    0.000008049418504242,
    0.000000000000000000,
    -0.000015308669299527,
    -0.000039872881432108,
    -0.000075696739251486,
    -0.000124702063597561,
    -0.000188619380307119,
    -0.000268860751616603,
    -0.000366376541151608,
    -0.000481499610929939,
    -0.000613781864767988,
    -0.000761829390579139,
    -0.000923143627361203,
    -0.001093976906663885,
    -0.001269211315707637,
    -0.001442270034670204,
    -0.001605070064652783,
    -0.001748024555847608,
    -0.001860101760190441,
    -0.001928945986059025,
    -0.001941063865646483,
    -0.001882076823533511,
    -0.001737037944514534,
    -0.001490808585314581,
    -0.001128487178253197,
    -0.000635879864214795,
    0.000000000000000001,
    0.000790418658898304,
    0.001744412073947191,
    0.002868297511424331,
    0.004165225315938381,
    0.005634789402736775,
    0.007272714070818130,
    0.009070633057981892,
    0.011015974362003439,
    0.013091961305227984,
    0.015277736719646757,
    0.017548613099743472,
    0.019876447257386937,
    0.022230133579826422,
    0.024576205611490438,
    0.026879531528611313,
    0.029104085323707111,
    0.031213772323349716,
    0.033173285166749930,
    0.034948964688091465,
    0.036509639354810920,
    0.037827417064691629,
    0.038878404206708822,
    0.039643328915587216,
    0.040108048331903859,
    0.040263923316952742,
    0.040108048331903866,
    0.039643328915587216,
    0.038878404206708822,
    0.037827417064691629,
    0.036509639354810913,
    0.034948964688091472,
    0.033173285166749930,
    0.031213772323349726,
    0.029104085323707122,
    0.026879531528611306,
    0.024576205611490442,
    0.022230133579826422,
    0.019876447257386944,
    0.017548613099743478,
    0.015277736719646757,
    0.013091961305227988,
    0.011015974362003446,
    0.009070633057981895,
    0.007272714070818135,
    0.005634789402736775,
    0.004165225315938384,
    0.002868297511424331,
    0.001744412073947192,
    0.000790418658898304,
    0.000000000000000001,
    -0.000635879864214795,
    -0.001128487178253197,
    -0.001490808585314581,
    -0.001737037944514536,
    -0.001882076823533514,
    -0.001941063865646484,
    -0.001928945986059025,
    -0.001860101760190442,
    -0.001748024555847608,
    -0.001605070064652784,
    -0.001442270034670206,
    -0.001269211315707636,
    -0.001093976906663885,
    -0.000923143627361204,
    -0.000761829390579140,
    -0.000613781864767989,
    -0.000481499610929939,
    -0.000366376541151609,
    -0.000268860751616603,
    -0.000188619380307119,
    -0.000124702063597562,
    -0.000075696739251486,
    -0.000039872881432108,
    -0.000015308669299527,
    0.000000000000000000,
    0.000008049418504242,
    0.000010756486529060,
    0.000009904268076118,
    0.000007102582583561,
    0.000003761072074216,
    0.000001071988036901,
    0.000000000000000000
};
// WiFi
const char* ssid = "MSI 8198";    // your network SSID (name)
const char* password = "7269y5H!";    // your network password (use for WPA, or use as key for WEP)
const char* broker = "broker.emqx.io"; 
const char* topic  = "audio.wav";
int        port     = 1883;
WiFiClient espClient;
PubSubClient client(espClient);
void connectWIFI() {
  // attempt to connect to WiFi network:
  Serial.print("Attempting to connect to WPA SSID: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi ..");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print('.');
    delay(1000);
  }

  Serial.println("You're connected to the network");
  Serial.println();  
}
void connectMQTT() {
  // You can provide a unique client ID, if not set the library uses Arduino-millis()
  // Each client must have a unique client ID
 client.setServer(mqtt_broker, mqtt_port);
    client.setCallback(callback);
    while (!client.connected()) {
        String client_id = "esp32-client-";
        client_id += String(WiFi.macAddress());
        Serial.printf("The client %s connects to the public MQTT broker\n", client_id.c_str());
        if (client.connect(client_id.c_str(), mqtt_username, mqtt_password)) {
            Serial.println("Public EMQX MQTT broker connected");
        } else {
            Serial.print("failed with state ");
            Serial.print(client.state());
            delay(2000);
        }
    }
    // Publish and subscribe
    client.publish(topic, "Hi, I'm ESP32 ^^");
    client.subscribe(topic);
}  
void callback() {
    // make sure that we write wav header
    out_stream.begin(info);

    // send message, the Print interface can be used to set the message contents
    client.beginMessage(topic, SIZE * N, true);

    // copy audio data to mqtt: 100 * 1024 bytes
    copier.copyN(N);

    client.endMessage();
}
// Arduino Setup
void setup(void) {
  Serial.begin(115200);
  AudioToolsLogger.begin(Serial, AudioToolsLogLevel::Warning);

  filtered.setFilter(0, new FIR<float>(coef));
  auto cfgRx = in.defaultConfig(RX_MODE);
  // cfgRx.start_pin = A1; // optinally define pin
  // cfgRx.is_auto_center_read = true;
  cfgRx.copyFrom(info);
  in.begin(cfgRx);

  // noise.begin(info);
  // in_stream.begin(info);
  // open output
  out.begin(info);

}

// Arduino loop - copy data 
void loop() {
  copier.copy();  // 
}