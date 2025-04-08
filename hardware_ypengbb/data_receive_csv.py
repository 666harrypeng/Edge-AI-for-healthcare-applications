import serial
import csv
import time
import re

# Configuration
SERIAL_PORT = '/dev/cu.usbserial-0001'  # Yiyan Macbook's second port # Replace with actual serial port
BAUD_RATE = 115200
OUTPUT_FILE = './sensor_data.csv'

def parse_data(line):
    """Parse a single line of sensor data"""
    try:
        # Match pattern A:123,T:23.45,H:45.67,P:1013.25,O:98.45,HR:72.00
        pattern = r'A:(-?\d+),T:(\d+\.\d+),H:(\d+\.\d+),P:(\d+\.\d+),O:(\d+\.\d+),HR:(\d+\.\d+)'
        match = re.match(pattern, line)
        
        if match:
            # Convert each value to appropriate type
            audio = int(match.group(1))
            temp = float(match.group(2))
            hum = float(match.group(3))
            press = float(match.group(4))
            oxygen = float(match.group(5))
            hr = float(match.group(6))
            
            return audio, temp, hum, press, oxygen, hr
    except Exception as e:
        print(f"Error parsing data: {e}")
        print(f"Line: {line}")
    return None

def read_serial_data():
    """
    Read data from the ESP32 and save it to a CSV file.
    """
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud")

            # Open the output file for writing
            with open(OUTPUT_FILE, mode='w', newline='') as file:
                writer = csv.writer(file)
                
                # Write CSV header
                writer.writerow([
                    "Timestamp", 
                    "Audio", 
                    "Temperature(C)", 
                    "Humidity(%)", 
                    "Pressure(Pa)",
                    "Oxygen(%)",
                    "HeartRate(BPM)"
                ])

                while True:
                    # Read a line from the serial port
                    line = ser.readline().decode('utf-8').strip()
                    if line:
                        print(f"Raw Data: {line}")  # Debug: Print raw data
                        parsed_data = parse_data(line)

                        if parsed_data:
                            # Each line gets its own timestamp
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Write single row with current timestamp
                            row = [timestamp] + list(parsed_data)
                            writer.writerow(row)
                            file.flush()
                            
                            print(f"Wrote data point: {row}")
                        else:
                            print("Failed to parse data:", line)  # Debug for malformed lines
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        print("Program terminated.")

if __name__ == "__main__":
    read_serial_data()
