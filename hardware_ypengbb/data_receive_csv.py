import serial
import csv
import time
import re
import argparse
import os

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

def read_serial_data(serial_port, baud_rate, output_file):
    """
    Read data from the ESP32 and save it to a CSV file.
    
    Args:
        serial_port: Serial port to connect to
        baud_rate: Baud rate for serial communication
        output_file: Path to save the collected data
    """
    try:
        with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
            print(f"Connected to {serial_port} at {baud_rate} baud")

            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Open the output file for writing
            with open(output_file, mode='w', newline='') as file:
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

                print(f"Data collection started. Saving to {output_file}")
                print("Press Ctrl+C to stop")

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

def main():
    """
    Main function to parse command line arguments and start data collection
    """
    parser = argparse.ArgumentParser(description="Collect sensor data from ESP32")
    parser.add_argument("--port", type=str, default="/dev/cu.usbserial-10",
                        help="Serial port to connect to")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Baud rate for serial communication")
    parser.add_argument("--output", type=str, default="./sensor_data.csv",
                        help="Path to save the collected data")
    
    args = parser.parse_args()
    
    read_serial_data(args.port, args.baud, args.output)

if __name__ == "__main__":
    main()
