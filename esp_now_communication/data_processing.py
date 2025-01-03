import serial
import csv
import time
import re

# Configuration
SERIAL_PORT = '/dev/cu.usbserial-0001'  # Yiyan Macbook's second port # Replace with actual serial port
BAUD_RATE = 115200
OUTPUT_FILE = './sensor_data.csv'

def parse_data(line):
    """
    Parse a line from the serial output to extract the labeled fields.
    """
    # Example input: A:5203515.00,T:203515.00,H:203515.00,B:203535.00
    match = re.match(r"A:(\d+\.\d+),T:(\d+\.\d+),H:(\d+\.\d+),B:(\d+\.\d+)", line)
    if match:
        # Extract the individual values
        a_value = float(match.group(1))
        t_value = float(match.group(2))
        h_value = float(match.group(3))
        b_value = float(match.group(4))
        return a_value, t_value, h_value, b_value
    else:
        return None  # Return None if the format doesn't match

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
                writer.writerow(["Timestamp", "A_Value", "T_Value", "H_Value", "B_Value"])

                while True:
                    # Read a line from the serial port
                    line = ser.readline().decode('utf-8').strip()
                    if line:
                        print(f"Raw Data: {line}")  # Debug: Print raw data
                        parsed_data = parse_data(line)

                        if parsed_data:
                            # Add a timestamp
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                            print(f"Parsed Data: {timestamp}, {parsed_data}")  # Debug: Print parsed data

                            # Write parsed data to CSV
                            writer.writerow([timestamp] + list(parsed_data))
                            file.flush()  # Ensure data is written to the file
                        else:
                            print("Failed to parse data:", line)  # Debug for malformed lines
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        print("Program terminated.")

if __name__ == "__main__":
    read_serial_data()
