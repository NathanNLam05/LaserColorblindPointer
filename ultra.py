import RPi.GPIO as GPIO
import time

# --- Pin definitions ---
TRIG = 23   # Pin 16 on the Pi
ECHO = 24   # Pin 18 on the Pi

# --- GPIO setup ---
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

def get_distance():
    """Measure distance using HC-SR04 and return meters"""
    # Ensure trigger is low
    GPIO.output(TRIG, False)
    time.sleep(0.05)

    # Send a 10 µs pulse to trigger
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    # Wait for echo to go high then low
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    # Calculate distance (speed of sound: 343 m/s)
    pulse_duration = pulse_end - pulse_start
    distance_m = (pulse_duration * 343) / 2  # divide by 2 for round trip
    distance_m = round(distance_m, 3)        # 3 decimal places (e.g., 0.256 m)
    return distance_m

try:
    print("Starting distance measurement (in meters). Press Ctrl+C to stop.")
    while True:
        dist = get_distance()
        print(f"Distance: {dist} m")
        time.sleep(1)

except KeyboardInterrupt:
    print("\nMeasurement stopped by user")
    GPIO.cleanup()
