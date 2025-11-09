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
    GPIO.output(TRIG, False)
    time.sleep(0.05)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance_m = (pulse_duration * 343) / 2
    return distance_m

def average_distance(duration=3):
    """Take readings for 'duration' seconds and return average distance"""
    readings = []
    start_time = time.time()
    while time.time() - start_time < duration:
        dist = get_distance()
        readings.append(dist)
        time.sleep(0.1)  # sample about 10 times per second
    if readings:
        return round(sum(readings) / len(readings), 3)
    return None

try:
    print("Measuring distance for 3 seconds...")
    avg_dist = average_distance(3)
    if avg_dist is not None:
        print(f"Average Distance: {avg_dist} m")
    else:
        print("No valid readings.")
finally:
    GPIO.cleanup()
