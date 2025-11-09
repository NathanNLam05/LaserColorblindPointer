import RPi.GPIO as GPIO
import time

TRIG = 23
ECHO = 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

def get_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.05)
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    pulse_start, pulse_end = 0, 0
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance_m = (pulse_duration * 343) / 2
    return distance_m

def average_distance(duration):
    readings = []
    start_time = time.time()
    while time.time() - start_time < duration:
        dist = get_distance()
        readings.append(dist)
        time.sleep(0.1)
    if readings:
        return round(sum(readings) / len(readings), 3)
    return None

def stop_sensor():
    GPIO.output(TRIG, False)
    GPIO.cleanup()

# --- Run for a fixed time only ---
try:
    avg = average_distance(3)  # measures for 10 seconds
finally:
    stop_sensor()
