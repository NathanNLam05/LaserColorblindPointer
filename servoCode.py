import RPi.GPIO as GPIO
import time
import math

# Constants
HORIZONTAL_PIN = 17
VERTICAL_PIN = 27
MAX_IMAGE_H = 3280
MAX_IMAGE_R = 2464
HOME_HORIZONTAL = 90
HOME_VERTICAL = 90
TIME_BETWEEN_QUEUE = 1.0  # seconds (use float, not 5000 ms)

horizontal_angles = []
vertical_angles = []

def set_angle(pwm, angle):
    """Safely move a servo to an angle between 0 and 180 degrees."""
    angle = max(0, min(180, angle))
    duty = 2 + (angle / 18)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)
    pwm.ChangeDutyCycle(0)  # reduce buzzing

def QueueCoordToAngle(r, c, dist):
    """Convert image coordinates to servo angles (rough placeholder math)."""
    horizAng = math.degrees(math.atan((c - (MAX_IMAGE_H / 2)) / dist))
    verticAng = math.degrees(math.atan((r - (MAX_IMAGE_R / 2)) / dist))
    horizontal_angles.append(horizAng)
    vertical_angles.append(verticAng)

def appendForTest(num):
    horizontal_angles.append(num)
    vertical_angles.append(num)

def cleanCoordQueue():
    """Clear both angle queues."""
    horizontal_angles.clear()
    vertical_angles.clear()

def runQueue():
    """Run servo queue safely, with setup and cleanup included."""
    try:
        # === GPIO Setup ===
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(HORIZONTAL_PIN, GPIO.OUT)
        GPIO.setup(VERTICAL_PIN, GPIO.OUT)

        pwm_horizontal = GPIO.PWM(HORIZONTAL_PIN, 50)
        pwm_vertical = GPIO.PWM(VERTICAL_PIN, 50)

        pwm_horizontal.start(0)
        pwm_vertical.start(0)

        # === Main Motion Loop ===
        for h_angle, v_angle in zip(horizontal_angles, vertical_angles):
            set_angle(pwm_horizontal, h_angle)
            set_angle(pwm_vertical, v_angle)
            time.sleep(TIME_BETWEEN_QUEUE)

    except KeyboardInterrupt:
        print("Interrupted by user.")
        pass

    finally:
        try:
            # Move servos to home position safely
            set_angle(pwm_horizontal, HOME_HORIZONTAL)
            set_angle(pwm_vertical, HOME_VERTICAL)
            time.sleep(0.5)
        except Exception as e:
            print("Error during final move:", e)

        # Stop PWM
        try:
            pwm_horizontal.stop()
            pwm_vertical.stop()
        except Exception as e:
            print("PWM stop error:", e)

        GPIO.cleanup()
        print("GPIO cleaned up successfully.")


