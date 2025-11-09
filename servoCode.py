import RPi.GPIO as GPIO
import time
import math

# === Constants ===
HORIZONTAL_PIN = 17
VERTICAL_PIN = 27
MAX_IMAGE_H = 3280
MAX_IMAGE_R = 2464
HOME_HORIZONTAL = 90
HOME_VERTICAL = 90
TIME_BETWEEN_QUEUE = 1.0  # seconds (use float, not 5000 ms)
VERTICAL_OFFSET = 3
HORIZONTAL_OFFSET = 2
CAMERA_FOV_H = 62.0  # degrees
CAMERA_FOV_V = 48.8  # degrees

# === Safe servo limits ===
MIN_HORIZONTAL = 60
MAX_HORIZONTAL = 120
MIN_VERTICAL = 90    # only look upward
MAX_VERTICAL = 120   # upper limit

horizontal_angles = []
vertical_angles = []


def set_angle(pwm, angle):
    """Safely move a servo to an angle between 0 and 180 degrees."""
    angle = max(0, min(180, angle))
    duty = 2 + (angle / 18)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)
    pwm.ChangeDutyCycle(0)  # reduce buzzing


def queueCoordToAngle(r, c):
    """
    Convert image coordinates (r, c) into servo angles.
    r = row (vertical pixel position)
    c = column (horizontal pixel position)
    """

    # Compute offset from image center
    x_offset = c - (MAX_IMAGE_H / 2)
    y_offset = r - (MAX_IMAGE_R / 2)

    # Convert pixel offsets to angular offsets
    horizAng = HORIZONTAL_OFFSET + HOME_HORIZONTAL - (x_offset / MAX_IMAGE_H) * CAMERA_FOV_H
    verticAng = VERTICAL_OFFSET + HOME_VERTICAL - (y_offset / MAX_IMAGE_R) * CAMERA_FOV_V  # subtract because y increases downward

    # === Clamp to SAFE servo ranges ===
    horizAng = max(MIN_HORIZONTAL, min(MAX_HORIZONTAL, horizAng))
    verticAng = max(MIN_VERTICAL, min(MAX_VERTICAL, verticAng))

    # Add to the queues
    horizontal_angles.append(horizAng)
    vertical_angles.append(verticAng)


def appendForTest(num):
    horizontal_angles.append(num)
    vertical_angles.append(num)


def cleanCoordQueue():
    """Clear both angle queues."""
    global horizontal_angles, vertical_angles
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


# === Optional: test safe servo range ===
def test_servo_limits():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(HORIZONTAL_PIN, GPIO.OUT)
    GPIO.setup(VERTICAL_PIN, GPIO.OUT)

    pwm_horizontal = GPIO.PWM(HORIZONTAL_PIN, 50)
    pwm_vertical = GPIO.PWM(VERTICAL_PIN, 50)
    pwm_horizontal.start(0)
    pwm_vertical.start(0)

    try:
        print("Testing horizontal servo range...")
        for angle in range(MIN_HORIZONTAL, MAX_HORIZONTAL + 1, 10):
            set_angle(pwm_horizontal, angle)
            time.sleep(0.5)

        print("Testing vertical servo range...")
        for angle in range(MIN_VERTICAL, MAX_VERTICAL + 1, 5):
            set_angle(pwm_vertical, angle)
            time.sleep(0.5)
    finally:
        pwm_horizontal.stop()
        pwm_vertical.stop()
        GPIO.cleanup()
        print("Servo range test complete.")
