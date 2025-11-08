import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

HORIZONTAL_PIN = 17   # Servo 1
VERTICAL_PIN = 27     # Servo 2

GPIO.setup(HORIZONTAL_PIN, GPIO.OUT)
GPIO.setup(VERTICAL_PIN, GPIO.OUT)

pwm_horizontal = GPIO.PWM(HORIZONTAL_PIN, 50)
pwm_vertical = GPIO.PWM(VERTICAL_PIN, 50)

pwm_horizontal.start(0)
pwm_vertical.start(0)

def set_angle(pwm, angle):
    duty = 2 + (angle / 18)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)
    pwm.ChangeDutyCycle(0)  # reduce buzzing

# Define your home / reset angles here:
HOME_HORIZONTAL = 90
HOME_VERTICAL = 90

try:
    horizontal_angles = [90, 45, 120, 0]
    vertical_angles   = [30, 60, 90, 45]

    for h_angle, v_angle in zip(horizontal_angles, vertical_angles):
        set_angle(pwm_horizontal, h_angle)
        set_angle(pwm_vertical, v_angle)
        time.sleep(1)

except KeyboardInterrupt:
    pass

finally:
    # *** Reset both servos to home position before exiting ***
    set_angle(pwm_horizontal, HOME_HORIZONTAL)
    set_angle(pwm_vertical, HOME_VERTICAL)
    time.sleep(0.5)

    pwm_horizontal.stop()
    pwm_vertical.stop()
    GPIO.cleanup()












# import RPi.GPIO as GPIO
# import time

# # Setup
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(17, GPIO.OUT)  # Horizontal servo
# GPIO.setup(27, GPIO.OUT)  # Vertical servo

# # Create PWM objects
# pwm_horizontal = GPIO.PWM(17, 50)  # 50Hz frequency
# pwm_vertical = GPIO.PWM(27, 50)    # 50Hz frequency

# # Start PWM with 0 duty cycle (safe position)
# pwm_horizontal.start(0)
# pwm_vertical.start(0)

# def angle_to_duty_cycle(angle):
#     """Convert angle (0-180) to duty cycle (2-12)"""
#     return 2 + (angle / 180) * 10

# def lookatcoord(x, y, img_width=640, img_height=480):
#     """
#     Move servos based on image coordinates
    
#     Args:
#         x: X coordinate in image (0 to img_width)
#         y: Y coordinate in image (0 to img_height)
#         img_width: Width of the image (default 640)
#         img_height: Height of the image (default 480)
#     """
#     # Convert image coordinates to angles (0-180 degrees)
#     # X coordinate maps to horizontal angle (left-right)
#     horizontal_angle = (x / img_width) * 180
    
#     # Y coordinate maps to vertical angle (up-down)
#     # Note: In image coordinates, y=0 is typically top, so we invert
#     vertical_angle = 180 - (y / img_height) * 180
    
#     # Clamp angles to valid range
#     horizontal_angle = max(0, min(180, horizontal_angle))
#     vertical_angle = max(0, min(180, vertical_angle))
    
#     # Convert angles to duty cycles
#     horizontal_duty = angle_to_duty_cycle(horizontal_angle)
#     vertical_duty = angle_to_duty_cycle(vertical_angle)
    
#     # Move servos to new positions
#     pwm_horizontal.ChangeDutyCycle(horizontal_duty)
#     pwm_vertical.ChangeDutyCycle(vertical_duty)
    
#     # Allow time for servos to move
#     time.sleep(0.5)
    
#     # Stop the PWM signals to prevent jitter
#     pwm_horizontal.ChangeDutyCycle(0)
#     pwm_vertical.ChangeDutyCycle(0)
    
#     print(f"Moved to: X={x}, Y={y} -> Horizontal={horizontal_angle:.1f}°, Vertical={vertical_angle:.1f}°")

# # Test the function
# try:
#     while True:
#         # Test positions (assuming 640x480 image)
#         print("Moving to center...")
#         lookatcoord(320, 240)  # Center
#         time.sleep(1)
        
#         print("Moving to top-left...")
#         lookatcoord(0, 0)      # Top-left
#         time.sleep(1)
        
#         print("Moving to bottom-right...")
#         lookatcoord(640, 480)  # Bottom-right
#         time.sleep(1)
        
#         print("Moving to custom position...")
#         lookatcoord(100, 300)  # Custom position
#         time.sleep(1)
    
#     # You can add more test positions here
    
# finally:
#     # Cleanup
#     pwm_horizontal.stop()
#     pwm_vertical.stop()
#     GPIO.cleanup()
#     print("Servos stopped and GPIO cleaned up")




