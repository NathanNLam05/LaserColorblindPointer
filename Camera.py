# Camera clicks picture and saves it in jpg or png 

from picamera2 import Picamera2
import time
import os

# Create camera object
picam2 = Picamera2()

# Configure camera for still image
config = picam2.create_still_configuration()
picam2.configure(config)

# Optional preview (works only if monitor is attached)
picam2.start_preview()

# Start camera
picam2.start()
time.sleep(2)  # give camera time to adjust exposure

# Save image to a folder
save_path = "/home/pi/Pictures/test.jpg"
os.makedirs("/home/pi/Pictures", exist_ok=True)

picam2.capture_file(save_path)

print("Image saved to:", save_path)

picam2.stop()

