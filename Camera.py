# Camera clicks picture and saves it in jpg or png 

from picamera2 import PiCamera2
import time 

picam2 = Picamera2()

#warms up the camera
picam2.start_preview() 
time.sleep(5)

camera.capture("campic.jpg")



