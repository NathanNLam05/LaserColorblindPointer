from servoCode import appendForTest, runQueue,QueueCoordToAngle 
from ultra import get_distance, average_distance
from Camera import takePicture
from image_analyzer import ImageAnalyzer

# Example usage

path="/home/pi/Pictures/test.jpg"
takePicture(path)
image_analyzer = ImageAnalyzer(i_file=path)
dist = average_distance

appendForTest(45)

runQueue()
