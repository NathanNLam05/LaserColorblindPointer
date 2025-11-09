from servoCode import appendForTest, runQueue,queueCoordToAngle 
from ultra import get_distance, average_distance
from Camera import takePicture
from image_analyzer import ImageAnalyzer

# Example usage

path="/home/pi/Pictures/test.jpg"
takePicture(path)
image_analyzer = ImageAnalyzer(i_path=path)

coordinates = image_analyzer.get_coordinates()
    
for coord in coordinates:
    # Assuming coord is a tuple (x, y)
    queueCoordToAngle(coord[0], coord[1])
runQueue()
