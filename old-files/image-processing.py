from pprint import pprint
from typing import Tuple
from PIL import Image
import numpy as np
import math

fp = "./images/IMG_2748.png"
test = Image.open(fp, mode="r") 

def make_color(rgb, long_name, short_name):
    return {"rgb": rgb, "long_name": long_name, "short_name": short_name}

print("Processing an image")
colors = [
    make_color((0, 0, 0), "Black", "0"),
    make_color((255, 255, 255), "White", "1"), 
    make_color((97, 96, 92), "Wall", "2"),
    make_color((141, 115, 64), "Yellow", "Y"),
    make_color((126, 82, 95), "Pink", "P"),
    make_color((75, 88, 45), "Green", "G"),
]
# colors[(141, 115, 64)] = "Yellow" # Yellow  
# colors[(97, 96, 92)] = "Wall" # Wall
# colors[(126, 82, 95)] = "Pink" # Pink
# colors[(75, 88, 45)] = "Green" # Green

# colors[(0, 0, 0)] = "Black"
# colors[(255, 255, 255)] = "3white"

print(test.mode)
print(test.getpixel((0, 0)))

def remove_brightness(rgba: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    return (rgba[0], rgba[1], rgba[2])

def closest_pixel(input_color: Tuple[int, int, int]) -> dict:
    best_dist = float('inf')
    best_color = None
    for color_dict in colors:
        rgb = color_dict["rgb"]
        dist = math.dist(rgb, input_color)
        if dist < best_dist:
            best_color = color_dict
            best_dist = dist
    return best_color

# pixel = test.getpixel((0,0))
# pixel = remove_brightness(pixel)
# close_pixel = closest_pixel(pixel)
# print(close_pixel)
output_str = []
# for x in range(test.width):
#     output_str.append([])
#     for y in range(test.height):
#         pixel = test.getpixel((x, y))
#         # pixel = (0,0,0,0)
#         color = closest_pixel(remove_brightness(pixel))
#         output_str[x] += color["short_name"]
#     output_str[x] += "\n"
# print("".join(["".join(row) for row in output_str]))
print(np.array(test))