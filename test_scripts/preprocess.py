import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import cv2 as cv
import preprocessing

# Input file from samples
# input_file = 'samples/newspaper.jpg'
input_file = 'samples/1921f703e5be522b54ba3d532074b8c5.jpg'

opencv_image = cv.imread(input_file)

# Display the image
cv.imshow('Initial Image', opencv_image)

final_image = preprocessing.preprocess(opencv_image, True)

cv.imshow('Final Image', final_image)

while True:
    # Wait for key pressq
    key = cv.waitKey(1) & 0xFF

    # If the 'q' key is pressed, break from the loop
    if key == ord('q'):
        break
