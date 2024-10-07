import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import cv2 as cv
import preprocessing

# Input file from samples
# input_file = 'samples/handwriting.jpg'
# input_file = 'samples/handwriting2.jpeg'
# input_file = 'samples/stone.jpg'
# input_file = 'samples/newspaper.jpg'
# input_file = 'samples/images.jpeg'
input_file = 'samples/typed.jpg'
# input_file = 'samples/passport.png'
# input_file = 'samples/skewed_doc.jpg'

opencv_image = cv.imread(input_file)

# Display the image
cv.imshow('Initial Image', opencv_image)

preprocessor = preprocessing.Preprocessor(preprocessing.PreprocessPreset.NORMAL, 10, (0, 0, 0)) # typed
# preprocessor = preprocessing.Preprocessor(preprocessing.PreprocessPreset.NORMAL, 16, (65, 40, 30)) # newspaper
# preprocessor = preprocessing.Preprocessor(preprocessing.PreprocessPreset.NORMAL, 10, (230, 200, 110)) # passport
# preprocessor = preprocessing.Preprocessor(preprocessing.PreprocessPreset.NORMAL, 16, (120, 80, 110)) # handwriting
# preprocessor = preprocessing.Preprocessor(preprocessing.PreprocessPreset.NORMAL, 16) # handwriting 2
# preprocessor = preprocessing.Preprocessor(preprocessing.PreprocessPreset.NORMAL, 16) # stone
final_image = preprocessor.preprocess(opencv_image, True)

cv.imshow('Final Image', final_image)

while True:
    # Wait for key pressq
    key = cv.waitKey(1) & 0xFF

    # If the 'q' key is pressed, break from the loopq
    if key == ord('q'):
        break