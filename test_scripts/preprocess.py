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
# input_file = 'samples/typed.jpg'
# input_file = 'samples/poem.jpg'
input_file = 'samples/passport.webp'
# input_file = 'samples/skewed_doc.jpg'

opencv_image = cv.imread(input_file)

# Display the imageq
cv.imshow('Initial Image', opencv_image)

# preprocessor = preprocessing.Preprocessor(preprocessing.PreprocessPreset.NORMAL, 10, (0, 0, 0)) # typed
# preprocessor = preprocessing.Preprocessor(preprocessing.PreprocessPreset.NORMAL, 16, (60, 50, 30)) # newspaper
preprocessor = preprocessing.Preprocessor(preprocessing.PreprocessPreset.NORMAL, 26, (230, 200, 110)) # passport
# preprocessor = preprocessing.Preprocessor(preprocessing.PreprocessPreset.NORMAL, 16, (120, 80, 110)) # handwriting
# preprocessor = preprocessing.Preprocessor(preprocessing.PreprocessPreset.NORMAL, 16) # handwriting 2
# preprocessor = preprocessing.Preprocessor(preprocessing.PreprocessPreset.NORMAL, 16) # stone
# preprocessor = preprocessing.Preprocessor(preprocessing.PreprocessPreset.NORMAL, 10, (15, 15, 25)) # poem
final_image = preprocessor.preprocess(opencv_image, True)

cv.imshow('Final Image', final_image)

from PIL import Image
import pytesseract

image = Image.fromarray(final_image)

tessdata_dir_config = r'--tessdata-dir "./tessdata"'

print(pytesseract.image_to_string(image, lang='mya', config=tessdata_dir_config))

while True:
    # Wait for key pressq
    key = cv.waitKey(1) & 0xFF

    # If the 'q' key is pressed, break from the loopq
    if key == ord('q'):
        break