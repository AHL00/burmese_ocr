import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class PreprocessPreset(Enum):
    """Preprocessing presets."""

    NORMAL = 1


class Preprocessor:
    def __init__(
        self,
        preset: PreprocessPreset,
        approx_char_size: int = 20,
        text_color: tuple[int, int, int] | None = None,
    ):
        if approx_char_size < 10:
            raise ValueError("Approximate character size must be greater than 9.")

        self.preset = preset
        self.approx_char_size = approx_char_size
        self.text_color = text_color

    def preprocess(self, cv_image: cv.typing.MatLike, debug=False) -> cv.typing.MatLike:
        """Preprocess an image for OCR.

        Args:
            cv_image (cv.typing.MatLike): The image to be preprocessed. MatLike is the type
            returned by cv.imread() and cv.VideoCapture.read().

        Returns:
            cv.typing.MatLike: Proprocessed image.

        Steps performed:
            1. Convert the image to grayscale. [Self.grayscale()]
            2. Apply a threshold to the image. [Self.threshold()]
            3. Remove small islands of noise. [Self.remove_islands()]
        """
        # CV Filter image colors only leaving colors closer to the text color ideally
        color_filtered = None

        if self.text_color is not None:
            color_filtered = self.color_filter(cv_image, debug)

            if debug:
                cv.imshow("Color Filtered", color_filtered)
        else:
            color_filtered = self.grayscale(cv_image, debug)

            if debug:
                cv.imshow("Grayscaled", color_filtered)

        thresholded = self.threshold(color_filtered, debug)

        if debug:
            cv.imshow("Thresholded", thresholded)

        # removed_islands = self.remove_islands(thresholded, debug)

        # if debug:
        # cv.imshow("Removed Islands", removed_islands)

        thresholded_guassian = self.threshold_guassian(thresholded, debug)

        if debug:
            cv.imshow("Thresholded Guassian", thresholded_guassian)

        removed_islands_2 = self.remove_islands(thresholded_guassian, debug)

        if debug:
            cv.imshow("Removed Islands 2", removed_islands_2)

        resized = self.resize(removed_islands_2, self.approx_char_size)

        contours = self.contour(resized, debug)

        return resized

    def color_filter(
        self, cv_image: cv.typing.MatLike, debug=False
    ) -> cv.typing.MatLike:
        if self.text_color is None:
            raise ValueError("Text color must be set to use color filter.")

        # The closer a pixel's color is to the text color, the closer it will be set to black (0, 0, 0).
        # The further away it is, the closer it will be set to white.
        def closeness(
            color1: tuple[int, int, int], color2: tuple[int, int, int]
        ) -> float:
            # https://stackoverflow.com/questions/5392061/algorithm-to-check-similarity-of-colors
            # double ColourDistance(RGB e1, RGB e2)
            # {
            #   long rmean = ( (long)e1.r + (long)e2.r ) / 2;
            #   long r = (long)e1.r - (long)e2.r;
            #   long g = (long)e1.g - (long)e2.g;
            #   long b = (long)e1.b - (long)e2.b;
            #   return sqrt((((512+rmean)*r*r)>>8) + 4*g*g + (((767-rmean)*b*b)>>8));
            # }

            rmean = (color1[0] + color2[0]) / 2
            r = color1[0] - color2[0]
            g = color1[1] - color2[1]
            b = color1[2] - color2[2]
            return 1 - (
                np.sqrt(
                    (((512 + rmean) * r * r) / 256)
                    + 4 * g * g
                    + (((767 - rmean) * b * b) / 256)
                )
                / 1000
            )

        gray_image = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)

        for x in range(gray_image.shape[1]):
            for y in range(gray_image.shape[0]):
                pixel = cv_image[y, x]

                pixel_rgb = (pixel[2], pixel[1], pixel[0])

                closeness_float = closeness(pixel_rgb, self.text_color)

                # Power means that closeness is weighted more heavily. Basically, it
                # is a curve that is more flat when closeness is low and more steep when
                # closeness is high.
                # x^4 {x: 0 -> 1}
                gray_image[y, x] = 255 - (255 * (pow(closeness_float, 4)) // 1)

        return gray_image

    def grayscale(self, cv_image: cv.typing.MatLike, debug=False) -> cv.typing.MatLike:
        return cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)

    def threshold_guassian(
        self, cv_image: cv.typing.MatLike, debug=False
    ) -> cv.typing.MatLike:
        blockSize = self.approx_char_size

        if blockSize % 2 == 0:
            blockSize -= 1

        thresholded = cv.adaptiveThreshold(
            cv_image,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            blockSize,
            16,
        )

        return thresholded

    def threshold(self, cv_image: cv.typing.MatLike, debug=False) -> cv.typing.MatLike:
        # NOTE: According to https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html,
        # adaptive guassian thresholding is the most consistent and reliable method.
        # More research is required and tuning for the "C" and "blockSize" constants is required.

        # Plot a histogram of pixel counts binning by 8
        # plt.hist(cv_image.ravel(), 256, (0, 180))
        # plt.show()

        # Find the first peak, this will be the most common dark color
        # This will be used as the threshold for the non-adaptive thresholding
        hist, bins = np.histogram(cv_image.ravel(), 256, (0, 180))

        # Compute the bin centers
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Compute the mean of the histogram
        mean = np.sum(hist * bin_centers) / np.sum(hist)

        # Compute the variance
        variance = np.sum(hist * (bin_centers - mean) ** 2) / np.sum(hist)

        # Compute the standard deviation
        std_dev = np.sqrt(variance)

        # Find the first peak
        first_peak = np.argmax(hist)

        if debug:
            print(f"First peak: {first_peak}, Standard deviation: {std_dev}")

        # Set the threshold to the first peak
        threshold = first_peak + 1.2 * std_dev

        print(f"Threshold: {threshold}")

        # Remove pixels above the threshold
        thresholded = cv.threshold(cv_image, threshold, 255, cv.THRESH_BINARY)[1]

        return thresholded

    def remove_islands(
        self, cv_image: cv.typing.MatLike, debug=False
    ) -> cv.typing.MatLike:

        # Find contours
        (numLabels, labels, stats, centroids) = cv.connectedComponentsWithStats(
            cv.bitwise_not(cv_image)
        )

        # Parts to be removed are 0.
        remove_mask = np.full(cv_image.shape, 255, dtype=np.uint8)

        # perform statistical analysis and decide on a removal size based on standard deviations
        # Calculate mean and standard deviation of areas
        areas = stats[:, cv.CC_STAT_AREA][1:]  # 1: to Skip the background label

        # A threshold under which to remove islands.
        # This will allow the statistics to do a better job
        # of adapting if the data is not flooded with small islands.
        area_threshold = 2 + ((self.approx_char_size // 8) ** 2)

        if area_threshold <= 4:
            area_threshold = 5

        if debug:
            print(f"Area threshold: {area_threshold}")

        # Filter out the super small islands first
        for i in range(1, numLabels):
            if stats[i, cv.CC_STAT_AREA] < area_threshold:
                remove_mask[labels == i] = 0

        areas = areas[areas >= area_threshold]

        mean_area = np.mean(areas.astype(np.float64))
        std_area = np.std(areas.astype(np.float64))

        if debug:
            print(f"Mean area: {mean_area}, Standard deviation: {std_area}")

        # Width and height removal
        max_1 = self.approx_char_size // 6
        max_2 = self.approx_char_size * 2

        for i in range(1, numLabels):
            if (
                # Remove long horizontal lines
                stats[i, cv.CC_STAT_HEIGHT] <= max_1
                and stats[i, cv.CC_STAT_WIDTH] <= max_2
                # Don't remove dots as they are important
                and stats[i, cv.CC_STAT_WIDTH] > max_1
            ):
                remove_mask[labels == i] = 0

        # If the pixels are the same, set to 255.
        img_clone = cv_image.copy()

        img_clone[remove_mask == 0] = 255

        # Straight line removal if way outside of the standard deviation
        # This removes any scratches or lines that are not part of the text.
        # This should be fine for burmese text as it does not contain straight lines.

        return img_clone

    def contour(self, cv_image: cv.typing.MatLike, debug=False):

        (numLabels, labels, stats, centroids) = cv.connectedComponentsWithStats(
            cv.bitwise_not(cv_image)
        )

        print(f"Number of labels: {numLabels}")

        cv_image = cv.cvtColor(cv_image, cv.COLOR_GRAY2BGR)

        # Draw a bounding box around every label
        for i in range(1, numLabels):
            x, y, w, h, area = stats[i]
            cv.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv.putText(
                cv_image,
                str(area),
                (x, y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        cv.imshow("Contours", cv_image)

        return cv_image

    def resize(
        self, cv_image: cv.typing.MatLike, target_approx_char_size=32
    ) -> cv.typing.MatLike:
        ratio = target_approx_char_size / self.approx_char_size
        return cv.resize(
            cv_image, (0, 0), fx=ratio, fy=ratio, interpolation=cv.INTER_NEAREST
        )
