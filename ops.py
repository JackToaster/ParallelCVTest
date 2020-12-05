import cv2


def blur(image, radius):
    new_img = cv2.GaussianBlur(image, (6 * radius + 1,) * 2, round(radius))

    return new_img


def hsv_threshold(image, hue=(100, 200), lum=(100, 200), sat=(100, 200)):
    ranges = tuple(zip(hue, lum, sat))
    img = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), *ranges)

    return img


def contour(image):
    contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Stands in for a simple visio pipeline - blur, threshold, contour.
def benchmark_op(image):
    blurred = blur(image, 5)
    thresholded = hsv_threshold(blurred)
    contours = contour(thresholded)
    return contours
