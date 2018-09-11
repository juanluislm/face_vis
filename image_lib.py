import numpy as np

def extract_patch(image, xmin, ymin, xmax, ymax):
    h = image.shape[0]
    w = image.shape[1]
    background = np.zeros((ymax-ymin, xmax-xmin, 3), image.dtype)
    x_from = max(xmin, 0)
    x_from_out = max(-xmin, 0)
    x_to = min(xmax, w)
    y_from = max(ymin, 0)
    y_from_out = max(-ymin, 0)
    y_to = min(ymax, h)
    background[y_from_out:(y_to-y_from+y_from_out), \
               x_from_out:(x_to-x_from+x_from_out)] = image[y_from:y_to, x_from:x_to]
    return background
