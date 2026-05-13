import cv2
import numpy as np

def segment_object_grabcut(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    height, width = img.shape[:2]
    rect = (10, 10, width - 20, height - 20)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask_final = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")

    return mask_final