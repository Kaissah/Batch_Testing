import sys
import cv2
import numpy as np

sys.path.append(r"F:\USER\Documents\FEU\Programming\Thesis QA\Puspin v1\Puspin_v1.py")
from Puspin_v1 import detect

if __name__ == "__main__":
    img = cv2.imread(r"..\Dataset\Allergy dermatitis\AD_Opulencia_2.png")
    if img is not None:
        res = detect(img)
        cv2.imshow("Adaptive FAD Detection", res.overlay_image)
        cv2.waitKey(0)