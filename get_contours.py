import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import numpy as np
import cv2
from PIL import Image

def process_an_image(img):
    output_image_path = "output_pic/out_tmp.jpg"
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
    plt.imshow(binary_image)
    plt.show()  
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    blank_image = np.zeros_like(binary_image)
    cv2.drawContours(blank_image, contours, -1, (255), thickness=2)
    plt.imshow(blank_image)
    plt.show()  
    return contours


img = mplimg.imread("pic/t8.jpg")

print("start to process the image....")
track1_contours=process_an_image(img)
np.save("track8_contours.npy", track1_contours)
print("Contours saved to contours.npy")