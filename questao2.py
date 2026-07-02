import numpy as np 
import cv2
import matplotlib.pyplot as plt

img2 = cv2.imread("onion.jpg", cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Redimensiona para vetor de pixels
pixel_values = img_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Define critérios de parada
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)


for K in range(1,15):
    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Converte centros para valores inteiros
    centers = np.uint8(centers)

    # Mapeia cada pixel para o centro do cluster
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(img_rgb.shape)

    cv2.imwrite("vegetais_"+str(K)+".jpg", cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))