import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

#carregar imagem
image = cv2.imread("example.jpeg")
#pequeno blur para melhorar reflexos
image_blur = cv2.medianBlur(image, 7)
#imagem para preto e branco
image_blur_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/bw.jpeg", image_blur_gray)
#aplicando threshold, tornando imagem totalmente branca ou preta
#calculando parte clara da imagem
image_res, image_thresh = cv2.threshold(image_blur_gray, 190, 255,
                                        cv2.THRESH_BINARY)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, kernel)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, last_image_bright = cv2.threshold(dist_transform, 0.1 * dist_transform.max(),
                                255, 0)
last_image_bright = np.uint8(last_image_bright)

cv2.imwrite("output/bwImg_bright.jpeg", last_image_bright)
#calculando parte escura da imagem
image_res2, image_thresh2 = cv2.threshold(image_blur_gray, 150, 255,
                                        cv2.THRESH_BINARY_INV)
kernel2 = np.ones((3, 3), np.uint8)
opening2 = cv2.morphologyEx(image_thresh2, cv2.MORPH_OPEN, kernel2)
dist_transform2 = cv2.distanceTransform(opening2, cv2.DIST_L2, 5)
ret, last_image_dark = cv2.threshold(dist_transform2, 0.1 * dist_transform2.max(),
                                255, 0)
last_image_dark = np.uint8(last_image_dark)

cv2.imwrite("output/bwImg_dark.jpeg", last_image_dark)

#juntando resultado das imagens claras e escuras
last_image = cv2.add(last_image_bright,last_image_dark)
cv2.imwrite("output/bwImg.jpeg", last_image)
#calculando contornos
edged = cv2.Canny(last_image, 30, 200)
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
#gerando img dos contornos
cv2.imwrite("output/edged.jpeg", edged)
#desenhando contornos e editando img inicial
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
cv2.imwrite("output/contours.jpeg", image)
#mostrando contagem de conntorrnos
print("total de objetos na imagem: " + str(len(contours)))

#plt.imshow(imutils.opencv2matplotlib(last_image_dark))
#plt.show()
