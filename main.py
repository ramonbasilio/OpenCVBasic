import cv2
loadAlg = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

image = cv2.imread('imagens/imagem_03.jpeg')
image = cv2.resize(image, (1280, 900))

imageGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = loadAlg.detectMultiScale(imageGrey)

print(faces)

for(x, y, l, a) in faces:
    cv2.rectangle(image, (x,y), (x+l, y+a), (0, 255, 0), 2)

cv2.imshow('Faces', image)

cv2.waitKey()
