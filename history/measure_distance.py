import imgsim
import cv2


vtr = imgsim.Vectorizer()

penguin_img = cv2.imread("./input/base.jpg")
penguin_lr_img = cv2.imread("./input/flip_lr.jpg")
another_penguin_img = cv2.imread("./input/small.jpg")
hummingbird_img = cv2.imread("./input/bird.png")


penguin_vec = vtr.vectorize(penguin_img)
penguin_lr_vec = vtr.vectorize(penguin_lr_img)
another_penguin_vec = vtr.vectorize(another_penguin_img)
hummingbird_vec = vtr.vectorize(hummingbird_img)

dist0 = imgsim.distance(penguin_vec, penguin_vec)
print("Same Distance =", round(dist0, 2))
dist1 = imgsim.distance(penguin_vec, penguin_lr_vec)
print("Reversal Distance =", round(dist1, 2))
dist2 = imgsim.distance(penguin_vec, another_penguin_vec)
print("Another Distance =", round(dist2, 2))
dist3 = imgsim.distance(penguin_vec, hummingbird_vec)
print("Other Distance =", round(dist3, 2))
