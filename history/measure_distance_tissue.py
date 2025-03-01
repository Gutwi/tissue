import imgsim
import cv2
import os

vtr = imgsim.Vectorizer()

base = "./my_dir/matigai/ref.jpg"
tar_good_e = "./my_dir/matigai/tar_good_easy.jpg"
tar_good_h = "./my_dir/matigai/tar_good_hard.jpg"
tar_bad_e = "./my_dir/matigai/tar_bad_easy.jpg"
tar_bad_h = "./my_dir/matigai/tar_bad_hard.jpg"


base_img = cv2.imread(base)
tar_goodE_img = cv2.imread(tar_good_e)
tar_goodH_img = cv2.imread(tar_good_h)
tar_badE_img = cv2.imread(tar_bad_e)
tar_badH_img = cv2.imread(tar_bad_h)


base_vec = vtr.vectorize(base_img)
good_e_vec = vtr.vectorize(tar_goodE_img)
good_h_vec = vtr.vectorize(tar_goodH_img)
bad_e_vec = vtr.vectorize(tar_badE_img)
bad_h_vec = vtr.vectorize(tar_badH_img)

dist0 = imgsim.distance(base_vec, base_vec)
print(os.path.basename(base)+" Distance =", round(dist0, 2))
dist1 = imgsim.distance(base_vec, good_e_vec)
print(os.path.basename(tar_good_e)+" Distance =", round(dist1, 2))
dist2 = imgsim.distance(base_vec, good_h_vec)
print(os.path.basename(tar_good_h)+" Distance =", round(dist2, 2))
dist3 = imgsim.distance(base_vec, bad_e_vec)
print(os.path.basename(tar_bad_e)+" Distance =", round(dist3, 2))
dist4 = imgsim.distance(base_vec, bad_h_vec)
print(os.path.basename(tar_bad_h)+" Distance =", round(dist4, 2))