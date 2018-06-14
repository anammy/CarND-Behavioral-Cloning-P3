import numpy as np
import cv2
import functions as fcns

path = 'center_2016_12_01_13_30_48_287.jpg'

img = cv2.imread(path)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_flipped = np.fliplr(img)
cv2.imshow('image',img_flipped)
cv2.waitKey(0)
cv2.destroyAllWindows()

path_save = 'center_2016_12_01_13_30_48_287_flipped.jpg'
cv2.imwrite(path_save, img_flipped)

img_trans = fcns.translate(img, 3, 3)
path_save2 = 'center_2016_12_01_13_30_48_287_translated.jpg'
cv2.imwrite(path_save2, img_trans)