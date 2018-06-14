import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Functions



def loadcentreimgs():
	lines = []
	with open('./data/data/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		first_line = next(reader)
		for line in reader:
			lines.append(line)

	images = []
	measurements = []
	for line in lines:
		source_path = line[0]
		filename = source_path.split('/')[-1]
		current_path = './data/data/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		measurements.append(measurement)

	X_train = np.array(images)
	y_train = np.array(measurements)
	return X_train, y_train
	
def loadallimgs():
	csv_file = './data/data/driving_log.csv'
	with open(csv_file, 'r') as f:
		reader = csv.reader(f)
		first_line = next(reader)
		car_images = []
		steering_angles = []
		for row in reader:
			for i in range(3):
				row[i] = row[i].split('/')[-1]
			steering_center = float(row[3])
			# create adjusted steering measurements for the side camera images
			correction = 0.2 # this is a parameter to tune
			steering_left = steering_center + correction
			steering_right = steering_center - correction
			# read in images from center, left and right cameras
			path = './data/data/IMG/' # fill in the path to your training IMG directory
			img_center = cv2.imread(path + row[0])
			img_left = cv2.imread(path + row[1])
			img_right = cv2.imread(path + row[2])
			#img_center = process_image(np.asarray(Image.open(path + row[0])))
			#img_left = process_image(np.asarray(Image.open(path + row[1])))
			#img_right = process_image(np.asarray(Image.open(path + row[2])))
			# add images and angles to data set
			car_images.append(img_center)
			car_images.append(img_left) 
			car_images.append(img_right)
			steering_angles.append(steering_center)
			steering_angles.append(steering_left)
			steering_angles.append(steering_right)
	X_train = np.array(car_images)
	y_train = np.array(steering_angles)
	return X_train, y_train

def ConverttoYUV(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	return img

def resize(img, size):
	img = cv2.resize(img, (size[0], size[1]), interpolation  = cv2.INTER_AREA)
	return img

def crop(img, crop_size):
	top = crop_size[0][0]
	bottom = crop_size[0][1]
	left = crop_size[1][0]
	right = crop_size[1][1]
	x = img.shape[0]
	y = img.shape[1]
	#remove from the top of the image and from both sides
	img = img[top:x-bottom , left:y-right]
	return img

def plot_img(img, str):
	cv2.imshow("{}".format(str),img)
	cv2.waitKey(0)
	#cv2.destroyAllWindows()

def data_augment(imgs, measurement):
	imgs_aug = []
	measurement_aug =[]
	for i in range(len(imgs)):
		image_flipped = np.fliplr(imgs[i])
		measurement_flipped = -measurement[i]
		imgs_aug.append(image_flipped)
		measurement_aug.append(measurement_flipped)
	
	imgs_aug = np.array(imgs_aug)
	measurement_aug = np.array(measurement_aug)
	return imgs_aug, measurement_aug
	
def histplot(labels, filename):
	num_bins = len(np.unique(labels))
	n, bins, patches = plt.hist(labels, num_bins, facecolor='blue')
	plt.title("Steering angle distribution in the training dataset")
	plt.xlabel("Steering Angle")
	plt.ylabel("Frequency")
	plt.savefig('./figures/' + filename)

def translate(X_img, pixel_x, pixel_y):
    rows,cols,channels = X_img.shape
    M = np.float32([[1,0,pixel_x],[0,1,pixel_y]])
    X_img_trans = cv2.warpAffine(X_img,M,(cols,rows))
    if channels == 1:
        X_img_trans = X_img_trans[:, :, np.newaxis]
    assert (X_img.shape == X_img_trans.shape)
    
    return X_img_trans