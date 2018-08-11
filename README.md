# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./figures/center_2016_12_01_13_30_48_287.jpg "CenterImage"
[image2]: ./figures/left_2016_12_01_13_30_48_287.jpg "LeftImage"
[image3]: ./figures/right_2016_12_01_13_30_48_287.jpg "RightImage"
[image4]: ./figures/center_2016_12_01_13_30_48_287_flipped.jpg "CenterImageFlipped"
[image5]: ./figures/center_2016_12_01_13_30_48_287_translated.jpg "CenterImageTranslated"
[image6]: ./figures/Trainingdatasetdistrib.jpg "TrainingDataSetOriginal"
[image7]: ./figures/Trainingdatasetdistrib_augment_batch.jpg "TrainingDataSetAugmented"
[image8]: ./figures/mse.jpg "MSE"

<!---## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.--->  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py](https://github.com/anammy/CarND-Behavioral-Cloning-P3/blob/master/model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consisted of a modified version of Nvidia's end-to-end convolution neural network (model.py lines 102-120). 

The data was normalized in the model using a Keras lambda layer. The top of the images were cropped since artifacts such as the sky and trees are not relevant inputs to predict the steering angle required and may confuse the model. RELU activation and dropout layers were added to introduce nonlinearity into the model and prevent overfitting, respectively.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with a learning rate of 0.001. I also varied the number of epochs to train the model on.

#### 4. Appropriate training data

I used the training data set provided with the Udacity project.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I used a convolution neural network model similar to Nvidia's End-to-End Deep Learning model architecture. This model seemed appropriate since it was used to train systems to drive based on camera images and steering angle inputs.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set using a 80/20 split ratio. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. I then added dropout layers to the model to reduce overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track such as sharp turns and the transition to the bridge. To improve the driving behavior in these cases, I augmented modified images to better even out steering angle inputs represented in the training data set. I also added RELU activation in the convolution layers in order to add nonlinearity in the model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 102-120) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 Grayscale image						|
| Normalization    		|											 	|
| Cropping		    	| 											 	|  
| Convolution 5x5    	| 2x2 stride, valid padding, output depth of 24	|
| RELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, output depth of 36	|
| RELU					|         										|
| Convolution 5x5	    | 2x2 stride, valid padding, output depth of 48	|
| RELU					|         										|
| Convolution 3x3	    | 1x1 stride, valid padding, output depth of 64	|
| RELU					|         										|
| Convolution 3x3	    | 1x1 stride, valid padding, output depth of 64	|
| RELU					|         										|
| Fully Connected		| Output = 1164									|
| Dropout				| Probability of keeping value = 0.5			| 
| Fully Connected		| Output = 100									|
| Dropout				| Probability of keeping value = 0.5			|
| Fully Connected		| Output = 50									|
| Dropout				| Probability of keeping value = 0.5			|
| Fully Connected		| Output = 10									|
| Dropout				| Probability of keeping value = 0.5			|
| Fully Connected		| Output = 1									|  

#### 3. Creation of the Training Set & Training Process

I used the sample training data set provided in the project repository. Here is an example image of center lane driving:

![alt text][image1]

I also used images from the left and right cameras in the training data set. I applied a correction factor of 0.2 to the corresponding center steering angles. The following are examples of images from the side cameras:

![alt text][image2]
![alt text][image3]

The following figure displays the training dataset distribution. The distribution is dominated by steering angles close to -0.2, 0, and 0.2 degrees.

![alt text][image6]

To augment the dataset, I flipped images and angles in order to even out the distribution of right and left turns in the training dataset. For example, here is an image that has then been flipped:

![alt text][image1]
*Original*

![alt text][image4]
*Flipped Image*

I also translated images corresponding to steering angles that weren't well represented in order to augment the training dataset.

![alt text][image1]
*Original*

![alt text][image5]
*Translated Image*

After data augmentation, the batches of images chosen within the generator function resembled more of a normal distribution as shown by the figure below.

![alt text][image7]

I then randomly shuffled the training data set prior to training the model. The validation set helped determine if the model was over or under fitting. I varied the number of epochs to minimize fluctuations in the mean squared error (mse). The following figure displays the training and validation mse for each epoch. The overall trend is a decrease in mse with some fluctuations.

![alt text][image8]

The video of the vehicle driving autonomously in the simulator using the trained model is given [here.](./video.mp4)

<!---<video width="320" height="240" controls>
  <source src="video.mp4" type="video/mp4">
</video>-->
