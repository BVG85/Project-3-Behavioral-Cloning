# **Behavioral Cloning** 

<img src="https://github.com/BVG85/Project-3-Behavioral-Cloning/blob/master/header.jpg">
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./
[image2]: ./centre_example.jpg "Centre Example"
[image3]: ./centre_example_t2.jpg "Centre Example - Track 2"
[image4]: ./output_DFkI7t.gif
[image5]: ./left_cam.jpg "Left Camera Image" 
[image6]: ./centre_cam.jpg "Centre Camera Image"
[image7]: ./right_cam.jpg "Right Camera Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://github.com/BVG85/Project-3-Behavioral-Cloning/blob/master/model.py) containing the script to create and train the model 
* [drive.py](https://github.com/BVG85/Project-3-Behavioral-Cloning/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/BVG85/Project-3-Behavioral-Cloning/blob/master/model.h5) containing a trained convolution neural network 
* writeup.md report (this file)
* a [video](https://github.com/BVG85/Project-3-Behavioral-Cloning/blob/master/video_track1.mp4) recording of the car driving around the track in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Initially the LeNet architecture was used for the model, with a reasonable amount of success. The vehicle struggled only with a few corners of the track. However once the Nvidia architecture was implemented, the results improved significantly. 

The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

During experimenting with different solutions dropout and maxpooling (with the LeNet architecture) was used. However, for the final model this was not used. The data was split for a 20% validation set and shuffled.The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use the LeNet architecture to evaluate performance based on the data acquired. The LeNet architecture was chosen as an initial model to fine tune to achieve the desired results. A Lambda layer was added for the normalization of the data. Good results were obtained using this model. However some of the corners on the track proved to be problematic. After cropping the images to remove the influence of objects not on the track itself, results deteriorated.

The NVIDEA architecture was then implemented and results improved. During the autonomous mode it was observed that the model struggled recovering to center. Additional data was then acquired to capture corrections to the centre of the track. 
Once the model was trained with this additional data set, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model summary can be seen below
```sh
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 31, 158, 24)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       dropout_1[0][0]                  
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0

```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

Then two laps were recorded of the vehicle driving the track in reverse, using center lane driving.  

Then I repeated this process on track two in order to get more data points. 

![alt text][image3]

After training the model with some degree of success, the left and right camera images were incorporated with a correction factor of 0.3

##### Left Camera Image
![alt text][image5]
##### Centre Camera Image
![alt text][image6]
##### Right Camera Image
![alt text][image7]

The images were not flipped in the final design. However two more laps were recorded, specifically on recovering from the side of the road back to the centre. One lap was recorded recovering from the left and another lap to record recovering from the right hand side of the road.

All data was recorded in training mode, using the keyboard, to control acceleration and braking, and the mouse for steering. I found the mouse had finer and precise control allowing for smoother turns and corrections.

##### Example of recovering to centre data
![alt text][image4]


Finally the data was shuffled and 20% of the data was placed into a validation set. 

I used this training data for training the model. The model was trained for 10 epochs on a GPU. I used an adam optimizer so that manually training the learning rate wasn't necessary.
