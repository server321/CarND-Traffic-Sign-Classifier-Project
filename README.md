#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

1. Load the data set (see below for links to the project data set)
2. Explore, summarize and visualize the data set
3. Design, train and test a model architecture
4. Use the model to make predictions on new images
5. Analyze the softmax probabilities of the new images
6. Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/training_samples_distribution.png "Training samples distribution"
[image2]: ./images/test_samples_distribution.png "Test samples distribution"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[image01]: ./big-new-signs/50.jpg "Traffic Sign 1"
[image02]: ./big-new-signs/road_work1.jpg "Traffic Sign 2"
[image03]: ./big-new-signs/road_work2.jpg "Traffic Sign 3"
[image04]: ./big-new-signs/roundabout.jpg "Traffic Sign 4"
[image05]: ./big-new-signs/stop.jpg "Traffic Sign 5"
[image06]: ./big-new-signs/stop2.jpg "Traffic Sign 6"
[image07]: ./big-new-signs/warning.jpg "Traffic Sign 7"

[image11]: ./images/1.png "Softmax"





## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/server321/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43



####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

![alt text][image2]
###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

I decided not to convert the images to grayscale because we loss some important data and models work worse.

<!---
Here is an example of a traffic sign image before and after grayscaling.-->

I normalized the image data.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly shuffle and split the training data into a training set and validation set (80% - 20%). I did this by train_test_split

My final training set had 27839 number of images. My validation set and test set had 6960 and 12630 number of images.

<!---
The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 
-->

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

<!---
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
--> 

**LeNet**

This architecture is based on the LeNet-5 neural network architecture.

___________________________________________________________________________________________________
Layer (type)                    | Output Shape          Param      Connected to                     
___________________________________________________________________________________________
convolution2d_1 (Convolution2D) | (None, 30, 30, 32)    896         convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)   | (None, 15, 15, 32)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)             | (None, 15, 15, 32)    0           maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
activation_1 (Activation)       | (None, 15, 15, 32)    0           dropout_1[0][0]                  
____________________________________________________________________________________________________
flatten_1 (Flatten)             | (None, 7200)          0           activation_1[0][0]               
____________________________________________________________________________________________________
dense_1 (Dense)                 | (None, 128)           921728      flatten_1[0][0]                  
____________________________________________________________________________________________________
activation_2 (Activation)       | (None, 128)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                 | (None, 43)            5547        activation_2[0][0]               
____________________________________________________________________________________________________
activation_3 (Activation)       | (None, 43)            0           dense_2[0][0]                    
___________________________________________________________________________________________




**CNN**

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
____________________________________________________________________________________________________
conv1 (Convolution2D)            (None, 32, 32, 32)    2432        convolution2d_input_3[0][0]      
____________________________________________________________________________________________________
relu1 (Activation)               (None, 32, 32, 32)    0           conv1[0][0]                      
____________________________________________________________________________________________________
maxpool1 (MaxPooling2D)          (None, 31, 31, 32)    0           relu1[0][0]                      
____________________________________________________________________________________________________
conv2 (Convolution2D)            (None, 31, 31, 64)    51264       maxpool1[0][0]                   
____________________________________________________________________________________________________
relu2 (Activation)               (None, 31, 31, 64)    0           conv2[0][0]                      
____________________________________________________________________________________________________
maxpool2 (MaxPooling2D)          (None, 15, 15, 64)    0           relu2[0][0]                      
____________________________________________________________________________________________________
conv3 (Convolution2D)            (None, 15, 15, 64)    102464      maxpool2[0][0]                   
____________________________________________________________________________________________________
relu3 (Activation)               (None, 15, 15, 64)    0           conv3[0][0]                      
____________________________________________________________________________________________________
maxpool3 (MaxPooling2D)          (None, 7, 7, 64)      0           relu3[0][0]                      
____________________________________________________________________________________________________
flatten (Flatten)                (None, 3136)          0           maxpool3[0][0]                   
____________________________________________________________________________________________________
dropout1 (Dropout)               (None, 3136)          0           flatten[0][0]                    
____________________________________________________________________________________________________
hidden1 (Dense)                  (None, 128)           401536      dropout1[0][0]                   
____________________________________________________________________________________________________
relu4 (Activation)               (None, 128)           0           hidden1[0][0]                    
____________________________________________________________________________________________________
dropout2 (Dropout)               (None, 128)           0           relu4[0][0]                      
____________________________________________________________________________________________________
hidden2 (Dense)                  (None, 128)           16512       dropout2[0][0]                   
____________________________________________________________________________________________________
relu5 (Activation)               (None, 128)           0           hidden2[0][0]                    
____________________________________________________________________________________________________
output (Dense)                   (None, 43)            5547        relu5[0][0]                      
____________________________________________________________________________________________________
softmax (Activation)             (None, 43)            0           output[0][0]                     
____________________________________________________________________________________________________


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

The adam optimizer was used with a learning rate of 0.001. Batchsize was 128 with 10 epochs.


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:

* training set accuracy of 96,03%
* validation set accuracy of 95,33%

<!---
 * test set accuracy of ?
--> 

Model loaded from file: my_model_cnn.h5 
12630/12630 [==============================] - 8s     
Test accuracy = 0.960332541567696
4410/4410 [==============================] - 2s     
Test accuracy = 0.9532879818594104

<!---
If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
--> 
If a well known architecture was chosen:
* What architecture was chosen? LeNet
* Why did you believe it would be relevant to the traffic sign application? Accuracy is rather high
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? The accuracy is more than 90%
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 7 German traffic signs that I found on the web:

![alt text][image01]

![alt text][image02]

![alt text][image03]
 
![alt text][image04]

![alt text][image05]

![alt text][image06]

![alt text][image07]

The third image might be difficult to classify because of unusual angle.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50 km/h       		| No passing   									| 
| Road work    			| Road work 									|
| Road work				| Yield											|
| Roundabout mandatory	| Roundabout mandatory			 				|
| Stop sign				| Stop sign		      							|
| Stop sign				| Stop sign	   		   							|
| General caution		| General caution      							|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

<!---
For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...
--> 
![alt text][image11]



I used ideas of Vivek Yadav on image augmentation to generate additional data to train.
https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc#.gsvngk4mu 
The module for data augmentation is under development.

Future steps are implementations of keras model.
 