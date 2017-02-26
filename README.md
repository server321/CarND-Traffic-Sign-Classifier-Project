#**Traffic Sign Recognition** 

##Udacity project 2

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

1. Load the data set ([link](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip "Link"))
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


- Number of training examples = 34799
- Number of testing examples = 12630
- Number of validation  examples = 4410
- Image data shape = (32, 32, 3)
- Number of classes train dataset = 43
- Number of classes test dataset = 43
- Number of classes validation dataset = 43


####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

![alt text][image2]
###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as normalization, etc.

The code for this step is contained in the the IPython notebook.

I decided not to convert the images to grayscale because we loss some important data and models work worse.

I normalized the image data. Input data normalization is very important to construct a neural network model. It convert all features (pixels) to the same scale. Normalized data helps Gradient Descent or similar algorithms to work quickly.

Normalization implemented at min_max_normalization function. I used mix max normalization (min=-0.5, max=0.5). In my case MinMax Scaling works better than other algoritms.

One-Hot Encoding was used to convert label numbers to vectors. One-Hot encoding implemented in one_hot() function. It uses tensorflow fuction tf.one_hot.




####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly shuffle and split the training data into a training set and validation set (80% - 20%). I did this by train_test_split



####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the ipython notebook. 

**Hyperparameters:** 
I chose:

**learning rate** of 0.0005

**batch size** of 128 

and ran optimization for a total of 28 epochs.


**Epochs:** I ran a total of 28 epochs for training the neural network.


**Stopping conditions:** I used accuracy of validation data as a criteria to monitor if model was overfitting. After 28 epoches accuracy of validation changes minimally or drops. 


**Optimization:** I used adamoptimizer with default settings for optimization.


###LeNet Model:
______________________________________


**Layer 1**
Convolutional. Input = 32x32x1. Output = 28x28x6.
This layer transforms the Tensor 32x32x1 to 28x28x6.
Use a filter with the shape (5, 5, 1, 6) with Valid padding.
Use a standard ReLU activation.
Use max pooling. Input = 10x10x16. Output = 5x5x16.

**Layer 2**
Convolutional. Output = 10x10x16.
This layer transforms the Tensor 5x5x16 to 10x10x16.
Valid padding
Use a standard ReLU activation.
Use max pooling. Input = 10x10x16. Output = 5x5x16.
Use flatten. Input = 5x5x16. Output = 400

**Layer 3**
Fully Connected. Input = 400. Output = 120.
Use a standard ReLU activation.    

**Layer 4**
Fully Connected. Input = 120. Output = 84
Use a standard ReLU activation.   

**Layer 5**
Softmax. Fully Connected. Input = 84. Output = 43.




####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
 

LeNet Model:
Hyperparameters: I chose a learning rate of 0.0005, batch size of 128 and ran optimization for a total of 28 epochs.
Epochs : I ran a total of 28 epochs for training the neural network.
Stopping conditions : I used accuracy of validation data as a criteria to monitor if model was overfitting. After 28 epoches accuracy of validation changes minimally or drops. 
Optimization : I used adamoptimizer with default settings for optimization.


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:


Validation set accuracy of 98,0%


If a well known architecture was chosen:
* What architecture was chosen? LeNet
* Why did you believe it would be relevant to the traffic sign application? Accuracy is rather high
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? 


I used an architecture model LeNet. I selected this model because it is 
Well-known model, widely udes for computer vision tasks and rather fast to train on slow computers (because I have unrecoverable tensorflow error when use GPU in some kind of models)
At the initial stage I didn't make augmented data, but constructed functions for data augmentation and could implement it later. Even without these fucntions I have 96.6% accuracy.
 

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
| Road work				| Wild animals crossing											|
| Roundabout mandatory	| Keep right			 				|
| Stop sign				| Stop sign		      							|
| Stop sign				| Stop sign	   		   							|
| General caution		| General caution      							|

The model was able to correctly guess 4 of the 7 traffic signs, which gives an accuracy of 57%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.


![alt text][image11]



I used ideas of Vivek Yadav on image augmentation to generate additional data to train.
https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc#.gsvngk4mu 
The module for data augmentation is under development.

Future steps are implementations of keras model.
 