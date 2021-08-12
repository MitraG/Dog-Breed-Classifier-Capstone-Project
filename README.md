[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


# Dog Breed Classifier Capstone Project
A repository containing a Jupyter Notebook, saved models, requirements and images to deploy a fully-functioning Convolutional Neural Network (CNN) that detects dog and human faces as well as predict the dog breed or resembling breed respectively. 

### Table of Contents

1. [Project Motivation](#motivation)
2. [File Descriptions](#files)
3. [Installation](#installation)
4. [Strategy](#strategy)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Motivation<a name="motivation"></a>

The focus for my final project under the Data Science Nanodegree by Udacity is to create, train and test a deep learning pipeline, particularly Convolutional Neural Networks (CNNs) and transfer learning, to classify pictures of dogs to their respective breed. In this project, images of both dogs and humans are provided in the initial training dataset. 

If a dog is detected in the image, the algorithm should also predict the corresponding dog breed. However, the algorithm should provide a resembling dog breed if a human is detected in the image. If neither are present, the algorithm should provide an output that indicates an error. 


## File Descriptions <a name="files"></a>
Below is a summary of all files in this repository that is needed to achieve the goal of this project:
- images (containing all sample and test images to use on the CNN)
- dog_breed_classifier (the Jupyter Notebook that creates, trains and tests the CNN as well as designs an algorithm to predict the dog breed)
- haarcascades (files for the Haar feature-based cascade classifiers)
- extract_bottleneck_features.py (Python file to extract relevant bottleneck features)

## Installation <a name="installation"></a>

Below are all the libraries required to successfully run the code. These libraries are described in further detail in the Jupyter Notebook:

	- sklearn
	- keras
	- numpy
	- glob
	- Ipython.display
	- matplotlib.pylot
	- seaborn
	- random
	- pandas
	- matplotlib
	- seaborn
	- tqdm

## Strategy <a name="strategy"></a>

Udacity provided a skeleton strategy that supported my progress in completing this project. The model construction follows the steps below:

	1. Import and explore the dog and human datasets
	2. Create a human face detector using OpenCV
	3. Create a dog detector using Resnet50
	4. Create a CNN to classify dog breeds using Keras and evaluate model performance
	5. Use Transfer Learning to improve the initial CNN and evaluate model performance
	6. Write an algorithm to test new images
	

## Results<a name="results"></a>
The following is the accuracy each model received on the test set: 

	1. CNN from scratch: 7.4%
	2. CNN from VGG-16: 44.7%
	3. CNN from ResNet-50: 80.9%
	
To test out my algorithm, I thought it would be fun to use images of my friend’s pets and images of me and my friend as human examples! Although the model has only a test accuracy of 78.7%, the outcome of testing the algorithm is positive! Images of dogs are correctly classified by the model, most dogs with known breeds are correctly predicted by the model, and the model can distinguish between a human image and a dog image. When neither of those type of images are supplied, the model also produces the correct output which is saying ‘neither dog nor human’.

![alt text](https://github.com/MitraG/Dog-Breed-Classifier-Capstone-Project/blob/main/images/result_1.jpg)
![alt text](https://github.com/MitraG/Dog-Breed-Classifier-Capstone-Project/blob/main/images/result_2.jpg)
![alt text](https://github.com/MitraG/Dog-Breed-Classifier-Capstone-Project/blob/main/images/result_3.jpg)

Three ways to improve the algorithm are:

1. Use a better model fit for face detection; such as OpenCV’s DNN module or VGG Face.
2. Increasing the training data and possibly including mixed breed classifications to improve the accuracy of the model. This could lead to multi-classification and make the model more flexible as well.
3. Conducting a GridSearch on several Neural Networks will help find the optimal model for face detection as well.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
I'd like to acknowledge Udacity for publicly providing an amazing dataset to perform this classification project. 

I'd also like to thank my friends who provided images of their dogs and horse to be used on this classifier, I'm glad it worked on most of them!

Special acknowldgement goes to my colleagues and mentor who have supported my learning and wellbeing while I was undertaking the nanodegree.
