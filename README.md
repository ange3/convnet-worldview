# convnet-worldview

## Image Geo-Location
Our project focuses on geo-location for a subset of the 2015 MediaEval Placing dataset [1] and produces comparable performance to existing submissions on the classification task and superior performance on the regression task. We restrict our training set to images only, and attempt to extract sufficient information using a Convolutional Neural Network (CNN). 

## Country Classification Task and Coordinate Regression Task
We tackle two image location tasks – first a classification task to predict the country, and second a regression task to predict the exact coordinates. For the classification task, we use as our baseline simple feature extraction followed by an SVM, and for our final runs experiment with different sets of pre-trained models and CNN architectures. These methods yield results slightly above baseline. For the regression task, we use pre-trained models to predict latitude and longitude coordinates. This approach shows great improvements on the baseline.

## Dataset
[1] MediaEval Benchmarking Initiative for Multimedia Evaluation. 2015 Placing Task: Multimodal geo-location prediction. http://www.multimediaeval.org/mediaeval2015/placing2015
