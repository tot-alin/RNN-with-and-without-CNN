# RNN-with-and-without-CNN

This project deals with the approach of recurrent neural networks (RNN) by
implementing six models (three RNN models and three RNN + CNN models) designed to
classify 12 human actions (at night). names_class : ['Drink', 'Picking', 'Push', 'Run', 'Throwing
objects', 'boxing', 'lifting weights', 'receiving the phone', 'stand', 'walking on stairs', 'walking
with flashlight', 'waving']

The dataset is 'https://www.kaggle.com/api/v1/datasets/download/lakavathakshay/noctact-har', used in this project, contains 6613 .mp4 videos of up to 15 seconds.

The project contains:
* Download dataset (download_data.ipynb)
* Data reprocessing(video_to_dataset.ipynb)
* Model created with SimpleRNN and description of SimpleRNN networks
* Model created with LSMT and description of LSMT networks
* Model created with GRU and description of GRU networks
* Model created with SimpleRNN + CNN
* Model created with LSMT + CNN
* Model created with GRU + CNN
* Comparison of the results obtained
* [RNN with and without CNN.pdf](https://github.com/user-attachments/files/21918238/RNN.with.and.without.CNN.pdf)

## Data preprocessing
Data preparation for processing is performed in the __video_to_dataset.ipynb__  file using the  __video_convert__  class. This class has the following features:
* Creating the database, i.e., retrieving information from the directory structure containing the films (indexing characteristics and indexing labe
* Changing image size in two steps:
  * Define the desired image ratio between width and height by adding values of 0, as the case may be, left and right, or top and bottom
  * changing the resolution of images to the desired form
* extracting the desired number of frames so that the captured frames are distributed evenly  over the length of the film
* generating the training dataset
* generation of the data set for validation
* generating a small test data set (normally, the test set should be much larger, but in this case, it is used to test functionality)
* method of saving data sets on physical media

## Recurrent neural networks (RNN)

Recurrent neural networks (RNNs) are neural networks used especially for time series or models whose predictions are based on data sets that have the characteristic of a chain of successive phenomena, such as sequences of film, sound, text, etc.

Recurrent neural networks (RNNs) have an architecture similar to artificial neural networks (ANNs), but unlike ANNs, RNNs use the same weights in each run sequence, while also passing on a feature emitted by the previous sequence, i.e., HL t1  combined with the characteristics from sequence HL t2

<img width="1107" height="449" alt="image" src="https://github.com/user-attachments/assets/964b6058-b843-421c-880f-92df7aabbd16" />

Taking the idea illustrated on the right side of the diagram above, the distribution of the data set ordered over a length of time (xt) in the cells of the RNN network is illustrated in the diagram below. It should be noted that in the case of multiple RNN layers, y’t (the prediction from the current layer) becomes the input feature of the upper layer, i.e., xt.

<img width="1216" height="573" alt="image" src="https://github.com/user-attachments/assets/e0caa73f-022e-4cd4-893f-be066ea7429b" />



  
  
## Bibliographer

* https://www.datacamp.com/tutorial/tutorial-for-recurrent-neural-network
* https://www.exxactcorp.com/blog/Deep-Learning/recurrent-neural-networks-rnn-deep-learning-for-sequential-data
* https://medium.com/analytics-vidhya/what-is-rnn-a157d903a88
* https://medium.com/analytics-vidhya/lstms-explained-a-complete-technically-accurate-conceptual-guide-with-keras-2a650327e8f2
* https://www.geeksforgeeks.org/deep-learning/deep-learning-introduction-to-long-short-term-memory/
* https://www.geeksforgeeks.org/machine-learning/gated-recurrent-unit-networks/
* https://d2l.ai/chapter_recurrent-modern/gru.html
* https://www.detailedpedia.com/wiki-Gated_recurrent_unit
* https://medium.com/smileinnovation/how-to-work-with-time-distributed-data-in-a-neural-network-b8b39aa4ce00
* https://levelup.gitconnected.com/hands-on-practice-with-time-distributed-layers-using-tensorflow-c776a5d78e7e
* https://colah.github.io/posts/2015-08-Understanding-LSTMs/
