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

  
Bibliographer

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
