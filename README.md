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

## Simple RNN

SimpleRNN is the simplest form of the recurrent neural networks. Its functionality is achieved by combining the input features of sequence t1 with the features emitted by the previous sequence t <img width="54" height="18" alt="image" src="https://github.com/user-attachments/assets/aab7cee7-cef8-4237-b077-eff5b4fa2ed2" /> and activated with a hyperbolic tangent. The equation of the activation function is <img width="144" height="18" alt="image" src="https://github.com/user-attachments/assets/83479a47-5f8f-4cf3-9b4f-87898fd663ac" />

Equations for implementing the SimpleRNN block

<br /> <img width="243" height="18" alt="image" src="https://github.com/user-attachments/assets/7b936ff8-369c-4f2d-ae26-6e086efa3815" />
<img width="897" height="754" alt="image" src="https://github.com/user-attachments/assets/f37fc8aa-5172-4aaf-9015-6892a2ab3e5a" />

*A simple approach :*
<br /> *# Define sample input shape (e.g., 16 sequences, 5 time steps, 3 features)*
<br /> *batch_size = 16*
<br /> *time_steps = 5*
<br /> *features = 3*
<br />*input_data = tf.random.normal((batch_size, time_steps, features), dtype = tf.float32)*
<br /> *print(input_data.shape)*
<br />*# number of units per layer*
<br />*units_size_per_layer  = 4*
<br />*SimpleRNN_whole_sequence_output, SimpleRNN_final_memory_state = SimpleRNN(units_size_per_layer, return_sequences=True, return_state=True)(input_data)*
<br />*print(SimpleRNN_whole_sequence_output.shape)*
<br />*print(SimpleRNN_final_memory_state.shape)*
<br />*(16, 5, 3)*
<br />*(16, 5, 4)*
<br />*(16, 4)*


## Long Short-Term Memory ( LSTM )

Long Short-Term Memory (LSTM) is an improved version of simple Recurrent Neural Networks. The main difference between SimpleRNN and LSTM is that, in addition to the hidden state taken from the previous sequence and concatenated with the input features corresponding to the time sequence, LSTM networks have a memory cell with extended information as a period (Cell State).
<img width="1366" height="796" alt="image" src="https://github.com/user-attachments/assets/e7499695-16f1-4337-9d3f-93acd499674a" />

The architecture of LSTM networks consists of three gates:
* Forget gate: determines what information is deleted from the cell memory
* Input gate: controls what information is added to the cell memory
* Output gate: controls what information comes out of the cell memory

Equations for implementing the LSTM block
<br /><img width="258" height="24" alt="image" src="https://github.com/user-attachments/assets/fc0d0b95-b970-42c1-ba10-91a4f186d883" />  - Forget gate
<br /><img width="246" height="18" alt="image" src="https://github.com/user-attachments/assets/560b33d7-0674-48ce-a46d-9cc5d10d72ae" />  - Input gate
<br /><img width="232" height="19" alt="image" src="https://github.com/user-attachments/assets/c7078c1f-9746-435e-a874-612b81c2efd9" />
<br /><img width="253" height="18" alt="image" src="https://github.com/user-attachments/assets/215326a8-7ab2-4e14-99e5-7635b3a870bd" />
<br /><img width="124" height="23" alt="image" src="https://github.com/user-attachments/assets/e91ddbcd-e3a4-4b45-9847-394763a5fb59" />  - Cell State
<br /><img width="147" height="23" alt="image" src="https://github.com/user-attachments/assets/afa6cf35-46e8-49ed-b569-755d789e6dc1" />  - Output gate

  
  
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
