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
<p align="center">
<img width="680" height="400" alt="image" src="https://github.com/user-attachments/assets/e7499695-16f1-4337-9d3f-93acd499674a" />
</p>
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

*A simple approach :*
<br /> *# Define sample input shape (e.g., 16 sequences, 5 time steps, 3 features)*
<br /> *batch_size = 16*
<br /> *time_steps = 5*
<br /> *features = 3*
<br /> *input_data = tf.random.normal((batch_size, time_steps, features), dtype = tf.float32)*
<br /> *print(input_data.shape)*
<br /> *# number of units per layer*
<br /> *units_size_per_layer  = 4*
<br /> *LSTM_whole_sequence_output, LSTM_final_memory_state, LSTM_final_carry_state = LSTM(units_size_per_layer, return_sequences=True, return_state=True)(input_data)*
<br /> *print(LSTM_whole_sequence_output.shape)*
<br /> *print(LSTM_final_memory_state.shape)*
<br /> *print(LSTM_final_carry_state.shape)*
<br /> *(16, 5, 3)*
<br /> *(16, 5, 4)*
<br /> *(16, 4)*
<br /> *(16, 4)*

## Gated Recurrent Unit ( GRU )

Gate recurrent units (GRUs) are a type of RNN whose principle is to use gate mechanisms to selectively update the hidden state at each time step, allowing them to retain important information and eliminate irrelevant details. GRU is a simplified version of LSTM architectures and consists of two main gates: the update gate and the reset gate.
* Update gate zt: this gate decides how much information from the previous hidden state h (t-1) should be retained for the next
* Reset Gate rt: This gate determines how much of the hidden state from the past h (t-1) should be forgotten.
<p align="center">
<img width="570" height="450" alt="image" src="https://github.com/user-attachments/assets/71077324-0fe7-4db2-8abc-04e0d3146083" />
</p>
The implementation equations of the GRU block
<br /><img width="226" height="18" alt="image" src="https://github.com/user-attachments/assets/0651d6f7-fbf9-477d-9561-c791e47f0f80" />  Update Gate
<br /><img width="224" height="18" alt="image" src="https://github.com/user-attachments/assets/683bbcfd-0469-412e-91f3-ae697fa3b475" />  Reset Gate
<br /><img width="270" height="19" alt="image" src="https://github.com/user-attachments/assets/d1910a96-9b5e-417c-b956-f4467d705ac3" />
<br /><img width="277" height="27" alt="image" src="https://github.com/user-attachments/assets/902d0629-11e8-457c-81f3-dad431cf63d5" />  Ouput

*A simple approach :*
<br /> *# Define sample input shape (e.g., 16 sequences, 5 time steps, 3 features)*
<br /> *batch_size = 16*
<br /> *time_steps = 5*
<br /> *features = 3*
<br /> *input_data = tf.random.normal((batch_size, time_steps, features), dtype = tf.float32)*
<br /> *print(input_data.shape)*
<br /> *# number of units per layer*
<br /> *units_size_per_layer  = 4*
<br /> *GRU_whole_sequence_output, GRU_final_memory_state = GRU(units_size_per_layer, return_sequences=True, return_state=True, unroll=True)(input_data)*
<br /> *print(GRU_whole_sequence_output.shape)*
<br /> *print(GRU_final_memory_state.shape)*
<br /> *(16, 5, 3)*
<br /> *(16, 5, 4)*
<br /> *(16, 4)*

## Time distributed layer

TimeDistributed is a method whereby a certain layer, method function, etc., can be executed successively with different input characteristics, returning a number of results equal to the number of inputs. This method is used, for example, in processing data series, video frames, audio sequences, etc., where each time step is treated independently with the same method. For example, the figure below shows the successive approach of a CNN layer using TimeDistributed.
<p align="center">
<img width="450" height="200" alt="image" src="https://github.com/user-attachments/assets/bc190453-7569-40ce-b4ca-d528d38d6440" />
</p>
 
## Description of models
The six models were designed to highlight the functionality of RNNs in the context of their use in models involving the analysis of actions in a video recording. To this end, three models were created that use only the three types of RNN, namely SimpleRNN, LSTM, and GRU. As can be seen in the diagram below, these models are composed of three RNN layers and a final Dense layer.
<p align="center">
<img width="460" height="287" alt="image" src="https://github.com/user-attachments/assets/3f1b1c19-6116-42a9-9c52-684802ac93db" />
 </p>
<br /> SimpleRNN_model - Trainable params: 29,523,564 (112.62 MB) 
<br /> LSTM_model - Trainable params: 118,093,068 (450.49 MB) 
<br /> GRU_model - Trainable params: 88,570,572 (337.87 MB) 

<br />Due to the large input characteristics, i.e., 10 time steps containing 240x320x3 frames, the parameter matrices are very large, as can be seen above. One solution for improving the efficiency of models containing RNN networks for predicting datasets composed of video recordings is to use CNN networks in the composition of the models. Thus, three other models were created by extending the models mentioned above using the TimeDistributed method, in which a CNN layer was integrated. The diagram below shows the approach of the three new models.
<p align="center">
<img width="645" height="300" alt="image" src="https://github.com/user-attachments/assets/23631400-bb56-480a-b21a-ff1b421c1096" />
</p>
<br />CNN_SimpleRNN_model - Trainable params: 3,089,004 (11.78 MB)
<br />CNN_LSTM_model - Trainable params: 11,835,660 (45.15 MB)
<br />CNN_GRU_model - Trainable params: 8,920,780 (34.03 MB)

<br />**It should be noted that the descriptions of the Input, MaxPooling2...., and other methods have been omitted !**

## Comparative diagram of the loss function for the 6 models created
<br /><img width="1630" height="855" alt="image" src="https://github.com/user-attachments/assets/ad718f4d-7f53-4c32-bfaa-ba54a4d31888" />

## Comparative diagram of efficiency (prediction accuracy) for the six models created
<br /><img width="1621" height="855" alt="image" src="https://github.com/user-attachments/assets/5c1a93a8-1fec-4658-a003-74f70df3eeb5" />

## Confusion matrices for models built with RNN
<br /><img width="2000" height="700" alt="image" src="https://github.com/user-attachments/assets/f8f08148-c14d-4c3a-9733-b89eae57ff6a" />
<br /><img width="2000" height="700" alt="image" src="https://github.com/user-attachments/assets/5006129c-338e-466c-a8d9-201258e977fa" />
<br /><img width="2000" height="700" alt="image" src="https://github.com/user-attachments/assets/06b2c64a-1bdd-4b9a-94a2-1c68742a96fe" />

## Confusion matrices for models built with RNN and CNN
<br /><img width="2000" height="700" alt="image" src="https://github.com/user-attachments/assets/16accff8-4fdd-4a71-b55c-db11fd174ad5" />
<br /><img width="2000" height="700" alt="image" src="https://github.com/user-attachments/assets/6cecae01-5f0e-46b8-b4dd-74b39b571ddd" />
<br /><img width="2000" height="700" alt="image" src="https://github.com/user-attachments/assets/25969561-22e0-429f-9e6d-834163bb71a0" />








  
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
