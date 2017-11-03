# Project 2 Traffic sign recognition

## Take a close look at the data
Assuming the all features have the correct labels we will take a close look at the distribution at first

### Data distribution
The first step will be to look at the total number of labels for the training samples(n train samples), the validation samples(n valid samples) and test samples(n test samples) for each Label(Label No.). In addition the Ratio of samples for training, validation and test are calculated to the total number of samples(n train + n valid + n test) for each label. The results are displayed in the following table:

|Label No. | n train samples | n valid samples | n test samples | ratio train | ratio valid | ratio test|
|----------|-----------------|-----------------|----------------|-------------|-------------|-----------|
|00|0180|0030|0060|0.67|0.11|0.22|
|01|1980|0240|0720|0.67|0.08|0.24|
|02|2010|0240|0750|0.67|0.08|0.25|
|03|1260|0150|0450|0.68|0.08|0.24|
|04|1770|0210|0660|0.67|0.08|0.25|
|05|1650|0210|0630|0.66|0.08|0.25|
|06|0360|0060|0150|0.63|0.11|0.26|
|07|1290|0150|0450|0.68|0.08|0.24|
|08|1260|0150|0450|0.68|0.08|0.24|
|09|1320|0150|0480|0.68|0.08|0.25|
|10|1800|0210|0660|0.67|0.08|0.25|
|11|1170|0150|0420|0.67|0.09|0.24|
|12|1890|0210|0690|0.68|0.08|0.25|
|13|1920|0240|0720|0.67|0.08|0.25|
|14|0690|0090|0270|0.66|0.09|0.26|
|15|0540|0090|0210|0.64|0.11|0.25|
|16|0360|0060|0150|0.63|0.11|0.26|
|17|0990|0120|0360|0.67|0.08|0.24|
|18|1080|0120|0390|0.68|0.08|0.25|
|19|0180|0030|0060|0.67|0.11|0.22|
|20|0300|0060|0090|0.67|0.13|0.20|
|21|0270|0060|0090|0.64|0.14|0.21|
|22|0330|0060|0120|0.65|0.12|0.24|
|23|0450|0060|0150|0.68|0.09|0.23|
|24|0240|0030|0090|0.67|0.08|0.25|
|25|1350|0150|0480|0.68|0.08|0.24|
|26|0540|0060|0180|0.69|0.08|0.23|
|27|0210|0030|0060|0.70|0.10|0.20|
|28|0480|0060|0150|0.70|0.09|0.22|
|29|0240|0030|0090|0.67|0.08|0.25|
|30|0390|0060|0150|0.65|0.10|0.25|
|31|0690|0090|0270|0.66|0.09|0.26|
|32|0210|0030|0060|0.70|0.10|0.20|
|33|0599|0090|0210|0.67|0.10|0.23|
|34|0360|0060|0120|0.67|0.11|0.22|
|35|1080|0120|0390|0.68|0.08|0.25|
|36|0330|0060|0120|0.65|0.12|0.24|
|37|0180|0030|0060|0.67|0.11|0.22|
|38|1860|0210|0690|0.67|0.08|0.25|
|39|0270|0030|0090|0.69|0.08|0.23|
|40|0300|0060|0090|0.67|0.13|0.20|
|41|0210|0030|0060|0.70|0.10|0.20|
|42|0210|0030|0090|0.64|0.09|0.27|

The first conclusion is that the number of samples varies for each label. For example label no. 0 has a total of 180 training samples whereas label no. 1 has a total of 1980 training samples being 11 times higher. This could result in an overfitting on certain labels and underfitting on others.
The second conclusion is that there is a small range of ratios for each label regarding the split of training(0.63-0.70), validation(0.08-0.14) and test(0.20-0.27) labels. Some kind of fixed ratio might be more successful for training, validating and testing the convnet.

### Conclusion on data analysis

It might make sense to train the network evenly on all labels by choosing a random set of each label roughly the same size for each batch.

## Implementing the Network

For learning purposes a new class was built called tfclassify. Own methods were implemented to add specific layers like Convolutional Layers and Fully connected layers as well as methods to add activation and Pooling possibilities.
Also routines for created for testing and prediction

## Network selection & Training

As unexperienced nn-developer I went for a trial&error approach. In the P2_full jupyter notebook(P2_full.html) shows multiple variations on the network architecture and training including different learning rates, decaying learning rates, epochs, training with even samples per label and additional layers to retrieve better performance on the nn. The saved training data can be found in the directory ./modeldata/models_from_P2_full_notebook. To enhance training performance g2.2xlarge instance of amazon web services was in use.
Finally a network with an additional convolutional layer with relu activation and pooling was chosen achieving validation performance beyond 93%. Using the test samples to test the network resulting in exactly 93% accuracy. The nn setup can be viewed in P2_V1 or P2_full[at the end] jupyter notebooks (P2_V1.html):
--------------------------------------------------
Convolutional Layer as Layer 1 with input dimensions (32, 32, 3) and output dimensions (28.0, 28.0, 7)
Activation on Layer 1 with type relu
Pooling on Layer 1 with input dimensions (28.0, 28.0, 7) and output dimensions (26.0, 26.0, 7)
Convolutional Layer as Layer 2 with input dimensions (26.0, 26.0, 7) and output dimensions (24.0, 24.0, 9)
Activation on Layer 2 with type relu
Pooling on Layer 2 with input dimensions (24.0, 24.0, 9) and output dimensions (12.0, 12.0, 9)
Convolutional Layer as Layer 3 with input dimensions (12.0, 12.0, 9) and output dimensions (10.0, 10.0, 16)
Activation on Layer 3 with type relu
Pooling on Layer 3 with input dimensions (10.0, 10.0, 16) and output dimensions (5.0, 5.0, 16)
Fully Connected Layer as Layer 4 with input dimensions (5.0, 5.0, 16) and output dimensions (120, 1, 1)
Activation on Layer 4 with type relu
Fully Connected Layer as Layer 5 with input dimensions (120, 1, 1) and output dimensions (84, 1, 1)
Activation on Layer 5 with type relu
Fully Connected Layer as Layer 6 with input dimensions (84, 1, 1) and output dimensions (43, 1, 1)
--------------------------------------------------
The training was setup with training rate of 0.0009, a batch size of 128 and 50 epochs

## Traffic sign prediction on self-selected images

5 signs were selected from the internet and reshaped t0 32x32 pixels. Then the images were fed to the network adding softmax and argmax to return probabilities and the exact prediction.
All signs except for one were identified correctly giving an accuracy of 80%.
The sign identified incorrectly was the speed limit sign of 70km/h which was predicted as 60km/h. 

## Conclusion

Surely a more effective nn architecture could be found to raise the accuracy. Also better training rates, number of epochs for training, more suitable image preparation as well as regularization methods like L2 or dropout (not used in this project submission) can be chosen. Regarding the accuracy of the predicting image content a convolutional network seems to be an ´easy´ and efficient way to classify visual data, if there is eneugh information to trian the network.