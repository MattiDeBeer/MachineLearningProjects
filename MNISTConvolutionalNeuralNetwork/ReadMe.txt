Author: Matti de Beer
Last edited: 06/09/2023

For any of the code here to run correctly you are required to download the .npy files containing the formatted MNIST dataset that I have formatted for use in this project. They can be found here:
https://www.dropbox.com/sh/wyo0u8dltqjtppt/AADM_NAgC4SPz5WukL5qYLlba?dl=0
Save them in the same directory as the .py code files.

Please also refer to the documentation pdf file if you are interested as it has lots of background information about how both convolutional and normal neural network works, as well as a detailed outline of this projects development.

This Project consists of a Convolutional neural network with various features. Below is a description of the code files in this project.

CNNTrainer.py will train the Convolutional Neural Network to the specified parameters. It created a random weight and bias array and then performs a specified amount of backpropagation steps. The parameters can be adjusted in the body of the code.

CNN test.py tests the trained neural network using the MNIST dataset. It performs the image convolutions with the specified filters and then a forward propagation on result of the convolution and pooling layers. It then uses the label file to check if the network was correct or not. Using this it calculates the accuracy of the network.

customNumberTrainer.py prompts the user to hand draw a digit. It then runs this digit through the CNN and predicts what it is. If incorrect it will perform a backpropagation step in order to home on its accuracy.

imageReader.py  opens the custom image saved from the number painter program and converts it into a 28 x 28 matrix for input to the neural networks convolution layer.
numberPainter.py allows the user to paint their own image and save it locally. It is used by the predictor trained program to get a user hand drawn image.

WeightsCNN.npy and biasCNN.npy are pre trained weight and bias files that are used by the test and CNNTest and customNumberTrainer programs. These programs can be used on weight and bias files you have trained yourself using the CNNtrainer, just change the filenames in the code.
