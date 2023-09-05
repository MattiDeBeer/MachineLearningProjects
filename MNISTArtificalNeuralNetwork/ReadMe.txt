Author: Matti de Beer 
Updated: 05/09/2023

All the code here requires a formatted version of the MNIST dataset to run. To download the formatted MNIST datasets, the 4 files name X.npy, Xtest.npy, Y.npy and Ytest.npy can be found in my DropBox here: https://www.dropbox.com/sh/wkk4oz2sl51tiqf/AABwwS-oXN-mOv8zQReUxcmCa?dl=0
Just put them in the same directory as the code.

This project consists of an artificial neural network that can train and test itself on the MNIST dataset of handwritten digits. It can also prompt the user to write a digit and attempts to predict what it is, and if incorrect can further train using that information. Make sure to use IDLE or another Python IDE (such as Spyder) to run the programs as Tkinter and MatPlotLib only work as intended when this is done. 

TrainerV2.py is a program that initializes and trains a neural network from scratch each time you run it. It saves the output weights and biases to a file that can be used by the other programs. The user can change the sample sizes, the amount of backpropagation iterations performed and the learning rate.

TesterV2.py is a program that can test the trained networks on an MNIST training dataset. The user can specify the weights and biases they want to test.

PredictorTrainer.py is a program that allows the user to input their own handwritten digit. The program will then use the trained neural network to attempt to guess what the digit was. If incorrect the user can input the correct answer and then the program will perform a backpropagation step to hopefully make the network more accurate.

CustomNumberPredictor.py Is a file that allows the user to hand draw an image and have the network guess what it is.

CustomReader.py is a program that reads a custom saved image and turns it into a NumPy column vector that can be inputted into the network. 

numberPainter.py is a program that allows the user to paint an image and save it locally using a Tkinter window.

datasetViewer.py allows the user to view an entry in the MNIST dataset.
ImageReader.py converts a saved image into a column vector so that it can be inputted into the neural network. 

X.npy, Xtest.npy, Y.npy and Ytest.npy are the MNIST datasets formatted as NumPy variables and saved locally for the other programs to use.

Weights.npy, Bias.npy, Weights1.npy and  Bias1.npy are a pair pretrained weights and bias files for the user to use. They are about 95% on the test dataset.
