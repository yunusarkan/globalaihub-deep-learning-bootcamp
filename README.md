# globalaihub-deep-learning-bootcamp
This project done together with [Hilal Küçük](https://github.com/hllkck) for Global AI Hub's Deep Learning Bootcamp.
In this project, a deep learning model was created using [a large-scale fish dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset) on Kaggle. 

1. Data Loading and Preparation
The dataset directory is navigated using the os library, and two lists named label and path are created. Using Dataframe, the file paths of the images in the dataset are assigned to the path[] array, and the labels obtained from the file paths are assigned to the label[] array.

2. Processing of Images
Images are read using cv2 (OpenCV), each is converted to 128x128 pixels, and normalization is performed to the range of 0-1. Then, pixel values ​​are converted to a NumPy array.

3. Preparing Labels
Fish species are labeled using LabelEncoder and categorized using to_categorical.

4. Partitioning the Dataset
Training, validation and test datasets are created. The dataset is divided into 80% training and 20% temporary dataset, then the temporary dataset is divided into 60% validation and 40% testing.

5. Creating the Model
A deep learning model is established with Keras Sequential API. The 128-neuron input layer is created with the relu activation function, then 3 hidden layers containing 256, 128 and 64 neurons respectively and 0.3 ratio dropout layers accompanying these layers are added. The output layer contains 9 neurons and uses the softmax function.
Overfitting is tried to be prevented by using dropout layers.

6. Training the Model
The model is trained with Adam Optimizer and categorical_crossentropy loss function. Validation loss is monitored using early stopping and the training of the model is stopped when necessary.

7. Evaluating the Model Performance
The model is evaluated with the test dataset, the accuracy value and loss function are calculated. Afterwards, the performance of the model is visualized with graphs, the graphical output is shown in Figure 1 and Figure 2.
![Figure 1](https://github.com/yunusarkan/globalaihub-deep-learning-bootcamp/blob/main/model_output.jpg)
<p align="center">Figure 1</p>

![Figure 2](https://github.com/yunusarkan/globalaihub-deep-learning-bootcamp/blob/main/test_output.jpg)
<p align="center">Figure 2</p>

The results are analyzed in detail with the classification report and confusion matrix, the analysis output is shown in Figure 3.

![Figure 3](https://github.com/yunusarkan/globalaihub-deep-learning-bootcamp/blob/main/matrix.jpg)
<p align="center">Figure 3</p>

You can access the Kaggle notebook of the project from this [link](https://www.kaggle.com/code/hilalkk/globalaihub-deep-learning-bootcamp) .
