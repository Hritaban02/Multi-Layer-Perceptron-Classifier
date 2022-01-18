# README
# Brief Description:
---
The __Analysis__ directory contains the following files and folders:
- __1) Analysis With Preprocessed Data of Dimension 8__ : This folder contains the following graphs using which we analyse the model with standardized data of dimension 8:
    - __Accuracy_versus_Models_with_Increasing_Complexity_for_Preprocessed_Data_of_dimension_8.png__: This graph plots the Accuracy versus the Models of Increasing Complexity for all learning rates. For different model specifications, the accuracy attained is recorded and plotted on this graph.
    - __Accuracy_versus_Models_with_Learning_Rates_for_Preprocessed_Data_of_dimension_8.png__: This graph plots the Accuracy versus the Learning Rates for all models. For different learning rates, the accuracy attained is recorded and plotted on this graph.
- __2) Analysis With Reduced Dimensional Data after PCA__: This folder contains the following graphs obtained after performing the Test_A as described in the assignment:
    - __Accuracy_versus_Models_with_Increasing_Complexity_for_Reduced_Dimensional_Data_after_PCA.png__: This graph plots the Accuracy versus the Models of Increasing Complexity for all learning rates after PCA has been performed on the dataset. For different model specifications, the accuracy attained is recorded and plotted on this graph. 
    - __Accuracy_versus_Models_with_Learning_Rates_for_Reduced_Dimensional_Data_after_PCA.png__: This graph plots the Accuracy versus the Learning Rates for all models after PCA has been performed on the dataset. For different learning rates, the accuracy attained is recorded and plotted on this graph.
- __Diabetes_Attributes_Correlation_Heatmap.png__ : This image is the correlation matrix obtained on the features of the Diabetes dataset.
- __Plot_of_Reduced_Dimensional_Data_in_a_2D_Plane.png__ : This image is the 2D plot of all the data samples in the dataset whose data points are colored differently based on class values.
- __Report.pdf__: Contains a detailed analysis report of our Multi Layered Perceptron Models using different hyperparameters such as n_epochs, learning_rate, batch_learning etc. and a section of extra analysis that we have performed to have better understanding of our models.

The __Project__ directory contains the following files and folders:
- __Analysis__ : Analysis is a folder which gets populated when analysis.py is run
- __diabetes.csv__ : Contains the data used for training the Multi Layered Perceptron Models. The data is comprised of 768 samples with 8 continuous valued attributes and 1 binary valued target label.
- __requirements.txt__ : Contains all the necessary dependencies and their versions
- __utility.py__ : Contains all the helper functions such as eval_model etc.
- __MLP_models.py__ : Contains the implementation of all the Multi Layered Perceptron Models when the number of input nodes is 8 and output node is 1. [BEFORE PCA]
- __r_MLP_models.py__ : Contains the implementation of all the Multi Layered Perceptron Models when the number of input nodes is 2 and output node is 1. [AFTER PCA]
- __analysis.py__ : Contains the python code we implemented to perform analysis on the Data and  Multi Layered Perceptron Models.

# Directions to use the code
---
1. Download the __Project__ directory in  into your local machine.
2. Ensure all the necessary dependencies with required version and latest version of Python3 are available (verify with requirements.txt)
```sh
pip3 install -r requirements.txt
```
3. Run the analysis.py file.
    - Initially, it will output some information regarding the project such as author etc.
    - Next, it will __prompt you to enter the value of n_epochs__, which is the number of epochs for which all the Multi Layered Perceptron Models will be trained. The value of n_epochs has to be greater than or equal to 0. Please enter an integer value (n_epochs>=0). [We suggest the value of 500]
    - Now, all the models are built with the above user given n_epochs as a parameter for all learning rates and all the plots are automatically generated and saved in their respective folders. (More details in the Report.pdf)
    - After that, Principal Component Analysis is performed on the dataset and its dimension is reduced to 2.
    -  Again, all the models are built with the above user given n_epochs as a parameter for all learning rates and all the plots are automatically generated and saved in their respective folders. (More details in the Report.pdf)

# Final Remarks 
---
The dataset is taken from the following link. 
https://www.kaggle.com/mathchi/diabetes-data-set









  
