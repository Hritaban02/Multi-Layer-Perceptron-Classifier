# MACHINE LEARNING GROUP-40 ASSIGNMENT-3

# Neha Dalmia 19CS30055

# Hritaban Ghosh 19CS30053

# DATASET DESCRIPTION
# Context
# This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.

# Content
# Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

# Pregnancies: Number of times pregnant
# Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# BloodPressure: Diastolic blood pressure (mm Hg)
# SkinThickness: Triceps skin fold thickness (mm)
# Insulin: 2-Hour serum insulin (mu U/ml)
# BMI: Body mass index (weight in kg/(height in m)^2)
# DiabetesPedigreeFunction: Diabetes pedigree function
# Age: Age (years)
# Outcome: Class variable (0 or 1)

# Relevant Information:
#   Several constraints were placed on the selection of these instances from
#   a larger database.  In particular, all patients here are females at
#   least 21 years old of Pima Indian heritage.  ADAP is an adaptive learning
#   routine that generates and executes digital analogs of perceptron-like
#   devices.  It is a unique algorithm; see the paper for details.

# Number of Instances: 768
# Number of Attributes: 8 plus class

# For Each Attribute: (all numeric-valued)
# Number of times pregnant
# Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# Diastolic blood pressure (mm Hg)
# Triceps skin fold thickness (mm)
# 2-Hour serum insulin (mu U/ml)
# Body mass index (weight in kg/(height in m)^2)
# Diabetes pedigree function
# Age (years)
# Class variable (0 or 1)

# Missing Attribute Values: Yes
# Class Distribution: (class value 1 is interpreted as "tested positive for diabetes")

# Print Required Information
print("\n###################################################")
print("\n MACHINE LEARNING GROUP-40 ASSIGNMENT-3 \n Neha Dalmia 19CS30055 \n Hritaban Ghosh 19CS30053")
print("\n Dataset Used: https://www.kaggle.com/mathchi/diabetes-data-set")
print("\n###################################################")

# IMPORT REQUIRED LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utility import fit, eval_model
import MLP_models
import r_MLP_models

# IMPORT THE DATASET
df = pd.read_csv("diabetes.csv")
X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values

# Checking for Null Values (Missing Data)

# Checking for null value in the columns
# print("\n  Checking for null values in the dataframe: ")
# print(df.isnull().sum())
# No null value found

# Plot the Attributes Correlation Heatmap for the Liver Disorders Dataset
f, ax = plt.subplots(figsize=(10, 6))
corr = df.iloc[:, 0:-1].corr()
hm = seaborn.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f', linewidths=.05)
f.subplots_adjust(top=0.93)
t = f.suptitle('Diabetes Attributes Correlation Heatmap', fontsize=14)
f.savefig("Analysis/Diabetes_Attributes_Correlation_Heatmap", bbox_inches='tight')
plt.close(f)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Feature Scaling
# Feature Scaling is absolutely compulsory for Deep Learning. Whenever we built an ANN we must apply Feature Scaling.
# It is so fundamental to do this that we are going to scale every feature regardless of whether or not it encodes any categorical data.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)

n_epochs = abs(int(input("  Enter the number of epochs: ")))
print("\n  Start Training\n")

learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
accuracy1 = []
accuracy2 = []
accuracy3 = []
accuracy4 = []
accuracy5 = []

models = ["M1", "M2", "M3", "M4", "M5"]
lr = [[]]

counter = 0
for rate in learning_rates:
    model = MLP_models.Net1()
    fit(model, X_train, y_train, n_epochs=n_epochs, evaluation_set=(X_val, y_val), learning_rate=rate)
    model.load_state_dict(torch.load('model.pt'))
    accuracy1.append(eval_model(model, test_set=(X_test, y_test)))
    lr[counter].append(accuracy1[-1])

    model = MLP_models.Net2()
    fit(model, X_train, y_train, n_epochs=n_epochs, evaluation_set=(X_val, y_val), learning_rate=rate)
    model.load_state_dict(torch.load('model.pt'))
    accuracy2.append(eval_model(model, test_set=(X_test, y_test)))
    lr[counter].append(accuracy2[-1])

    model = MLP_models.Net3()
    fit(model, X_train, y_train, n_epochs=n_epochs, evaluation_set=(X_val, y_val), learning_rate=rate)
    model.load_state_dict(torch.load('model.pt'))
    accuracy3.append(eval_model(model, test_set=(X_test, y_test)))
    lr[counter].append(accuracy3[-1])

    model = MLP_models.Net4()
    fit(model, X_train, y_train, n_epochs=n_epochs, evaluation_set=(X_val, y_val), learning_rate=rate)
    model.load_state_dict(torch.load('model.pt'))
    accuracy4.append(eval_model(model, test_set=(X_test, y_test)))
    lr[counter].append(accuracy4[-1])

    model = MLP_models.Net5()
    fit(model, X_train, y_train, n_epochs=n_epochs, evaluation_set=(X_val, y_val), learning_rate=rate)
    model.load_state_dict(torch.load('model.pt'))
    accuracy5.append(eval_model(model, test_set=(X_test, y_test)))
    lr[counter].append(accuracy5[-1])

    counter = counter + 1
    lr.append([])

plt.xscale("log")
plt.plot(learning_rates, accuracy1, label="Model1")
plt.plot(learning_rates, accuracy2, label="Model2")
plt.plot(learning_rates, accuracy3, label="Model3")
plt.plot(learning_rates, accuracy4, label="Model4")
plt.plot(learning_rates, accuracy5, label="Model5")
plt.title("Accuracy versus Learning Rates for Preprocessed Data of dimension 8")
plt.xlabel('Learning Rates')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend()
plt.savefig("Analysis/Accuracy_versus_Models_with_Learning_Rates_for_Preprocessed_Data_of_dimension_8", bbox_inches='tight')
plt.close()

plt.plot(models, lr[0], label="lr1")
plt.plot(models, lr[1], label="lr2")
plt.plot(models, lr[2], label="lr3")
plt.plot(models, lr[3], label="lr4")
plt.plot(models, lr[4], label="lr5")
plt.title("Accuracy versus Models with Increasing Complexity for Preprocessed Data of dimension 8")
plt.xlabel('Models with Increasing Complexity')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend()
plt.savefig("Analysis/Accuracy_versus_Models_with_Increasing_Complexity_for_Preprocessed_Data_of_dimension_8", bbox_inches='tight')
plt.close()
print("\n  Done\n")

# After Principal Component Analysis
print("\n  ###########################################\n")
print("\n  Start Training With Reduced Dimensional Data after PCA\n")

pca = PCA(n_components=2)
PCA_reduced_X_train = pca.fit_transform(X_train)
PCA_reduced_X_test = pca.transform(X_test)
PCA_reduced_X_val = pca.transform(X_val)

plt.scatter(PCA_reduced_X_train[y_train == 0, 0], PCA_reduced_X_train[y_train == 0, 1], s=50, color='green',
            label='Class 0')
plt.scatter(PCA_reduced_X_train[y_train == 1, 0], PCA_reduced_X_train[y_train == 1, 1], s=50, color='red',
            label='Class 1')
plt.scatter(PCA_reduced_X_test[y_test == 0, 0], PCA_reduced_X_test[y_test == 0, 1], s=50, color='green')
plt.scatter(PCA_reduced_X_test[y_test == 1, 0], PCA_reduced_X_test[y_test == 1, 1], s=50, color='red')
plt.scatter(PCA_reduced_X_val[y_val == 0, 0], PCA_reduced_X_val[y_val == 0, 1], s=50, color='green')
plt.scatter(PCA_reduced_X_val[y_val == 1, 0], PCA_reduced_X_val[y_val == 1, 1], s=50, color='red')
plt.title("Plot of Reduced Dimensional Data in a 2D Plane")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.savefig("Analysis/Plot_of_Reduced_Dimensional_Data_in_a_2D_Plane", bbox_inches='tight')
plt.close()

learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
accuracy1 = []
accuracy2 = []
accuracy3 = []
accuracy4 = []
accuracy5 = []

models = ["M1", "M2", "M3", "M4", "M5"]
lr = [[]]

if counter != 0:
    counter = 0
for rate in learning_rates:
    model = r_MLP_models.Net1()
    fit(model, PCA_reduced_X_train, y_train, n_epochs=n_epochs, evaluation_set=(PCA_reduced_X_val, y_val), learning_rate=rate)
    model.load_state_dict(torch.load('model.pt'))
    accuracy1.append(eval_model(model, test_set=(PCA_reduced_X_test, y_test)))
    lr[counter].append(accuracy1[-1])

    model = r_MLP_models.Net2()
    fit(model, PCA_reduced_X_train, y_train, n_epochs=n_epochs, evaluation_set=(PCA_reduced_X_val, y_val), learning_rate=rate)
    model.load_state_dict(torch.load('model.pt'))
    accuracy2.append(eval_model(model, test_set=(PCA_reduced_X_test, y_test)))
    lr[counter].append(accuracy2[-1])

    model = r_MLP_models.Net3()
    fit(model, PCA_reduced_X_train, y_train, n_epochs=n_epochs, evaluation_set=(PCA_reduced_X_val, y_val), learning_rate=rate)
    model.load_state_dict(torch.load('model.pt'))
    accuracy3.append(eval_model(model, test_set=(PCA_reduced_X_test, y_test)))
    lr[counter].append(accuracy3[-1])

    model = r_MLP_models.Net4()
    fit(model, PCA_reduced_X_train, y_train, n_epochs=n_epochs, evaluation_set=(PCA_reduced_X_val, y_val), learning_rate=rate)
    model.load_state_dict(torch.load('model.pt'))
    accuracy4.append(eval_model(model, test_set=(PCA_reduced_X_test, y_test)))
    lr[counter].append(accuracy4[-1])

    model = r_MLP_models.Net5()
    fit(model, PCA_reduced_X_train, y_train, n_epochs=n_epochs, evaluation_set=(PCA_reduced_X_val, y_val), learning_rate=rate)
    model.load_state_dict(torch.load('model.pt'))
    accuracy5.append(eval_model(model, test_set=(PCA_reduced_X_test, y_test)))
    lr[counter].append(accuracy5[-1])

    counter = counter + 1
    lr.append([])

plt.xscale("log")
plt.plot(learning_rates, accuracy1, label="Model1")
plt.plot(learning_rates, accuracy2, label="Model2")
plt.plot(learning_rates, accuracy3, label="Model3")
plt.plot(learning_rates, accuracy4, label="Model4")
plt.plot(learning_rates, accuracy5, label="Model5")
plt.title("Accuracy versus Learning Rates for Reduced Dimensional Data after PCA")
plt.xlabel('Learning Rates')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend()
plt.savefig("Analysis/Accuracy_versus_Models_with_Learning_Rates_for_Reduced_Dimensional_Data_after_PCA", bbox_inches='tight')
plt.close()

plt.plot(models, lr[0], label="lr1")
plt.plot(models, lr[1], label="lr2")
plt.plot(models, lr[2], label="lr3")
plt.plot(models, lr[3], label="lr4")
plt.plot(models, lr[4], label="lr5")
plt.title("Accuracy versus Models with Increasing Complexity for Reduced Dimensional Data after PCA")
plt.xlabel('Models with Increasing Complexity')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend()
plt.savefig("Analysis/Accuracy_versus_Models_with_Increasing_Complexity_for_Reduced_Dimensional_Data_after_PCA", bbox_inches='tight')
plt.close()
print("\n  Done\n")

print("\n  ###########################################\n")

# # BATCH LEARNING
#
# batch_learning = True
# print("\n  ###########################################\n")
# batch_size = int(input("  Enter the batch size for batch learning: "))
# print("\n  Start Training With Batch Learning\n")
#
# learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
# accuracy1 = []
# accuracy2 = []
# accuracy3 = []
# accuracy4 = []
# accuracy5 = []
#
# models = ["M1", "M2", "M3", "M4", "M5"]
# lr = [[]]
#
# if counter != 0:
#     counter = 0
# for rate in learning_rates:
#     model = MLP_models.Net1()
#     fit(model, X_train, y_train, n_epochs=n_epochs, evaluation_set=(X_val, y_val), learning_rate=rate, batch_learning=batch_learning, batch_size=batch_size)
#     model.load_state_dict(torch.load('model.pt'))
#     accuracy1.append(eval_model(model, test_set=(X_test, y_test)))
#     lr[counter].append(accuracy1[-1])
#
#     model = MLP_models.Net2()
#     fit(model, X_train, y_train, n_epochs=n_epochs, evaluation_set=(X_val, y_val), learning_rate=rate, batch_learning=batch_learning, batch_size=batch_size)
#     model.load_state_dict(torch.load('model.pt'))
#     accuracy2.append(eval_model(model, test_set=(X_test, y_test)))
#     lr[counter].append(accuracy2[-1])
#
#     model = MLP_models.Net3()
#     fit(model, X_train, y_train, n_epochs=n_epochs, evaluation_set=(X_val, y_val), learning_rate=rate, batch_learning=batch_learning, batch_size=batch_size)
#     model.load_state_dict(torch.load('model.pt'))
#     accuracy3.append(eval_model(model, test_set=(X_test, y_test)))
#     lr[counter].append(accuracy3[-1])
#
#     model = MLP_models.Net4()
#     fit(model, X_train, y_train, n_epochs=n_epochs, evaluation_set=(X_val, y_val), learning_rate=rate, batch_learning=batch_learning, batch_size=batch_size)
#     model.load_state_dict(torch.load('model.pt'))
#     accuracy4.append(eval_model(model, test_set=(X_test, y_test)))
#     lr[counter].append(accuracy4[-1])
#
#     model = MLP_models.Net5()
#     fit(model, X_train, y_train, n_epochs=n_epochs, evaluation_set=(X_val, y_val), learning_rate=rate, batch_learning=batch_learning, batch_size=10)
#     model.load_state_dict(torch.load('model.pt'))
#     accuracy5.append(eval_model(model, test_set=(X_test, y_test)))
#     lr[counter].append(accuracy5[-1])
#
#     counter = counter + 1
#     lr.append([])
#
# plt.xscale("log")
# plt.plot(learning_rates, accuracy1, label="Model1")
# plt.plot(learning_rates, accuracy2, label="Model2")
# plt.plot(learning_rates, accuracy3, label="Model3")
# plt.plot(learning_rates, accuracy4, label="Model4")
# plt.plot(learning_rates, accuracy5, label="Model5")
# plt.title("Accuracy versus Learning Rates for Preprocessed Data of dimension 8 (Batch Learning)")
# plt.xlabel('Learning Rates')
# plt.ylabel('Accuracy')
# plt.ylim([0, 1])
# plt.legend()
# plt.savefig("Analysis/Accuracy_versus_Models_with_Learning_Rates_for_Preprocessed_Data_of_dimension_8_Batch_Learning", bbox_inches='tight')
# plt.close()
#
# plt.plot(models, lr[0], label="lr1")
# plt.plot(models, lr[1], label="lr2")
# plt.plot(models, lr[2], label="lr3")
# plt.plot(models, lr[3], label="lr4")
# plt.plot(models, lr[4], label="lr5")
# plt.title("Accuracy versus Models with Increasing Complexity for Preprocessed Data of dimension 8 (Batch Learning)")
# plt.xlabel('Models with Increasing Complexity')
# plt.ylabel('Accuracy')
# plt.ylim([0, 1])
# plt.legend()
# plt.savefig("Analysis/Accuracy_versus_Models_with_Increasing_Complexity_for_Preprocessed_Data_of_dimension_8_Batch_Learning", bbox_inches='tight')
# plt.close()
# print("\n  Done\n")
#
# # After Principal Component Analysis
# print("\n  ###########################################\n")
# print("\n  Start Training With Batch Learning With Reduced Dimensional Data after PCA\n")
#
# learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
# accuracy1 = []
# accuracy2 = []
# accuracy3 = []
# accuracy4 = []
# accuracy5 = []
#
# models = ["M1", "M2", "M3", "M4", "M5"]
# lr = [[]]
#
# if counter != 0:
#     counter = 0
# for rate in learning_rates:
#     model = r_MLP_models.Net1()
#     fit(model, PCA_reduced_X_train, y_train, n_epochs=n_epochs, evaluation_set=(PCA_reduced_X_val, y_val), learning_rate=rate, batch_learning=batch_learning, batch_size=10)
#     model.load_state_dict(torch.load('model.pt'))
#     accuracy1.append(eval_model(model, test_set=(PCA_reduced_X_test, y_test)))
#     lr[counter].append(accuracy1[-1])
#
#     model = r_MLP_models.Net2()
#     fit(model, PCA_reduced_X_train, y_train, n_epochs=n_epochs, evaluation_set=(PCA_reduced_X_val, y_val), learning_rate=rate, batch_learning=batch_learning, batch_size=10)
#     model.load_state_dict(torch.load('model.pt'))
#     accuracy2.append(eval_model(model, test_set=(PCA_reduced_X_test, y_test)))
#     lr[counter].append(accuracy2[-1])
#
#     model = r_MLP_models.Net3()
#     fit(model, PCA_reduced_X_train, y_train, n_epochs=n_epochs, evaluation_set=(PCA_reduced_X_val, y_val), learning_rate=rate, batch_learning=batch_learning, batch_size=10)
#     model.load_state_dict(torch.load('model.pt'))
#     accuracy3.append(eval_model(model, test_set=(PCA_reduced_X_test, y_test)))
#     lr[counter].append(accuracy3[-1])
#
#     model = r_MLP_models.Net4()
#     fit(model, PCA_reduced_X_train, y_train, n_epochs=n_epochs, evaluation_set=(PCA_reduced_X_val, y_val), learning_rate=rate, batch_learning=batch_learning, batch_size=10)
#     model.load_state_dict(torch.load('model.pt'))
#     accuracy4.append(eval_model(model, test_set=(PCA_reduced_X_test, y_test)))
#     lr[counter].append(accuracy4[-1])
#
#     model = r_MLP_models.Net5()
#     fit(model, PCA_reduced_X_train, y_train, n_epochs=n_epochs, evaluation_set=(PCA_reduced_X_val, y_val), learning_rate=rate, batch_learning=batch_learning, batch_size=10)
#     model.load_state_dict(torch.load('model.pt'))
#     accuracy5.append(eval_model(model, test_set=(PCA_reduced_X_test, y_test)))
#     lr[counter].append(accuracy5[-1])
#
#     counter = counter + 1
#     lr.append([])
#
# plt.xscale("log")
# plt.plot(learning_rates, accuracy1, label="Model1")
# plt.plot(learning_rates, accuracy2, label="Model2")
# plt.plot(learning_rates, accuracy3, label="Model3")
# plt.plot(learning_rates, accuracy4, label="Model4")
# plt.plot(learning_rates, accuracy5, label="Model5")
# plt.title("Accuracy versus Learning Rates for Reduced Dimensional Data after PCA (Batch Learning)")
# plt.xlabel('Learning Rates')
# plt.ylabel('Accuracy')
# plt.ylim([0, 1])
# plt.legend()
# plt.savefig("Analysis/Accuracy_versus_Models_with_Learning_Rates_for_Reduced_Dimensional_Data_after_PCA_Batch_Learning", bbox_inches='tight')
# plt.close()
#
# plt.plot(models, lr[0], label="lr1")
# plt.plot(models, lr[1], label="lr2")
# plt.plot(models, lr[2], label="lr3")
# plt.plot(models, lr[3], label="lr4")
# plt.plot(models, lr[4], label="lr5")
# plt.title("Accuracy versus Models with Increasing Complexity for Reduced Dimensional Data after PCA (Batch Learning)")
# plt.xlabel('Models with Increasing Complexity')
# plt.ylabel('Accuracy')
# plt.ylim([0, 1])
# plt.legend()
# plt.savefig("Analysis/Accuracy_versus_Models_with_Increasing_Complexity_for_Reduced_Dimensional_Data_after_PCA_Batch_Learning", bbox_inches='tight')
# plt.close()
# print("\n  Done\n")
#
# print("\n  ###########################################\n")
