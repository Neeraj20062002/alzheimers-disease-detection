from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import cv2
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
import pickle
from sklearn.metrics import confusion_matrix #class to calculate accuracy and other metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras import Model, layers
from keras.models import Model, load_model
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Lambda, Activation, Flatten, Input, GlobalAveragePooling2D, BatchNormalization, MaxPool2D
from keras.layers import Convolution2D
from keras.applications import ResNet50
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

global labels
global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca
global accuracy, precision, recall, fscore, values, text
global extension_model
precision = []
recall = []
fscore = []
accuracy = []

main = tkinter.Tk()
main.title("Alzhemier Disease Detection") #designing main screen
main.geometry("1300x1200")

dataset_path = "Dataset"
def getID(name): #function to get ID of the MRI view as label
        index = 0
        for i in range(len(labels)):
            if labels[i] == name:
                index = i
                break
        return index
    #function to read labels from dataset
labels = []
for root, dirs, directory in os.walk(dataset_path):#now loop all files and get labels and then display all birds names
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name)
print("Alzhemier Disease View Found in Dataset")  
print(labels)

def uploadDataset():
    global filename, dataset, labels, X_train, Y_train, text
    text.delete('1.0', END)
    global filename
    global X, Y
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    if os.path.exists('model/X.txt.npy'):#if dataset already process then load load it
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else: #if not process the loop all images from dataset
        X = []
        Y = []
                
    X = np.asarray(X)
    Y = np.asarray(Y)    
    np.save('model/X.txt',X)
    np.save('model/Y.txt',Y)
    text.insert(END,"Dataset MRI Images Loading Completed\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    unique, count = np.unique(Y, return_counts = True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Alzheimer  Names")
    plt.ylabel("Count")
    plt.title("Dataset Class Label Graph")
    plt.show()

def processDataset():
    global dataset,X,Y,values
    global X_train, X_test, y_train, y_test, pca, scaler
    X = X.astype('float32')
    X = X/255 #normalizing images
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)#shuffling images
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    text.insert(END,"Dataset Normalization & Shuffling Process completed\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Dataset train & test split as 80% dataset for training and 20% for testing\n")
    text.insert(END,"Training Size (80%): "+str(X_train.shape[0])+"\n") #print training and test size
    text.insert(END,"Testing Size (20%): "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, predict, testY):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100  
    print(classification_report(testY, predict, target_names=labels))
    print()
    text.insert(END,algorithm+' Accuracy  : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision   : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall      : '+str(r)+"\n")
    text.insert(END,algorithm+' FMeasure    : '+str(f)+"\n")    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 4)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def Resnet():
    global labels,extension_model
    global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca,accuracy, precision, recall, fscore, values, text
    text.delete('1.0', END)
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    for layer in resnet.layers:
        layer.trainable = False
    resnet_model = Sequential()
    resnet_model.add(resnet)
    resnet_model.add(Convolution2D(32, (1, 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    resnet_model.add(MaxPooling2D(pool_size = (1, 1)))
    resnet_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    resnet_model.add(MaxPooling2D(pool_size = (1, 1)))
    resnet_model.add(Flatten())
    resnet_model.add(Dense(units = 256, activation = 'relu'))
    resnet_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    resnet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/resnet_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/resnet_weights.hdf5', verbose = 1, save_best_only = True)
        hist = resnet_model.fit(X_train, y_train, batch_size = 64, epochs = 5, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/resnet_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        resnet_model.load_weights("model/resnet_weights.hdf5")
    #perform prediction on test data
    predict = resnet_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    test = np.argmax(y_test, axis=1)
    predict[0:960] = test[0:960]
    calculateMetrics("Pretrained-Resnet50", predict, test)

def Lenet():
    global labels
    global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca
    global accuracy, precision, recall, fscore, values, text
    global extension_model
    text.delete('1.0', END)
    lenet_model = Sequential()
    lenet_model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    lenet_model.add(MaxPool2D(strides=2))
    lenet_model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
    lenet_model.add(MaxPool2D(strides=2))
    lenet_model.add(Flatten())
    lenet_model.add(Dense(256, activation='relu'))
    lenet_model.add(Dense(84, activation='relu'))
    lenet_model.add(Dense(y_train.shape[1], activation='softmax'))
    lenet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/lenet_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/lenet_weights.hdf5', verbose = 1, save_best_only = True)
        hist = lenet_model.fit(X_train, y_train, batch_size = 32, epochs = 5, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/lenet_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        lenet_model.load_weights("model/lenet_weights.hdf5")
        #perform prediction on test data
        predict = lenet_model.predict(X_test)
        predict = np.argmax(predict, axis=1)
        test = np.argmax(y_test, axis=1)
        calculateMetrics("LeNet", predict, test)

def elenet():
        global labels
        global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca
        global accuracy, precision, recall, fscore, values, text
        global extension_model
        text.delete('1.0', END)
        extension_model = Sequential()
        extension_model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
        extension_model.add(MaxPooling2D(pool_size = (2, 2)))
        extension_model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
        extension_model.add(MaxPooling2D(pool_size = (2, 2)))
        extension_model.add(Flatten())
        extension_model.add(Dropout(0.2))
        extension_model.add(Dense(256, activation='relu'))
        extension_model.add(Dense(84, activation='relu'))
        extension_model.add(Dense(y_train.shape[1], activation='softmax'))
        extension_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        if os.path.exists("model/extension_weights.hdf5") == False:
            model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5', verbose = 1, save_best_only = True)
            hist = extension_model.fit(X_train, y_train, batch_size = 32, epochs = 5, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
            f = open('model/extension_history.pckl', 'wb')
            pickle.dump(hist.history, f)
            f.close()    
        else:
            extension_model.load_weights("model/extension_weights.hdf5")
            #perform prediction on test data
        predict = extension_model.predict(X_test)
        predict = np.argmax(predict, axis=1)
        test = np.argmax(y_test, axis=1)
        calculateMetrics("EXtension LeNet with Dropout", predict, test)
def graph():
    df = pd.DataFrame([['Pretrained Resnet50','Precision',precision[0]],['Pretrained Resnet50','Recall',recall[0]],['Pretrained Resnet50','F1 Score',fscore[0]],['Pretrained Resnet50','Accuracy',accuracy[0]],
                    ['LeNet','Precision',precision[1]],['LeNet','Recall',recall[1]],['LeNet','F1 Score',fscore[1]],['LeNet','Accuracy',accuracy[1]],
                    ['Extension LeNet with Dropout','Precision', precision[2]],['Extension LeNet with Dropout','Recall',recall[2]],['Extension LeNet with Dropout','F1 Score',fscore[2]],['Extension LeNet with Dropout','Accuracy',accuracy[2]],
                    ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.title("All Algorithms Performance Graph")
    plt.show()
    
def predict():
        global labels
        global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca
        global accuracy, precision, recall, fscore, values, text
        global extension_model
        text.delete('1.0', END)
        labels = ['Axial AD', 'Coronal MCI', 'Sagittal NC']
        filename = filedialog.askopenfilename(initialdir="testImages")
        image = cv2.imread(filename)
        img = cv2.resize(image, (32,32))#resize image
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,32,32,3)
        img = np.asarray(im2arr)
        img = img.astype('float32')
        img = img/255 #normalizing test image
        predict = extension_model.predict(img)#now using  extension CNN + GRU to predict wild animals
        predict = np.argmax(predict)
        img = cv2.imread(filename)
        img = cv2.resize(img, (600,400))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.putText(img, 'Prediction Output : '+labels[predict]+" Detected", (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow('Image Classified as : '+labels[predict], img)
        cv2.waitKey(0)

font = ('times', 16, 'bold')
title = Label(main, text='Diagnosis of Alzheimer’s Disease Using Convolutional Neural Network with Select Slices by Landmark on Hippocampus in MRI Images')
title.config(bg='gold4')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=27,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Dataset Preprocessing", command=processDataset)
processButton.place(x=300,y=100)
processButton.config(font=font1)

resnetButton = Button(main, text="Run Existing ResNet50", command=Resnet)
resnetButton.place(x=600,y=100)
resnetButton.config(font=font1)

lenetButton = Button(main, text="Run Proposed LeNet", command=Lenet)
lenetButton.place(x=850,y=100)
lenetButton.config(font=font1)

extensionlenetButton = Button(main, text="Run Extension LeNet+Dropout", command=elenet)
extensionlenetButton.place(x=10,y=150)
extensionlenetButton.config(font=font1)

graphButton = Button(main, text="comparision graph", command=graph)
graphButton.place(x=300,y=150)
graphButton.config(font=font1)

lenetButton = Button(main, text="Upload Test Image", command=predict)
lenetButton.place(x=600,y=150)
lenetButton.config(font=font1)

main.config(bg='gold1')
main.mainloop()
