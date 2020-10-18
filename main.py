#Importing Modules
import tensorflow as tf
import zipfile
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import shutil

#Zip Data Path
training_data_file = "C:/Path/To/IMG Sorter/train.zip"

with zipfile.ZipFile(training_data_file, 'r') as z:
    z.extractall()

#Defining Functions
def load_data(file_path):
    return cv2.imread(file_path)

def extract_label(file_name):
    return 1 if "nsfw" in file_name else 0 #Input Data Weights

def preprocess_image(img, side=96):
    min_side = min(img.shape[0], img.shape[1])
    img = img[:min_side, :min_side]
    img = cv2.resize(img, (side,side))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img/255.0

train_path = "C:/Path/To/IMG Sorter/train/" #Zip File Dir
image_files = os.listdir(train_path)
train_images = [load_data(train_path + file) for file in image_files]
train_labels = [extract_label(file) for file in image_files]


preview_index = 200
plt.subplot(1,2,1)
plt.imshow(train_images[preview_index])
plt.subplot(1,2,2)

for i in range(len(train_images)):
    train_images[i] = preprocess_image(train_images[i])

train_images = np.expand_dims(train_images, axis=-1)
train_labels = np.array(train_labels)

#Creating 2 Layer CNN
layers = [
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", activation=tf.nn.relu, input_shape=train_images.shape[1:]),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1024, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)
]

model = tf.keras.Sequential(layers)
model.compile(optimizer = tf.optimizers.Adam(),
              loss = tf.losses.SparseCategoricalCrossentropy(),
              metrics = [tf.metrics.SparseCategoricalAccuracy()])
model.fit(train_images, train_labels, epochs=1, batch_size=50)#Epochs Are How Many Times Training Occurs
model.save_weights("C:/Path/To/IMG Sorter/model.h5")#Saving trained model to current directory

#Setting paths to image folders
nsfw_path = "C:/Path/To/IMG Sorter/Sorted_NSFW/"
sfw_path = "C:/Path/To/IMG Sorter/Sorted_SFW/"
eval_path = "C:/Path/To/IMG Sorter/Unsorted_Data/"
all_path = "C:/Path/To/IMG Sorter/All_Data/"

list1 = os.listdir(all_path)
num_files = (len(list1))#Get num of files in all_data path
num_files = num_files#Not sure why I put this here but it might be important so I'll leave it

eval_files = os.listdir(eval_path)
eval_images = [preprocess_image(load_data(eval_path + file)) for file in eval_files]
eval_model = tf.keras.Sequential(layers)
eval_model.load_weights("C:/Path/To/IMG Sorter/model.h5")#Load pretrained model
eval_predictions = eval_model.predict(np.expand_dims(eval_images, axis=-1))

###Start CNN Evaluation Cycle###
for i in range(len(eval_images)):
    files = i
    file_address = (str(files) + ".jpg")
    file_path = (eval_path + file_address)

    #Sending Sorted Data To Respective Folders
    if np.argmax(eval_predictions[i])==1:#Passing unsorted data to model
        shutil.move(file_path, nsfw_path)#Move file to nsfw folder

    else:
        shutil.move(file_path, sfw_path)#move data to sfw folder

print("---ALL TASKS COMPLETE!---")
###End CNN Evaluation Cycle###
