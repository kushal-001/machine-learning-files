

# %% Importing libraries

import pickle
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt


# %%  Extracting train data

train_file = 'train.p'
train_obj = open(train_file, 'rb')
train_data = pickle.load(train_obj)
train_data

# %%  Extracting test data

test_file = 'test.p'
test_obj = open(test_file, 'rb')
test_data = pickle.load(test_obj)
test_data

# %%  Extracting valid data

valid_file = 'valid.p'
valid_obj = open(valid_file, 'rb')
valid_data = pickle.load(valid_obj)
valid_data
# %%  Extracting training feature and training labels

train_features = train_data['features']
train_labels = train_data['labels']

# %%  Extracting testing feature and testing labels

test_features = test_data['features']
test_labels = test_data['labels']

# %%  Extracting valid feature and testing labels

valid_features = valid_data['features']
valid_labels = valid_data['labels']

# %%  

plt.axis('off')
plt.imshow(valid_features[55])
valid_labels[55]


# %%

for i in range(len(test_features)):
    test_img = test_features[i]

# %%

# for j in range(len(test_labels)):
#     labels = test_labels[j]

# %%
plt.axis('off')
plt.imshow(test_img)

# %%
len(test_img)

# %%

class_names = str(list(train_features)) # You can also take train_lables instead of it
class_names = list(class_names)

#%%  Verify the data


plt.figure(figsize=(20,20))
for i in range(50):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.axis('off')
    plt.imshow(test_features[i])  # cmap=plt.get_cmap('gray')
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[i])
plt.show()

# %%   Import TensorFlow

import keras
# from keras.datasets import mnist
from keras.models import Model
# from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Input ,Dropout, Flatten
from keras.utils import to_categorical
from keras.preprocessing import image
# from keras import backend as K

# from sklearn.model_selection import train_test_split
# from tqdm import tqdm


# %%  Reshape dataset to have a single channel

# img_rows, img_cols = 32,32

# if K.image_data_format() == 'channels_first':
#     x_train =  train_features.reshape(train_features.shape[0],3, img_rows, img_cols)
#     x_test =  test_features.reshape(test_features.shape[0],3, img_rows, img_cols)
#     input_x = (3, img_rows, img_cols)

# else:
#     x_train =  train_features.reshape(train_features.shape[0], img_rows, img_cols, 3)
#     x_test =  test_features.reshape(test_features.shape[0], img_rows, img_cols, 3)
#     input_x = (img_rows, img_cols, 3)    

# %%  Converting into float 

x_train = train_features.astype('float32')
x_test = test_features.astype('float32')
x_valid = valid_features.astype('float32')

# %%  Normalizing 0 to 1

x_train /= 255
x_test /= 255
x_valid /= 255 

# %% Convert class metrix to binary metrix 

# y_train = keras.utils.to_categorical(train_labels)
# y_test = keras.utils.to_categorical(test_labels)

y_train = train_labels
y_test = test_labels
y_valid = valid_labels

# %%   Create the convolutional base

input_x = Input(shape=(32,32,3))
layer_1 = Conv2D(32, kernel_size=(3,3), activation='relu') (input_x)
# layer_2 = Conv2D(64, (3, 3), activation='relu') (layer_1)
layer_3 = MaxPooling2D((2, 2))(layer_1)

layer_4 = Dropout(0.5) (layer_3) # To prevent overfitting

layer_5 = Flatten() (layer_4)
layer_6 = Dense(150, activation='relu') (layer_5)
layer_7 = Dense(80, activation='relu') (layer_6)
layer_8 = Dense(43, activation='softmax') (layer_7)


# %%

# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400,400,3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(25, activation='sigmoid'))


# %%  Displaying the architecture of model

model = Model(input_x, layer_8)
model.summary()


# %%  Compile and train the model


model.compile(
    optimizer='Adam',
    # optimizer='Adadelta',
    # loss=keras.losses.categorical_crossentropy,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )
    
model.fit(
    x_train,
    y_train, 
    epochs=10,
    batch_size=200,
    validation_data=(x_valid, y_valid ),
    # validation_data=(valid_test, y_test )
    shuffle=True
    )

# %%  Evaluate the score 

score = model.evaluate(x_test, y_test, verbose=0)

print('loss=', score[0])
print('accuracy=',score[1])

# %% labels name

label_names = {
    0 : 'Speed limit 20km/h', 
    1 : 'Speed limit 30km/h',
    2 : 'Speed limit 50km/h',
    3 : 'Speed limit 60km/h',
    4 : 'Speed limit 70km/h' , 
    5 : 'Speed limit 80km/h',
    6 : 'End of speed limit 80km/h' ,
    7 : 'Speed limit 100km/h' , 
    8 : 'Speed limit 120km/h' , 
    9 : 'No passing',
    10 : 'No passing for vehicles over 3.5 metric tons',
    11 : 'Rightofway at the next intersection',
    12 : 'Priority road',
    13 : 'Yield' ,
    14 : 'Stop' ,
    15 : 'No vehicles',
    16 : 'Vehicles over 3.5 metric tons prohiited' ,
    17 : 'No entry',
    18 : 'General caution' ,
    19 : 'Dangerous curve to the left',
    20 : 'Dangerous curve to the right' ,
    21 : 'Doule curve',
    22 : 'umpy road' ,
    23 : 'Slippery road',
    24 : 'Road narrows on the right' ,
    25 : 'Road work',
    26 : 'Traffic signals' ,
    27 : 'Pedestrians' ,
    28 : 'Children crossing',
    29 : 'icycles crossing' ,
    30 : 'eware of ice/snow',
    31 : 'Wild animals crossing',
    32 : 'End of all speed and passing limits' ,
    33 : 'Turn right ahead',
    34 : 'Turn left ahead' ,
    35 : 'Ahead only' ,
    36 : 'Go straight or right',
    37 : 'Go straight or left' ,
    38 : 'Keep right' ,
    39 : 'Keep left',
    40 : 'Roundaout mandatory',
    41 : 'End of no passing',
    42 :'End of no passing y vehicles over 3.5 metric tons'
}



labels =label_names.values()
val = list(labels)


# %%  single value prediction

number = 130

prediction = model.predict(x_valid)
warning = np.argmax(np.round(prediction[number]))
print(val[warning])

plt.axis('off')
plt.imshow(x_valid[number], cmap = plt.cm.binary)
plt.show()

# %%
