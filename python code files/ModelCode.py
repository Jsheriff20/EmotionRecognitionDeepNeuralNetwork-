from __future__ import print_function

import pickle

import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import time


#setting up a unique name so model can be tracked in tensorboard
name = "{}-optimizer-{}-additionLayers-{}-additionalChanges-baseLineTest-{}-".format("Adam", "none" , "final" , int(time.time()))
tensorBoard = TensorBoard(log_dir="final_tests\\{}".format(name))

#setting up infomation for the model
num_classes = 6
img_rows,img_cols = 48,48
batch_size = 64
nb_train_samples = 36260
nb_validation_samples = 4243
epochs=50


#loading data in
xTrain = pickle.load(open("xtrainPickleGreyScale.pickle", "rb"))
yTrain = pickle.load(open("ytrainPickleGreyScale.pickle", "rb"))
xVal = pickle.load(open("xvalidationPickleGreyScale.pickle", "rb"))
yVal = pickle.load(open("yvalidationPickleGreyScale.pickle", "rb"))

#preparing data so that the model can understand it
yTrain = np.array(yTrain)
yTrain = keras.utils.to_categorical(yTrain, num_classes=6)
yVal = np.array(yVal)
yVal = keras.utils.to_categorical(yVal, num_classes=6)

#argumenting the data
train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					width_shift_range=0.4,
					height_shift_range=0.4,
					horizontal_flip=True,
					fill_mode='nearest')

#normalising the data
validation_datagen = ImageDataGenerator(rescale=1./255)

#preparing the training data
train_generator = train_datagen.flow(
					xTrain,
                    yTrain,
					batch_size=batch_size,
					shuffle=True)

#preparing the validation data
validation_generator = validation_datagen.flow(
                    xVal,
                    yVal,
					batch_size=batch_size,
					shuffle=True)


#building the model
model = Sequential()

# Block-1
#input layer
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())

# Block-2
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-3
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())



# Block-4
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-5
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())

# Block-6
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-7
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())

# Block-8
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-9
model.add(Conv2D(512,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())

# Block-10
model.add(Conv2D(512,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))



# Block-11
#flattening the model so that the training weights can be understoop in a dense layer
model.add(Flatten())
model.add(Dense(128,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))


# Block-12
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())

# Block-13
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


# Block-14
#output layer
model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

#output a summary of how the model is built
print(model.summary())

#setting up checkpoints for the model to fall back to
checkpoint = ModelCheckpoint('emotionDetectionViaFace.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

#setting up early stops for the model to be able to stop early
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=5,
                          verbose=2,
                          restore_best_weights=True
                          )

#setting up Recude Learning Rate on Platau so models learning rate can change dynamically
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=5,
                              verbose=2,
                              min_delta=0.0001)

##setting up the callbacks
callbacks = [earlystop,checkpoint,reduce_lr, tensorBoard]

#compling the model together
model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

#fit the model so that it can start to learn
history=model.fit(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)

#saving the model
model.save("emotionDetectionViaFace.h5")