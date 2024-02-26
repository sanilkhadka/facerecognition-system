import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model

def identify_face(facearray):
    model = keras.models.load_model('static/face_recognition_model.h5')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face)
            labels.append(user)
    faces = np.array(faces)
    labels = np.array(labels)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(faces, keras.utils.to_categorical(labels), epochs=10, batch_size=32)

    # Save the model
    model.save('static/face_recognition_model.h5')


model = load_model('facerecognizationmodel.h5')

from keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'static/dataset'
validation_data_dir = 'static/dataset'
img_width, img_height = 224, 224
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='binary')
model.save('path/to/retrained_model.h5')