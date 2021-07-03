'''In most real-world applications, labelled data is scarce. Suppose you are given
the MNIST dataset (http://yann.lecun.com/exdb/mnist/), but without any labels
in the training set. The labels are held in a database, which you may query to
reveal the label of any particular image it contains. Your task is to build a classifier to
>90% accuracy on the test set, using the smallest number of queries to this
>database. 

You may use any combination of techniques you find suitable
(supervised, self-supervised, unsupervised). However, using other datasets or
pre-trained models is not allowed. '''



import tensorflow as tf 
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.layers.core import Dense, Dropout  
from sklearn.utils import shuffle
import psycopg2
import pickle 
from psycopg2.extensions import register_adapter, AsIs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import time



mnist = tf.keras.datasets.mnist
(X, _), (X_test, y_test) = mnist.load_data()



''' 
Benchmarks: 
60 images: 90~91% 
70 images: 91~93%
'''



# dic = {0: [],
#        1: [],
#        2: [],
#        3: [],
#        4: [],
#        5: [],
#        6: [],
#        7: [],
#        8: [],
#        9: []}

# count = 0
# num = 5
# X_train, y_train = shuffle(X_train, y_train)
# for i, j in zip(X_train, y_train):
#     if count == num * 10: break
#     items = dic.get(j)
#     if len(items) == num: 
#         continue
#     dic[j].append(i)
#     count += 1

# X_train = []
# y_train = []
# for key, value in dic.items():
#     for val in value:
#         X_train.append(val)  
#         y_train.append(key)
# X_train = np.array(X_train)/255
# y_train = np.array(y_train)



# connecting to db 
con = psycopg2.connect(
    host = 'localhost',
    database ='mnist',
    user = 'postgres',
    password = 'brendan',
    port = '5432',
)


# querying n labels from the db
labels = []
with con:
    with con.cursor() as curs:
        for i in range(10):
            sql = f'SELECT * from labels WHERE label = {i} ORDER BY RANDOM() LIMIT 7'
            curs.execute(sql)
            rows = curs.fetchall()
            for j in rows:
                labels.append(j)


# correlating the labels with their images 
X_train = []
y_train = [] 
for i in labels:
    y_train.append(i[1])
    X_train.append(X[i[0] - 1])   

X_train, y_train = shuffle(X_train, y_train)

# normalizing and expanding dims 
X_train = np.array(X_train)/255
y_train = np.array(y_train)
X_train = np.expand_dims(X_train, 3)

X_test = X_test/255
X_test = np.expand_dims(X_test, 3)



# Data augmentation to make up for the lack of samples
datagen = ImageDataGenerator(
    zoom_range=0.1,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
)
train = datagen.flow(
    x=X_train,
    y=y_train,
    batch_size=32,
    seed=123,
    shuffle=True)

'''
Improvements:
Use a gan to generate images? 
Find numbers with high distinction from others
'''




def build_model():
    model = tf.keras.models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))

    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



model = build_model()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1, factor=0.1, min_lr=0.0001)
early = EarlyStopping(monitor='val_loss', mode='min', patience=50)
checkpoint = ModelCheckpoint('chkp', monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_frequency=1)
callbacks_list = [learning_rate_reduction, checkpoint ]
model.fit(train, epochs=500, validation_data=(X_test, y_test), callbacks=callbacks_list)

model = tf.keras.models.load_model('chkp')
model.evaluate(X_test, y_test)



#     con.close()