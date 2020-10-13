#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np 
import pandas as pd 
import cv2


# In[ ]:


df = pd.read_csv('../input/facial-expression/fer2013.csv')


# In[ ]:


df.head()


# In[ ]:


len(df.iloc[0]['pixels'].split())
# 48 * 48


# In[ ]:


label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


img = df.iloc[0]['pixels'].split()


# In[ ]:


img = [int(i) for i in img]


# In[ ]:


type(img[0])


# In[ ]:


len(img)


# In[ ]:


img = np.array(img)


# In[ ]:


img = img.reshape(48,48)


# In[ ]:


img.shape


# In[ ]:


plt.imshow(img, cmap='gray')
plt.xlabel(df.iloc[0]['emotion'])


# In[ ]:


X = []
y = []


# In[ ]:


def getData(path):
    anger = 0
    fear = 0
    sad = 0
    happy = 0
    surprise = 0
    neutral = 0
    df = pd.read_csv(path)
    
    X = []
    y = []    
    
    for i in range(len(df)):
        if df.iloc[i]['emotion'] != 1:
            if df.iloc[i]['emotion'] == 0:
                if anger <= 4000:            
                    y.append(df.iloc[i]['emotion'])
                    im = df.iloc[i]['pixels']
                    im = [int(x) for x in im.split()]
                    X.append(im)
                    anger += 1
                else:
                    pass
                
            if df.iloc[i]['emotion'] == 2:
                if fear <= 4000:            
                    y.append(df.iloc[i]['emotion'])
                    im = df.iloc[i]['pixels']
                    im = [int(x) for x in im.split()]
                    X.append(im)
                    fear += 1
                else:
                    pass
                
            if df.iloc[i]['emotion'] == 3:
                if happy <= 4000:            
                    y.append(df.iloc[i]['emotion'])
                    im = df.iloc[i]['pixels']
                    im = [int(x) for x in im.split()]
                    X.append(im)
                    happy += 1
                else:
                    pass
                
            if df.iloc[i]['emotion'] == 4:
                if sad <= 4000:            
                    y.append(df.iloc[i]['emotion'])
                    im = df.iloc[i]['pixels']
                    im = [int(x) for x in im.split()]
                    X.append(im)
                    sad += 1
                else:
                    pass
                
            if df.iloc[i]['emotion'] == 5:
                if surprise <= 4000:            
                    y.append(df.iloc[i]['emotion'])
                    im = df.iloc[i]['pixels']
                    im = [int(x) for x in im.split()]
                    X.append(im)
                    surprise += 1
                else:
                    pass
                
            if df.iloc[i]['emotion'] == 6:
                if neutral <= 4000:            
                    y.append(df.iloc[i]['emotion'])
                    im = df.iloc[i]['pixels']
                    im = [int(x) for x in im.split()]
                    X.append(im)
                    neutral += 1
                else:
                    pass

            
            
    return X, y  
    


# In[ ]:


X, y = getData('../input/facial-expression/fer2013.csv')


# In[ ]:


np.unique(y, return_counts=True)


# In[ ]:


X = np.array(X)/255.0
y = np.array(y)


# In[ ]:


X.shape, y.shape


# In[ ]:


y_o = []
for i in y:
    if i != 6:
        y_o.append(i)
        
    else:
        y_o.append(1)


# In[ ]:


np.unique(y_o, return_counts=True)


# In[ ]:


for i in range(5):
    r = np.random.randint((1), 24000, 1)[0]
    plt.figure()
    plt.imshow(X[r].reshape(48,48), cmap='gray')
    plt.xlabel(label_map[y_o[r]])


# In[ ]:


X = X.reshape(len(X), 48, 48, 1)


# In[ ]:


# no_of_images, height, width, coloar_map


# In[ ]:


X.shape


# In[ ]:


from keras.utils import to_categorical
y_new = to_categorical(y_o, num_classes=6)


# In[ ]:


len(y_o), y_new.shape


# In[ ]:


y_o[150], y_new[150]


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization


# In[ ]:


model = Sequential()


input_shape = (48,48,1)


model.add(Conv2D(64, (5, 5), input_shape=input_shape,activation='relu', padding='same'))
model.add(Conv2D(64, (5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
model.add(Conv2D(128, (5, 5),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

## (15, 15) --->  30
model.add(Flatten())
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')


# In[ ]:


model.fit(X, y_new, epochs=22, batch_size=64, shuffle=True, validation_split=0.2)


# In[ ]:


model.save('model.h5')


# In[ ]:


import cv2


# In[ ]:


test_img = cv2.imread('../input/happy-img-test/pexels-andrea-piacquadio-941693.jpg', 0)


# In[ ]:


test_img.shape


# In[ ]:


test_img = cv2.resize(test_img, (48,48))
test_img.shape


# In[ ]:


test_img = test_img.reshape(1,48,48,1)


# In[ ]:


model.predict(test_img)


# In[ ]:


# label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']

