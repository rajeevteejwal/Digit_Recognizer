
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


# In[ ]:


output_dir = output_dir.replace('\\train','')
os.chdir(output_dir)


# In[ ]:


#Save training images
output_dir = os.getcwd()
output_dir = output_dir+'\\train'
os.chdir(output_dir)
for i in range(len(df_train)):
    fname = str(i+1)+'.jpg'
    plt.imsave(fname,np.reshape(df_train.iloc[i,1:],(28,28)))

output_dir = output_dir.replace('\\train','')
os.chdir(output_dir)


# In[ ]:


#Save test images
output_dir = os.getcwd()
output_dir = output_dir+'\\test'
os.chdir(output_dir)
for i in range(len(df_test)):
    fname = str(i+1)+'.jpg'
    plt.imsave(fname,np.reshape(df_test.iloc[i,:],(28,28)))

output_dir = output_dir.replace('\\test','')
os.chdir(output_dir)


# In[3]:


#create X column in traing df
df_train['X']=''
for i in range(len(df_train)):
    df_train.loc[i,'X'] = str(i+1)+'.jpg'

df_train_keras = df_train[['X','label']]


# In[4]:


#create X column in test DF
df_test['X']=''
for i in range(len(df_test)):
    df_test.loc[i,'X'] = str(i+1)+'.jpg'

df_test_keras = df_test[['X']]


# In[6]:


#plt.imshow(np.reshape(df_train.iloc[1,1:],(28,28)))


# In[7]:


# Importing all necessary libraries 
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 


# In[8]:


img_width = 28
img_height = 28
input_shape = (img_width, img_height, 3)

train_data_dir = './train/'
validation_data_dir = './test/'
nb_train_samples =42000
nb_validation_samples = 10500
epochs = 10
batch_size = 16


# In[9]:


#Define CNN
model = Sequential() 
model.add(Conv2D(32, (2, 2), input_shape=input_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
  
model.add(Conv2D(32, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
  
model.add(Conv2D(64, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
  
model.add(Flatten()) 
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(10)) 
model.add(Activation('softmax')) 


# In[10]:


#Compile function
model.compile(loss='binary_crossentropy', 
              optimizer='rmsprop', 
              metrics=['accuracy']) 


# In[11]:


#data generator
train_datagen = ImageDataGenerator( 
    rescale=1. / 255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True,validation_split=0.25) 
  
test_datagen = ImageDataGenerator(rescale=1. / 255)


# In[ ]:


train_generator = train_datagen.flow_from_directory( 
    train_data_dir, 
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[ ]:


validation_generator = test_datagen.flow_from_directory( 
    validation_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode='binary') 


# In[12]:


#from source
train_generator=train_datagen.flow_from_dataframe(
dataframe=df_train_keras,
directory="./train/",
x_col="X",
y_col="label",
subset="training",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(28,28))
valid_generator=train_datagen.flow_from_dataframe(
dataframe=df_train_keras,
directory="./train/",
x_col="X",
y_col="label",
subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(28,28))


# In[13]:


#test generator
test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
dataframe=df_test_keras,
directory="./test/",
x_col="X",
y_col=None,
batch_size=32,
seed=42,
shuffle=False,
class_mode=None,
target_size=(28,28))


# In[14]:


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)


# In[ ]:


#model fitting
model.fit_generator( 
    train_generator, 
    steps_per_epoch=nb_train_samples // batch_size, 
    epochs=epochs, 
    validation_data=validation_generator, 
    validation_steps=nb_validation_samples // batch_size) 


# In[15]:


#save the model
model.save_weights('model_saved.h5') 


# In[17]:


#Evaluate the model
model.evaluate_generator(generator=valid_generator,
steps=STEP_SIZE_TEST)


# In[18]:


#predict the output
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)


# In[19]:


predicted_class_indices=np.argmax(pred,axis=1)


# In[20]:


#predict labels
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


# In[21]:


#save result to csv file
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)

