#Part-1 Data Preprocessing

#Importing the Dataset
from keras.datasets.mnist import load_data
(X_train,y_train),(X_test,y_test) = load_data()

#flatten the  images
X_train = X_train.reshape((X_train.shape[0],784))
X_test = X_test.reshape((X_test.shape[0],784))
y_train = y_train.reshape((y_train.shape[0],1))
y_test = y_test.reshape((y_test.shape[0],1))

#Encoding of categorical data
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse = False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

#splitting the dataset into train and validation
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size = 0.1)

#Feature Scaling
X_train = X_train/255
X_test  = X_test/255
X_val  = X_val/255

#Part-2 Building Model Using Keras

#importing the libraries
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,BatchNormalization

#Build ANN model
model = Sequential()
model.add(Dense(units = 256,kernel_initializer = 'he_normal',input_dim = 784))
model.add(Activation('relu'))
model.add(Dropout(0.4))


model.add(Dense(units = 256,kernel_initializer = 'he_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(units = 10,activation='softmax',kernel_initializer = 'glorot_normal'))

#Compiling model
model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

#train our model
model.fit(X_train, y_train, batch_size = 512, epochs = 50)

train = model.evaluate(X_train,y_train)
print("Accuracies on train set" + str(train))
val = model.evaluate(X_val,y_val)
print("Accuracies on validation set" + str(val))
test = model.evaluate(X_test,y_test)
print("Accuracies on test set" + str(test))

model.save('digit_Classifier')
