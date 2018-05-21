# flight delay

# ece271b project


import numpy as np
# import matplotlib.pyplot as plt
import pandas
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU, SimpleRNN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime

task = 'fd_01'

dataset = np.load('flight_delay_2016.npy')
dataset = dataset.reshape(-1,1)
print "dataset shape:", dataset.shape

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
# scaler = StandardScaler()
dataset = scaler.fit_transform(dataset)


# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print("num train:", len(train), "num test:", len(test))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print "x shape:", trainX.shape
print "y.shape:", trainY.shape

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
num_units = 2
layer_name = 'rnn'
model = Sequential()
# model.add(LSTM(num_units, return_sequences=True, input_shape=(1, look_back)))
# model.add(LSTM(num_units))
# model.add(LSTM(num_units, input_shape=(1, look_back)))
# model.add(GRU(num_units, input_shape=(1, look_back)))
model.add(SimpleRNN(num_units, input_shape=(1, look_back)))

model.add(Dense(1))
adam = keras.optimizers.Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=adam)

loss_hist = np.zeros((30,2))

t1 = datetime.now()

for i in range(30):
    model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY_temp = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY_temp = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY_temp[0], trainPredict[:,0]))
    loss_hist[i, 0] = trainScore
    # print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY_temp[0], testPredict[:,0]))
    loss_hist[i, 1] = testScore
    # print('Test Score: %.2f RMSE' % (testScore))

t2 = datetime.now()
t_diff = t2 - t1
seconds = t_diff.total_seconds()
print('time: {} seconds'.format(seconds))

np.save('{}_{}_loss_hist.npy'.format(task, layer_name), loss_hist)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
np.save('{}_{}_trainPredict.npy'.format(task, layer_name), trainPredict)

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
np.save('{}_{}_testPredict.npy'.format(task, layer_name), testPredict)

# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()

