"""
Created on Feb 18 2026


This file is intended to train a model using the whole dataset from all environments and test it on each environment separately

This code is based on the code from the following paper:
Klemen Bregar. (2023). UWB Positioning and Tracking Data Set (Version 2.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.8280736

"""


import json
import ast

import numpy as np
import time
import math
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU, LeakyReLU, Conv1D, MaxPooling1D, Activation, Flatten , SeparableConv1D , AveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import data

estimated_range = []
true_ranging_error = []
cir = []
train_data = []
train_labels = []

# import data
start_time = time.time()

# calculate input data size
pooling_size = 2
input_data_size = int(math.ceil(150 / (math.pow(pooling_size, 2))) * (math.pow(pooling_size, 2)))





L1_patch_w = 2
L1_depth = 10
L2_patch_w = 4
L2_depth = 10
L3_patch_w = 4
L3_depth = 10
L4_patch_w = 4
L4_depth = 20
L5_patch_w = 4
L5_depth = 20



def generate_sep_model16(batch_normalization=True, dropout_regularization=True):

	model = Sequential()
	# L1
	model.add(SeparableConv1D(input_shape=(input_data_size,1), 
						   filters=L1_depth, 
						   kernel_size=L1_patch_w, 
						   strides=1,
						    padding='valid',
				            use_bias=True,
							depthwise_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
							pointwise_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
							bias_initializer=keras.initializers.Constant(value=0.1)))
	
	if True == batch_normalization:
		model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	# L2
	model.add(SeparableConv1D(filters=L2_depth, kernel_size=L2_patch_w, strides=2, padding='same', use_bias=True,
							depthwise_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
							pointwise_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
							bias_initializer=keras.initializers.Constant(value=0.1)))
	if True == batch_normalization:
		model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	# L3
	model.add(SeparableConv1D(filters=L3_depth, kernel_size=L3_patch_w, strides=2, padding='same', use_bias=True,
											depthwise_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
							pointwise_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
							bias_initializer=keras.initializers.Constant(value=0.1)))
	if True == batch_normalization:
		model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	# pool			 
	model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
	# L4
	model.add(SeparableConv1D(filters=L4_depth, kernel_size=L4_patch_w, strides=1, padding='valid', use_bias=True,
						depthwise_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
						pointwise_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
						bias_initializer=keras.initializers.Constant(value=0.1)))
	if True == batch_normalization:
		model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	# L5
	model.add(SeparableConv1D(filters=L5_depth, kernel_size=L5_patch_w, strides=2, padding='same', use_bias=True,
				 depthwise_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
				 pointwise_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
				 bias_initializer=keras.initializers.Constant(value=0.1)))
	if True == batch_normalization:
		model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.3))
	# pool
	model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
	# flatten
	model.add(Flatten())
	# dense
	model.add(Dense(units=16, activation='relu', use_bias=True,
				kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
				bias_initializer=keras.initializers.Constant(value=0.1)))
	if  True == dropout_regularization:
		model.add(Dropout(rate=0.5))
	model.add(Dense(units=1, activation='linear', kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
				bias_initializer=keras.initializers.Constant(value=0.1)))
	
	return model


def save_model(model, model_name):

    """
    Saves trained model to disk as JSON for structure and HDF5 for weights
    :param model: Keras model
    :param model_name: model name
    :return:
    """
    model_json = model.to_json()
    with open('./models/lightweight_models/' + str(model_name) + '.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('./models/lightweight_models/' + str(model_name) + '.weights.h5')
    model.save('./models/lightweight_models/' + str(model_name) + '.keras')



anchors = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
environments = ['environment0', 'environment1', 'environment2', 'environment3']
channels = ['ch5']

# prepare train data
ds = data.DataSet()

names_mask = []
# prepare train data
train_data = []
train_labels = []
for k in range(len(environments)):
		names_mask.append(environments[k])
		
names_mask = np.asarray(names_mask)
channel = 'ch5'
for j in range(len(environments)):
	# load data first
	true_ranges, meas_ranges, ranging_error , cir = ds.load_data_set(channel, names_mask[j])
	labels = ranging_error
	train_data.append(cir)
	train_labels.append(labels)
train_data = np.concatenate(train_data, axis=0)	
train_labels = np.concatenate(train_labels, axis=0)	


# 1) split out 10% test
train_data, test_data, train_labels, test_labels = train_test_split(
    train_data,
    train_labels,
    test_size=0.1,
    shuffle=True,
    random_state=42
)

# 2) split remaining 90% into 80% train, 10% validation
train_data, val_data, train_labels, val_labels = train_test_split(
    train_data,
    train_labels,
    test_size=0.111111,   # 10% of total
    shuffle=True,
    random_state=42
)


model = generate_sep_model16(batch_normalization=True, dropout_regularization=True)
adam = Adam(learning_rate=1e-4)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error','mae'])
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=200, verbose=1, mode='min', baseline=None, restore_best_weights=True)
history = model.fit(x=train_data, y=train_labels, batch_size=8192, epochs=500, verbose=1, validation_data=(val_data, val_labels), shuffle=False, callbacks=[earlystopping])

evaluation16 = model.evaluate(x=test_data, y=test_labels, verbose=0)
print(evaluation16)
print("mean_squared_error16: %0.2f" % evaluation16[0])
fn = 'regressor_SeparableCNN_D16_whole_environments'
# fn = 'original'
save_model(model, fn)
print('DONE')


