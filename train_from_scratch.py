""" 
this is a code to train teh model with library dataset withot using transfer learning

it trains the modle with collected dataset from Tyndall library- it represents the "model from scratch"

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
fc_size = 16

def save_model(model, model_name):
	"""
	Saves trained model to disk as JSON for structure and HDF5 for weights
	:param model: Keras model
	:param model_name: model name
	:return:
	"""
	model_json = model.to_json()
	with open('./models/' + str(model_name) + '.json', 'w') as json_file:
		json_file.write(model_json)
	model.save_weights('./models/' + str(model_name) + '.weights.h5')
	model.save('./models/' + str(model_name) + '.keras')




def generate_model(batch_normalization=True, dropout_regularization=True):
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
	model.add(Dense(units=fc_size, activation='relu', use_bias=True,
				kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
				bias_initializer=keras.initializers.Constant(value=0.1)))
	if  True == dropout_regularization:
		model.add(Dropout(rate=0.5))
	model.add(Dense(units=1, activation='linear', kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
				bias_initializer=keras.initializers.Constant(value=0.1)))
	
	return model



def combine_all_data(data):
    for c in range(len(data)):
     item = data[c]
     item['cir'] = item['cir'].replace('[', "")
     item['cir'] = item['cir'].replace(']', "")
     cir_str = item['cir']  # your string
     cir_array = np.array(ast.literal_eval(cir_str), dtype=np.float32)
     cir_dim = np.expand_dims(cir_array, axis=1)
     cir.append(cir_dim)

     true_ranging_error.append(item['range_error'])
     estimated_range.append(item['measured_range'])


    return estimated_range , true_ranging_error , cir


with open(f"./dataset/training_dataset/training_data_library.json", "r") as d:
	data = json.load(d)
estimated_range , true_ranging_error , cir  =  combine_all_data(data)
cir_model = np.asarray(cir)
true_ranging_error = np.asarray(true_ranging_error)
estimated_range = np.asarray(estimated_range)


model = generate_model(batch_normalization=True, dropout_regularization=True)
adam = Adam(learning_rate=1e-4)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error','mae'])
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=200, verbose=1, mode='min', baseline=None, restore_best_weights=True)



# 1) Use the first 30,000 samples for training and validation
train_val_data = cir_model[:30000]
train_val_labels = true_ranging_error[:30000]

# 2) Use the last 10,000 samples for testing
test_data = cir_model[30000:]
test_labels = true_ranging_error[30000:]

# 3) Split the first 30,000 into 80% training and 20% validation
train_data, val_data, train_labels, val_labels = train_test_split(
    train_val_data,
    train_val_labels,
    test_size=0.2,
    shuffle=True,
    random_state=42
)

history = model.fit(x=train_data, y=train_labels, batch_size=8192, epochs=500, verbose=1, validation_data=(val_data, val_labels), shuffle=True, callbacks=[earlystopping])
	# history = model.fit(x=train_data, y=train_labels, batch_size=8192, epochs=100, verbose=1, validation_split=0.1, shuffle=True)



evaluation = model.evaluate(x=test_data, y=test_labels, verbose=0)
print(evaluation)
print("mean_squared_error: %0.2f" % evaluation[0])

fn = 'proposedmodel_fromscratch_librarydataset'
save_model(model, fn)

plt.figure()
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])

plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training and Validation Mean Squared Error')
plt.legend(['Train MSE', 'Validation MSE'])

plt.show()
print("--- %s seconds ---" % (time.time() - start_time))
			
	# clear model
keras.backend.clear_session()
del model

