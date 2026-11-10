""""

This file is intended to train the model using TL

"""


import ast
import json
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
import time

from tensorflow import keras

from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


estimated_range = []
true_ranging_error = []
cir = []
train_data = []
train_labels = []

# import data
start_time = time.time()


# """	


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


# Reading the Dataset
with open(f"training_data_library.json", "r") as d:
	data = json.load(d)
estimated_range , true_ranging_error , cir  =  combine_all_data(data)
cir_model = np.asarray(cir)
true_ranging_error = np.asarray(true_ranging_error)
estimated_range = np.asarray(estimated_range)


#load the base model
model_name = 'whole_environments'
json_file = open('./models/' + str(model_name) + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('./models/' + str(model_name) + '.weights.h5')
adam = Adam(learning_rate=1e-4)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error'])
    




# Transfer learning for dense layers
# for layer in model.layers:
#     layer.trainable = False

# dense_layers = [l for l in model.layers if isinstance(l, keras.layers.Dense)]

# for layer in dense_layers[-1:]:
#     layer.trainable = True

# for i, layer in enumerate(model.layers):
#     print(i, layer.name, layer.trainable)
	

# Full fine-tuning
for layer in model.layers:
    layer.trainable = True

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)

adam = Adam(learning_rate=1e-5)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error','mae'])


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

history = model.fit(x=train_data, y=train_labels, batch_size=8192, epochs=500, verbose=1, validation_data=(val_data, val_labels), shuffle=True)

evaluation = model.evaluate(x=test_data, y=test_labels, verbose=0)
print(evaluation)
print("mean_squared_error: %0.2f" % evaluation[0])
fn = 'tl_library_wholemodel_500_minus5'
save_model(model, fn)
print("Validation MSE:", history.history['val_mean_squared_error'][-1])
print("Validation MAE:", history.history['val_mae'][-1])
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

