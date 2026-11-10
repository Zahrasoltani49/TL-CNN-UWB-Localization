
"""

Inference phase

"""
from sklearn.metrics import mean_squared_error



from tensorflow import keras
import time
from datetime import datetime
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import os
from scipy import optimize
import json
from sklearn.metrics import accuracy_score
# import numpy as np
import ast
# import xlwings as xw


try:
    import thread
except ImportError:
    import _thread as thread
CIRLEN = 152
OFFSET = 5
# from datetime import datetime

number_anchors = 6 # for 4 anchors

number_measurements = 50
x_tag_true = []
y_tag_true = []



x_tag, y_tag = [], []                # to store tag positions over time
    # to store true tag positions over time
ml_x_tag, ml_y_tag = [], []          # to store machine learning based tag positions over time
corrected_ranges = []                # to store corrected ranges after regression inference
nlos_anchors = []                    # to store positions of anchors classified as NLOS
collect_ranging_errors = []          # to collect ranging errors predicted by regression model for analysis

test_data = []
regression_test_labels = []
new_dataset = []
estimated_range = []

global cir_list

cir_model = []

true_rng = []
rng = []
rng_error = []
cir = []
nlos = []
wls_y_tag = []
wls_x_tag = []
true_ranging_error = []

anchors_position = [[3.68 , 0.50  ],
                    [-0.90 , 3.70  ],
                    [1.63 , 8.50  ],
                    [5.31 , 7.68  ],
                    [5.31 , 2.16  ],
                    [12.69 , 6.27 ]
                ] 




global item


def localization(distances, anchors):
    """
    Compute tag position using linear least squares (Method 1)
    distances: list of distances to anchors
    anchors: np.array of anchor positions [[x1,y1], [x2,y2], [x3,y3]]
    Returns: np.array([tag_x, tag_y])
    """
    
    A_2D = []
    b_2D = []
    tag_pos_2D = np.zeros(2)
    
    
# Use the first anchor as reference
    x1, y1 = anchors[0]
    d1 = distances[0]
    

    
    
    for i in range(1, number_anchors):
        xi, yi = anchors[i]
        di = distances[i]

        
        A_2D.append([2*(xi - x1), 2*(yi - y1)])
        b_2D.append(d1**2 - di**2 + xi**2 - x1**2 + yi**2 - y1**2)


    
    A_2D = np.array(A_2D)
    b_2D = np.array(b_2D)

    # Solve using pseudo-inverse

    tag_pos_2D = np.linalg.pinv(A_2D) @ b_2D

    print('2D Tag position',tag_pos_2D)
    return   tag_pos_2D


def WLS_localization(distances, anchors, ranging_errors):
    """
    Compute tag position using linear least squares (Method 1)
    distances: list of distances to anchors
    anchors: np.array of anchor positions [[x1,y1], [x2,y2], [x3,y3]]
    Returns: np.array([tag_x, tag_y])
    """
    
    A_2D = []
    b_2D = []
    weights = []
    ranging_errors =  ranging_errors.flatten()
    
    
    tag_pos_2D_WLS = np.zeros(2)
    
    
# Use the first anchor as reference
    x1, y1 = anchors[0]
    d1 = distances[0]

    

    
    for i in range(1, number_anchors):
        xi, yi = anchors[i]
        di = distances[i]
        weights.append(1 / (ranging_errors[i]**2 * distances[i]))  # Example: weight inversely proportional to error variance and distance

        
        A_2D.append([2*(xi - x1), 2*(yi - y1)])
        b_2D.append(d1**2 - di**2 + xi**2 - x1**2 + yi**2 - y1**2)


    W = np.diag(weights) 
    A_2D = np.array(A_2D)
    b_2D = np.array(b_2D)

    AT = A_2D.T
    tag_pos_2D_WLS = np.linalg.solve(AT @ W @ A_2D, AT @ W @ b_2D)


    # # Solve using pseudo-inverse

    # tag_pos_2D = np.linalg.pinv(A_2D) @ b_2D
    
    print('2D Tag position (WLS)',tag_pos_2D_WLS)
    return   tag_pos_2D_WLS



## Load the trained model for classification

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






########################################################################################################################
#load the base model
model_name = 'proposedmodel_fromscratch_librarydataset'
json_file = open('./models/' + str(model_name) + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('./models/' + str(model_name) + '.weights.h5')
adam = Adam(learning_rate=1e-5)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error'])
    


# read the dataset for testing
cnn_regression = model
# cnn_regression = data.load_transfer_learning_model('whole_500_minus5')

#! number of points: 7 Trajectory1, 6 Trajectory2, 6 Trajectory3
number_points = 7
for i in range (1, number_points + 1):
    # with open(f'./my_test_dataset_library/clearedExcel_library_dataset/cleared_point{i}.json', 'r') as f:

    with open(f'./combi_data_preprocessing/point{i}.json', 'r') as f:
        data = json.load(f)
    
    estimated_range , true_ranging_error , cir  =  combine_all_data(data)
    x_tag_true.append(data[1]['x_tag_true'])
    y_tag_true.append(data[1]['y_tag_true'])

  

cir_model = np.asarray(cir)
true_ranging_error = np.asarray(true_ranging_error)
estimated_range = np.asarray(estimated_range)
true_x = np.asarray(x_tag_true)
true_y = np.asarray(y_tag_true)
true_x = np.repeat(x_tag_true, number_measurements)
true_y = np.repeat(y_tag_true, number_measurements)


estimated_ranging_error = cnn_regression.predict(cir_model , verbose = 0 ).flatten()
mse = mean_squared_error(true_ranging_error , estimated_ranging_error)
print("mse is " , mse)

###########################################################################################

corrected_ranges = estimated_range - estimated_ranging_error
reshaped_estimated_range = estimated_range.reshape(number_points,number_measurements,number_anchors)
reshaped_corrected_ranges = corrected_ranges.reshape(number_points,number_measurements,number_anchors)
reshaped_ranging_error = estimated_ranging_error.reshape(number_points,number_measurements,number_anchors)

###########################################################################################




positions_original = []
positions_ls_corrected = []
positions_wls_corrected = []



for point in range(number_points):
    for meas in range(number_measurements):
        # Original ranges
        ranges_orig = reshaped_estimated_range[point, meas , :]
        x_orig, y_orig = localization(ranges_orig, anchors_position)
        positions_original.append([point, meas, x_orig, y_orig])

        # LS corrected
        ranges_corr = reshaped_corrected_ranges[point, meas , :]
        x_corr, y_corr = localization(ranges_corr, anchors_position)
        positions_ls_corrected.append([point, meas, x_corr, y_corr])

        # WLS corrected
        ranges_corr = reshaped_corrected_ranges[point, meas , :]
        x_corr, y_corr = WLS_localization(ranges_corr, anchors_position, reshaped_ranging_error[point, meas , :])
        positions_wls_corrected.append([point, meas, x_corr, y_corr])

        # ranges_corr = reshaped_corrected_ranges[point, meas , :]
        # x_corr, y_corr = WLS_localization(ranges_orig, anchors_position, reshaped_ranging_error[point, :, meas])
        # positions_wls_corrected.append([point, meas, x_corr, y_corr])

print("localization done")

# Convert lists to arrays for easy slicing
positions_original = np.array(positions_original)
positions_ls_corrected = np.array(positions_ls_corrected)
positions_wls_corrected = np.array(positions_wls_corrected)

# Build DataFrame
df = pd.DataFrame({
    "timestamp": datetime.now().isoformat(),
    "true_x": true_x,
    "true_y": true_y,
    "estimated_x_ls_original": positions_original[:,2],
    "estimated_y_ls_original": positions_original[:,3],
    "estimated_x_ls_corrected": positions_ls_corrected[:,2],
    "estimated_y_ls_corrected": positions_ls_corrected[:,3],
    "estimated_x_wls_corrected": positions_wls_corrected[:,2],
    "estimated_y_wls_corrected": positions_wls_corrected[:,3]
})

# Save
results_path = 'write your desired path.csv'
df.to_csv(results_path, index=False)
print(f"CSV file saved at {results_path}")
