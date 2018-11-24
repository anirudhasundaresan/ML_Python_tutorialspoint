import numpy as np
from sklearn import preprocessing

input_data = np.array([[3, -1.5, 3, -6.4], [0, 3, -1.3, 4.1], [1, 2.3, -2.9, -4.3]]) # 2d-array with 3x4 shape

# input_data.shape --> (3,4)

# mean removal | removes bias
data_standardized = preprocessing.scale(input_data)
print("Mean of std. data: ", data_standardized.mean(axis=0)) # 0 --> vertically down arrow; for all columns; 1 --> horizontally to-the-right arrow
print("STD of std. data: ", data_standardized.std(axis=0))

# scaling
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1)) # defining the scaler function
data_scaled = data_scaler.fit_transform(input_data) # fit and then transform
print("Min max scaled data: ", data_scaled) # all values between 0 and 1.

# normalization --> makes sure each column sums to 1. Converts data to a common scale.
data_normalized = preprocessing.normalize(input_data, norm='l1') # by default, axis = 1
print("L1 Normalized data: ", data_normalized)

# note: normalization is different from standardization

# binarization --> convert numerical feature to boolean vector
data_binarizer = preprocessing.Binarizer(threshold=1.4)
data_binarized = data_binarizer.transform(input_data)
print("Binarized data: ", data_binarized)

# 1-hot encoding
# https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
# learn this ..?

# label encoding --> if labels are not in words/ readable form: label encoding is changing the word labels into numbers so that algos can understand how to work with them.
label_encoder = preprocessing.LabelEncoder()
input_classes = ['suzuki', 'ford', 'suzuki', 'toyota', 'ford', 'bmw']
label_encoder.fit(input_classes)
print("Class mapping: ")
for ind, item in enumerate(label_encoder.classes_):
    print(item, '-->', ind)

# instead of using fit, you can use transform this way:
labels = ['toyota', 'ford', 'suzuki']
encoded_labels = label_encoder.transform(labels)
print("Labels: ", labels)
print("Encoded labels: ", encoded_labels)

# check by transforming numbers back to words
encoded_labels = [3,2,0,2,1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print("Encoded labels: ", encoded_labels)
print("Decoded labels: ", decoded_labels)
