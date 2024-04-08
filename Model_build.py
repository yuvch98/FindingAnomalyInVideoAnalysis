# importing needed libraries
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, AveragePooling3D ,BatchNormalization, Dense, Flatten, concatenate, Multiply, Reshape, Dropout, GlobalMaxPooling3D, GlobalAveragePooling3D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import TensorBoard
import time
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
random.seed(42)

def spatial_attention_module(input_tensor):
    # Global Average Pooling to reduce spatial dimensions to 1
    gap = GlobalAveragePooling3D()(input_tensor)

    # Global Max Pooling to reduce spatial dimensions to 1
    gmp = GlobalMaxPooling3D()(input_tensor)

    # Reshape to make them compatible for concatenation
    reshaped_gap = Reshape((1, 1, 1, gap.shape[-1]))(gap)
    reshaped_gmp = Reshape((1, 1, 1, gmp.shape[-1]))(gmp)

    # Concatenate the pooling results
    concatenated = concatenate([reshaped_gap, reshaped_gmp], axis=-1)

    # Adjust the 3D convolution to have a kernel size of (7, 7, 7) and match the proposed method
    conv3d = Conv3D(filters=input_tensor.shape[-1], kernel_size=(7, 7, 7), padding='same', activation='relu')(concatenated)

    # Multiply the convolution output by the input tensor
    output_tensor = Multiply()([input_tensor, conv3d])

    return output_tensor

def build_3d_cnn_with_attention(input_shape):
    # Initialize the Adam optimizer with a custom learning rate
    AdamOptimizer = Adam(learning_rate=0.01)

    # Input layer
    inputs = Input(shape=input_shape)

    # Spatial attention module
    attention_output = spatial_attention_module(inputs)

    # First 3D convolutional block
    conv1 = Conv3D(filters=5, kernel_size=(5, 5, 5), activation='relu', padding='same')(attention_output)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1), strides=1, padding='same')(conv1)
    norm1 = BatchNormalization()(pool1)

    # Second 3D convolutional block
    conv2 = Conv3D(10, kernel_size=(5, 5, 5), activation='relu', padding='same')(norm1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1), strides=1, padding='same')(conv2)
    norm2 = BatchNormalization()(pool2)

    # Flatten the output of the convolutional blocks
    flattened = Flatten()(norm2)

    # Fully connected layers with dropout for regularization
    fc1 = Dense(256, activation='relu')(flattened)
    fc2 = Dense(128, activation='relu')(fc1)
    dropout = Dropout(0.2)(fc2)

    # Output classification layer
    outputs = Dense(1, activation='sigmoid')(dropout)

    # Construct and compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=AdamOptimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Save the trained model
