from PlotLearning import PlotLearning
from InterestFrames import *
from Model_build import *
from DataExploratoryAnalysis import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
if __name__ == '__main__':
    train_file_paths, test_file_paths = get_data()
    x_train, x_val, y_train, y_val, test_labels = train_test_dataset(train_file_paths, test_file_paths)
    train_ds = preprocess_dataset(x_train, y_train)
    val_ds = preprocess_dataset(x_val, y_val)
    train_ds = train_ds.batch(2).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(2).prefetch(tf.data.AUTOTUNE)
    input_shape = (25, 80, 100, 3)  # Adjust this based on the actual shape of your input data and interest frame module
    model = build_3d_cnn_with_attention(input_shape)
    print(model.summary())
    plot_learning = PlotLearning()
    checkpoint_path = '/content/drive/My Drive/all files/best_modelMoreChanges10.h5'  # Update the path as needed
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    callbacks = [model_checkpoint, early_stopping, plot_learning]
    # Training
    history = model.fit(train_ds,
                        epochs=20,
                        validation_data=val_ds,
                        callbacks=callbacks)
    model.save('/content/drive/My Drive/all files/trained_model4.4.h5')  # Update the path as needed