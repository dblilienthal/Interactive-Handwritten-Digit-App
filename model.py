from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os
from pathlib import Path

def create_model():
    # Create the architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Get our model ready for training
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    return model 
     
def get_min_val_loss_epoch(history):
    return "0"+str(np.argmin(history.history['val_loss'])+1)

if __name__ == '__main__':

    ### Load the dataset
    objects = tf.keras.datasets.mnist
    (training_images, training_labels), (testing_images, testing_labels) = objects.load_data()


    # Scale the data
    training_images = training_images / 225.0
    testing_images = testing_images / 225.0

    ## Create Validation Data from the Test Data
    X_train, X_val, y_train, y_val = train_test_split(training_images, training_labels, test_size=0.1, random_state=42)

    print("There are {} Training examples\n{} Testing examples \n{} Validation Examples".format(X_train.shape[0], testing_images.shape[0], X_val.shape[0]))
   
    ### Create an output directory
    cwd = Path(__file__).resolve().parent ### Current Directory
    output_dir = str(cwd)+'/model_weights/'
    if not os.path.exists(output_dir): ### If the file directory doesn't already exists,
        os.makedirs(output_dir) ### Make it please

    ### Create a checkpoint to store the weights
    modelcheckpoint = ModelCheckpoint(filepath=output_dir+'/weights.{epoch:02d}.hdf5', save_weights_only=True, save_best_only=True, verbose=1)
   
    ### Create the model
    model = create_model()

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[modelcheckpoint], verbose=1)

    ### Load the model with the lowest validation loss
    epoch_num = get_min_val_loss_epoch(history)
    model.load_weights(output_dir+"/weights."+epoch_num+".hdf5") 

    # Evaluate the model
    model.evaluate(testing_images, testing_labels, verbose=2)

