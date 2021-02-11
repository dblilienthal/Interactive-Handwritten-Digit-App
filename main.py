import PIL
from PIL import ImageDraw
from tkinter import *
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Allocate GPU for predictions
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Parameters for Canvas
width = 500
height = 500
center = height//2
white = (255, 255, 255)
black = (0, 0, 0)


def save():
    """This method just saves the file to the local directory"""
    filename = 'temp.png'
    image1.save(filename)


def paint(event):
    """This method allows a user to draw on the canvas"""
    color = 'white'
    x1, y1 = (event.x-1), (event.y-1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill=color, width=30)
    draw.line([x1, y1, x2, y2], fill=color, width=30)


def create_model():
    """Run this method to create, train, and save the Keras model"""
    objects = tf.keras.datasets.mnist
    # Split the data set into training and testing data
    (training_images, training_labels), (testing_images, testing_labels) = objects.load_data()

    # Scale the data
    training_images = training_images / 225.0
    testing_images = testing_images / 225.0

    # Create our model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Get our model ready for training
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(training_images, training_labels, epochs=3)
    # Save the model
    model.save('my_model.h5')


def get_number_from_picture():
    """Get the image ready for inference"""
    from PIL import Image
    im = Image.open("temp.png")
    newsize = (28, 28)
    im = im.resize(newsize)

    # Convert the image into a 2d array and reshape it to the desired input
    result_arr = []
    for i in range(28):
        result_arr.append(np.asarray(im)[i][:, 0].tolist())
    pic_array = np.array(result_arr) / 255.0
    pic_array = pic_array.reshape((1, 28, 28))

    """Load the Model and weights"""
    loaded_model = load_model('my_model.h5')

    prediction = loaded_model.predict(x=pic_array)
    print("The predicted value is:", np.argmax(prediction))

    del loaded_model


"""Code for canvas drawing"""

root = Tk()
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()

image1 = PIL.Image.new("RGB", (width, height), black)
draw = ImageDraw.Draw(image1)

cv.pack(expand=YES, fill=BOTH)
cv.bind('<B1-Motion>', paint)

button = Button(text="Save", command=save)
button.pack()

root.mainloop() # Run the canvas

# Get the predictions
get_number_from_picture()
