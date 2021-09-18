import PIL
from PIL import ImageDraw
from tkinter import *
import numpy as np
import tensorflow as tf
from pathlib import Path
import os
import model


# Parameters for Canvas
width = 500
height = 500
center = height//2
white = (255, 255, 255)
black = (0, 0, 0)
cwd = Path(__file__).resolve().parent ### Current Directory


def save():
    """This method just saves the file to the local directory"""
    filename = str(cwd)+'/temp.png'
    image1.save(filename)


def paint(event):
    """This method allows a user to draw on the canvas"""
    color = 'white'
    x1, y1 = (event.x-1), (event.y-1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill=color, width=30)
    draw.line([x1, y1, x2, y2], fill=color, width=30)


def get_number_from_picture():
    """Get the image ready for inference"""
    from PIL import Image
    im = Image.open(str(cwd)+"/temp.png")
    newsize = (28, 28)
    im = im.resize(newsize)

    # Convert the image into a 2d array and reshape it to the desired input
    result_arr = []
    for i in range(28):
        result_arr.append(np.asarray(im)[i][:, 0].tolist())
    pic_array = np.array(result_arr) / 255.0
    pic_array = pic_array.reshape((1, 28, 28))

    """Load the Model and weights"""
    saved_model = model.create_model() # load the model from the model's file
    weights_dir = str(cwd)+'/model_weights'
    weights = os.listdir(str(cwd)+'/model_weights') # Weights directory
    saved_model.load_weights(weights_dir+'/'+str(sorted(weights)[-1]))

    prediction = saved_model.predict(x=pic_array)
    print("\n\n\nThe predicted value is: "+str(np.argmax(prediction))+"\n\n\n")

    del saved_model


if __name__ == '__main__':

    print("""
    ####################################

    Step 1: Draw your single digit

    Step 2: Save the digit at the bottom

    Step 3: Close out the window

    ####################################
    """)

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
