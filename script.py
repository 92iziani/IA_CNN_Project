import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras import layers

import numpy as np



print("Keras:", keras.__version__)
print("Tensorflow:", tf.__version__)

def main():
    #csv de la forme img1px1, ..., img1pxn\nimg2px1, ..., img2pxn\nimgmpx1, ..., imgmpxn 
    np.loadtxt(open("test.csv", "rb"), delimiter=",", skiprows=1)


if __name__ == "__main__":
    main()