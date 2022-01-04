#!/bin/bash

echo "Souhaitez-vous installer les modules python ?(Y/n)"

read ans

if [[ $ans == 'Y' || -z $ans ]];then
    pip install numpy
    pip install -U scikit-learn
    pip install scipy
    pip install matplotlib
    pip install opencv-python
    pip install scikit-image
    pip install tensorflow
    pip install --upgrade tensorflow
    pip install keras
else
    echo rien a été installé
fi