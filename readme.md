# Project Title

Face classification and detection

# Prerequisites

## Installing

python==3.7.1

matplotlib==3.0.1

pillow==5.3.0

skimage==0.15.0

pytorch==0.4.1

sklearn==0.20.0

# Folder

## Preparation

The FDDB datasets have two folders: originalPics, FDDB-folds (The pictures are in originalPucs and the annotations are in FDDB-folds)

You should prepare 7 empty folders: Model, samples, samples/positive, samples/negative, classify, detection, detection_Results

## Generation

The samples will be generated in samples. The positive samples will be in samples/positive and the negative samples will be in samples/negativ.

The test samples are divided into two parts: the first part is samples for classification which will be generated in classify folder. The second part is samples for detection which will be genreated in detection folder.

The rsults of detection will be generated in detection_Results folder. Th results mean    which picture that the classification detects as a face.

# Command

## sample generation

python generate.py

## face classification & detection

python main.py [-h] -model {logistic,CNN,SVM,Fisher} [-opt {SGD,Langevin}] [-load {y,n}] [-kernel {linear,RBF}]