# blpneumonia
Classification of pneumonia from Chest X-ray images

This code base allows you to train, validate and test different models (Alexnet, Resnet18 -34 -50, Densenet121) on the pneumonia chest x-ray dataset. 
Included in this repo is a toy dataset, the full dataset can be downloaded from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia.
All you need to do is ensure the data is added to this repository and change the root datafolder argument like so:

``` 
python run.py --datafolder new_data
```

The aim of this repo is not only to allow you to test different models, but also to allow you to test different data augmentation parameter configurations. 
Namely to play around with different values for rotations, shears and scaling, which can be done using the command line, e.g.:

``` 
python run.py --rotation 5 --shear 20 --scale [0.8,1.2]
```

Have fun!
