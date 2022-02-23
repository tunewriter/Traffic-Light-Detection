# Traffic Light Detection With Convolutional Neural Network

## About the Project

In this project I try to detect the state of traffic lights (red, yellow, green, off) using a convolutional neural network with the 

[Bosch Small Traffic Lights Dataset]: https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset

.

### Built With

- Tensorflow
- Keras

## Getting Started

Check the imports at the top of implementation.py to know which libraries & frameworks are necessary.

It's recommended to activate tensorflow gpu-support for performance reasons. Since this can be challenging I recommend to use an anaconda environment.

### Dataset

Download the dataset (≈35GB) from 

[here]: https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset	"Bosch Small Traffic Lights Dataset"

. 

```
@inproceedings{BehrendtNovak2017ICRA,
  title={A Deep Learning Approach to Traffic Lights: Detection, Tracking, and Classification},
  author={Behrendt, Karsten and Novak, Libor},
  booktitle={Robotics and Automation (ICRA), 2017 IEEE International Conference on},
  organization={IEEE}
}
```

Join all training images in one folder and join all test images in one folder.
After that, copy the images from the test folder to the train folder and all images from the test folder with the same file name should be renamed with adding " (2)".

I converted all images to .bmp files with 

[this]: https://github.com/wangjiosw/ImageFormatConversion	"GitHub"

 for better performance.

## Usage

This implementation works successfully for detecting the state of traffic lights in images with one traffic light.
You can load the weights I got after 10 epochs from 

Next possibly steps are improving the accuracy, work on a bounding box detection or detection of multiple bounding boxes in one image.
It comes with functions to read, process and display bounding boxes.



## My Results

With 1687 training images and 200 test images I got an accuracy of 91.5% after 8 epochs.
Before oversampling, 1052 images were green, 465 were red, 141 were off and 29 were yellow.

<!--1061/1061 [==============================] - 321s 295ms/step - loss: 0.9075 - accuracy: 0.6287 - val_loss: 0.7674 - val_accuracy: 0.7400-->

<!--Epoch 00001: saving model to training_1\cp.ckpt-->
<!--Epoch 2/10-->
<!--1061/1061 [==============================] - 316s 298ms/step - loss: 0.3573 - accuracy: 0.8709 - val_loss: 0.3528 - val_accuracy: 0.8850-->

<!--Epoch 00002: saving model to training_1\cp.ckpt-->
<!--Epoch 3/10-->
<!--1061/1061 [==============================] - 311s 293ms/step - loss: 0.2021 - accuracy: 0.9288 - val_loss: 0.2950 - val_accuracy: 0.9000-->

<!--Epoch 00003: saving model to training_1\cp.ckpt-->
<!--Epoch 4/10-->
<!--1061/1061 [==============================] - 312s 294ms/step - loss: 0.1436 - accuracy: 0.9451 - val_loss: 0.2933 - val_accuracy: 0.8900-->

<!--Epoch 00004: saving model to training_1\cp.ckpt-->
<!--Epoch 5/10-->
<!--1061/1061 [==============================] - 311s 293ms/step - loss: 0.1114 - accuracy: 0.9564 - val_loss: 0.4550 - val_accuracy: 0.8950-->

<!--Epoch 00005: saving model to training_1\cp.ckpt-->
<!--Epoch 6/10-->
<!--1061/1061 [==============================] - 314s 296ms/step - loss: 0.0979 - accuracy: 0.9616 - val_loss: 0.3419 - val_accuracy: 0.9000-->

<!--Epoch 00006: saving model to training_1\cp.ckpt-->
<!--Epoch 7/10-->
<!--1061/1061 [==============================] - 313s 295ms/step - loss: 0.0864 - accuracy: 0.9691 - val_loss: 0.3843 - val_accuracy: 0.8950-->

<!--Epoch 00007: saving model to training_1\cp.ckpt-->
<!--Epoch 8/10-->
<!--1061/1061 [==============================] - 316s 298ms/step - loss: 0.0804 - accuracy: 0.9734 - val_loss: 0.3150 - val_accuracy: 0.9150-->

<!--Epoch 00008: saving model to training_1\cp.ckpt-->
<!--Epoch 9/10-->
<!--1061/1061 [==============================] - 320s 301ms/step - loss: 0.0667 - accuracy: 0.9757 - val_loss: 0.3444 - val_accuracy: 0.8850-->

<!--Epoch 00009: saving model to training_1\cp.ckpt-->
<!--Epoch 10/10-->
<!--1061/1061 [==============================] - 311s 293ms/step - loss: 0.0737 - accuracy: 0.9762 - val_loss: 0.3538 - val_accuracy: 0.9050-->

Confusion Matrix:

## License

gpl-3.0

## Contact

## Acknowledgements

- Bosch Small Traffic Lights Dataset with scripts on GitHub (bosch-ros-pkg/bstld)

- [3Blue1Brown]: https://www.youtube.com/c/3blue1brown	"YouTube"

   on YouTube

- [codebasics]: https://www.youtube.com/channel/UCh9nVJoWXmFb7sLApWGcLPQ	"YouTube"

   on Youtube