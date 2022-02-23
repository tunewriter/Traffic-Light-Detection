import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from skimage.io import imread
import os
#   import seaborn as sn
import cv2
from read_label_file import get_all_labels
import random
from collections import Counter

# Check if tf gpu support is working
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# path where the images are located
PATH = './rgb/img_bmp/'


# --------------------
# Data Preprocessing
# --------------------


#   formats the label to a dict in our desired format
def label_format(label):
    return {
        'state': label['boxes'][0]['label'],
        'occluded': label['boxes'][0]['occluded'],
        'x_max': label['boxes'][0]['x_max'],
        'x_min': label['boxes'][0]['x_min'],
        'y_max': label['boxes'][0]['y_max'],
        'y_min': label['boxes'][0]['y_min'],
        'path': label['path']
    }


#   input: key of an image
#   output: path where the file is located
def key_to_path(key):
    return PATH + key + '.bmp'


#   input: key of a file
#   output: normalized array of the image containing pixel level data
def key_to_img_array(key):
    return cv2.imread(key_to_path(key)) / 255


#   input: path of a file
#   output: normalized array of the image containing pixel level data
def path_to_img_array(path):
    return cv2.imread(path) / 255


#   input: label of an image
#   output: array of len = 4 with the 2 coordinates of the bounding box
def label_to_bb(label):
    return [label['x_max'] / 1280, label['x_min'] / 1280, label['y_max'] / 720, label['y_min'] / 720]


#   input: path of an image
#   output: key (file name) of the image
def path_to_key(path):
    return os.path.basename(path)[:-4]


#   input: index of a image in the X_data dict
#   output: key (file name) of the image
def index_to_key(index):
    return os.path.basename(X_data[index])[:-4]


#   since cv2 reads images as BGR, we need to convert it to RGB to display them properly
def BGR_to_RGB(img):
    result = img
    for i in result:
        for j in i:
            temp = j[2]
            j[2] = j[0]
            j[0] = temp
    return result


#   input: normalized array of the image containing pixel level data
#   loads image into plt
def show_img_array(img_array):
    plt.imshow(BGR_to_RGB(img_array))


#   input: bounding box array, color of the bounding box
#   loads bounding box into plt
def draw_bb(bb, color):
    #   label['x_max'] / 1280, label['x_min'] / 1280, label['y_max'] / 720, label['y_min'] / 720
    width = (bb[0] - bb[1])
    height = (bb[2] - bb[3])
    plt.gca().add_patch(Rectangle((bb[1] * 1280, (bb[3] + height) * 720),
                                  width * 1280,
                                  -height * 720,
                                  linewidth=1, edgecolor=color,
                                  facecolor='none'))
    print('TL size: ', bb_to_pixel_size(bb))


#   input: path of an image, two bounding box arrays
#   loads image and the two bounding boxes in plt
def show_img_two_bb(path, bb1, bb2):
    print(data_dict[path_to_key(path)])
    show_img_array(path_to_img_array(path))
    draw_bb(bb1, 'g')
    draw_bb(bb2, 'r')


#   input: path of an image, bounding box array,
#          mapped number of traffic light state [0-4], array with the predicted probabilities
#   loads image, bounding boxes with 2 labels for the predicted and the true state of the light into plt
def show_label_and_bb(path, bb, class_nr, pred):
    show_img_array(path_to_img_array(path))
    draw_bb(bb, 'g')
    plt.xlabel('predicted: ' + light_classes[np.argmax(pred)] + '\n'
                                                                'truth: ' + light_classes[class_nr])


#   input: bounding box array
#   output: size/area of the bounding box in pixel
def bb_to_pixel_size(bb):
    width = (bb[0] - bb[1]) * 1280
    height = (bb[2] - bb[3]) * 720
    return width * height


#   loading all labels
train_labels_all = get_all_labels("train.yaml")
test_labels_all = get_all_labels("test.yaml")

#   label format before formatting
"""
{'boxes': [{'label': 'GreenLeft', 'occluded': False, 'x_max': 715.625, 'x_min': 685.75, 'y_max': 270.25, 'y_min': 205.0}], 
'path': 'D:\\Traffic Light Dataset\\rgb\\train\\2017-02-03-11-44-56_los_altos_mountain_view_traffic_lights_bag\\208562.png'}
"""

#   empty lists to put only labels from images with one boxes/traffic light
train_labels = []
test_labels = []

#   iterating through all labels
#   take only labels with one box/traffic light and whose traffic light is not occluded
for label in train_labels_all:
    if len(label["boxes"]) == 1 and label['boxes'][0]['occluded'] is False:
        train_labels.append(label_format(label))

for label in test_labels_all:
    if len(label["boxes"]) == 1 and label['boxes'][0]['occluded'] is False:
        test_labels.append(label_format(label))

#   storing all labels in a dict (hash table)
#   with the file name (without the format) as the key
labels_test_dict = {}
for i in test_labels:
    key = os.path.basename(i['path'])[:-4]
    labels_test_dict[key] = i

labels_train_dict = {}
for i in train_labels:
    key = os.path.basename(i['path'])[:-4]
    labels_train_dict[key] = i

#   Renaming the actual test files with ' (2)' ending (used once)

#   Originally test and training images were in separate folders
#   After joining these some files had the same file name,
#   so my os renamed the name of the test images with adding ' (2)' to the file name
#
#   Attention: Only the files from our filtered labels got renamed, for other files,
#   you need to modify and rerun the script
'''
counter = 0
temp_dict = {}
for i in labels_test_dict:
    if os.path.isfile('./rgb/img_bmp/' + i + ' (2).bmp'):
        print(i, 'exists!')
        os.rename(r'./rgb/img_bmp/' + i + ' (2).bmp', r'./rgb/img_bmp/' + i + 'b.bmp')
        temp_dict[i] = None
        counter += 1
print(counter)
'''

#   renaming keys in the dict to prevent same key name when joining (what we do afterwards)
temp_dict = {}
for i in labels_test_dict:
    if os.path.isfile('./rgb/img_bmp/' + i + 'b.bmp'):
        temp_dict[i] = None
for i in temp_dict:
    labels_test_dict[i + 'b'] = labels_test_dict.pop(i)


#   input: two dicts
#   output: merged dict
def merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


#   merge train and test label dicts
#   to have one dict with all data with image number as key
data_dict = merge(labels_train_dict, labels_test_dict)

#   X_data: Array of paths
#   y_data: Array of bounding-box arrays (my initial plan)
X_data, y_data = [], []
for i in data_dict:
    X_data.append(key_to_path(i))
    y_data.append(label_to_bb(data_dict[i]))

#   y_data_light: Array of 'states' [0-4] of traffic lights
y_data_light = []
for i in data_dict:
    if 'red' in data_dict[i]['state'].lower():
        y_data_light.append(1)
    elif 'green' in data_dict[i]['state'].lower():
        y_data_light.append(2)
    elif 'yellow' in data_dict[i]['state'].lower():
        y_data_light.append(3)
    elif 'off' in data_dict[i]['state'].lower():
        y_data_light.append(0)
    else:
        print('label not matching!')

#   mapping 'state' number to the color (string)
light_classes = ['off', 'red', 'green', 'yellow']

#   shuffling data to change the order
combined = list(zip(X_data, y_data, y_data_light))
random.shuffle(combined)
X_data, y_data, y_data_light = zip(*combined)

#   Array with the sizes of the bounding boxes
bb_sizes = []
for i in y_data:
    bb_sizes.append(bb_to_pixel_size(i))

#   set where to split between test & train data
split = len(X_data) - 200
print("Total number of images: ", len(X_data), "\n",
      "Number of training images: ", split, "\n",
      "Number of test images: ", len(X_data) - split)

#   separate train and test data
X_data_train = X_data[:split]
y_data_light_train = y_data_light[:split]
y_data_train = y_data[:split]

X_data_test = X_data[split:]
y_data_light_test = y_data_light[split:]
y_data_test = y_data[split:]


# Balancing four categories (our data is imbalanced)
#

#   Counting the number of occurrences of each traffic light
counted = Counter(y_data_light_train)
print("Number of images in each category: ", counted)

#   zipping together the three groups to balance them all at once
triple_list = list(zip(X_data_train, y_data_light_train, y_data_train))

#   script to balance the data by oversampling
target_amount = counted[2]
missing_0 = target_amount - counted[0]
missing_1 = target_amount - counted[1]
missing_3 = target_amount - counted[3]
missing = [missing_0, missing_1, 0, missing_3]

triple_list_extended = triple_list

i = 0
while sum(missing) > 0:
    if i == len(triple_list):
        i = 0
    if triple_list[i][1] != 2:
        for j in [0, 1, 3]:
            if triple_list[i][1] == j and missing[j] > 0:
                triple_list_extended.append(triple_list[i])
                missing[j] -= 1
                break
    i += 1

random.shuffle(triple_list_extended)

#   unzip the balanced data to the three groups
X_data_train_extended, y_data_light_train_extended, y_data_train_extended = zip(*triple_list_extended)
print("Number of images in each category: ", Counter(y_data_light_train_extended))

# --------------
# Data Handling
# --------------

#   number of images in one forward/backward pass
batch_size = 4


#   generator class to input only a batch at once (for memory space reasons)


class MyCustomGenerator(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        return np.array([
            imread('/Traffic Light Dataset/' + str(file_name))
            for file_name in batch_x]) / 255.0, np.array(batch_y)


#   batch generator for training and testing / validation
my_training_batch_generator = MyCustomGenerator(X_data_train_extended, y_data_light_train_extended, batch_size)
my_validation_batch_generator = MyCustomGenerator(X_data_test, y_data_light_test, batch_size)

# -----------------------------
# Convolutional Neural Network
# -----------------------------


#   where to save model weights checkpoint
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#   CNN model
cnn = models.Sequential([
    layers.Conv2D(filters=96, kernel_size=(7, 7), activation='relu', input_shape=(720, 1280, 3)),
    layers.MaxPool2D((4, 4)),

    layers.Dropout(0.4),

    layers.Conv2D(filters=32, kernel_size=(6, 6), activation='relu'),
    layers.MaxPool2D((7, 7)),

    layers.Dropout(0.4),

    layers.Conv2D(filters=16, kernel_size=(4, 4), activation='relu'),
    layers.MaxPool2D((4, 4)),

    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')
])

#   compile the CNN model
cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

#   specifying how to save checkpoints
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
#   prints a summary of our model
cnn.summary()

#   train the model
#   epochs: specify the number of epochs
history = cnn.fit_generator(generator=my_training_batch_generator,
                            epochs=1,
                            verbose=1,
                            validation_data=my_validation_batch_generator,
                            callbacks=[cp_callback]
                            )


#   loading weights and saving model
# cnn.load_weights(checkpoint_path)
# cnn.save('my_model1.h5')
# cnn.save('saved_model/my_model')


#   specific loss function
def l2_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred))))


#   returns array with the output-arrays from the test set
y_pred = cnn.predict(my_validation_batch_generator)

#   array with the predicted numbers / classes
y_classes = [np.argmax(e) for e in y_pred]

#   confusion matrix "that many y_test got recognized as y_classes"
cm = tf.math.confusion_matrix(y_data_light_test, y_classes)

#   visualizing confusion matrix
plt.figure(figsize=(10, 7))
# sn.heatmap(cm, annot=True, fmt='d')
plt.plot(cm)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

#   prints confusion matrix
print('confusion matrix: ', cm)

#   chose random examples and display them with a drawn bounding box
#   and shows how the model predicted the state and how it actually is
while True:
    n = int(random.random() * len(y_data_test))
    bb = y_data_test[n]
    class_nr = y_data_light_test[n]
    pred = y_pred[n]

    print('truth:', light_classes[class_nr])
    print('predicted', pred)

    show_label_and_bb(X_data_test[n], bb, class_nr, pred)
    print()

    plt.show()
