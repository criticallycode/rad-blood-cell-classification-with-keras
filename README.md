# rad-blood-cell-classification-with-keras

This repo demonstrates how to fine-tune a deep neural network implemented in Keras. The deep neural network is based on the VGG16 network, fine-tuned with layers unfrozen and added. The dataset that is being used in the Blood Cell Images dataset, which is available [HERE](https://www.kaggle.com/paultimothymooney/blood-cells). The repo demonstrates how to unfreeze layers in a network and how to add new layers to a previously defined network architecture. A Jupyter notebook is included with this repo so that the user can play with the code and try adjusting parameters.

The initial portion of the code imports all libraries, sets up the directories the images are in, and declares some necessary variables.

```Python
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import os
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.applications import VGG16

train_dataset_path = "/images/TRAIN/"
test_dataset_path = "/images/TEST/"
pred_dataset_path = "/images/TEST_SIMPLE/"

list_dir = os.listdir(train_dataset_path)
num_classes = len(list_dir)
batch = 16
classes = ['EOSINOPHIL','LYMPHOCYTE','MONOCYTE','NEUTROPHIL']
```

The following section creates the training and testing datasets, and then creates generators from them.

```Python
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dataset_path,
        target_size=(240, 320),
        batch_size=batch,
        shuffle=True,
        seed=None,
        class_mode="categorical")

test_generator = test_datagen.flow_from_directory(
        test_dataset_path,
        target_size=(240, 320),
        batch_size=batch,
        shuffle=True,
        seed=None,
        class_mode="categorical")
```

The next section creates the base model and unfreezes certain layers.

```Python
vgg_conv = VGG16(weights='imagenet',
                 include_top=False,
                 input_shape=(240, 320, 3))

for layer in vgg_conv.layers[:-8]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)
```

After this, new layers are added and the entire model is compiled.

```Python
new_model = Sequential()
new_model.add(vgg_conv)
new_model.add(Flatten())
new_model.add(BatchNormalization())
new_model.add(Dense(256, activation='relu'))
new_model.add(Dropout(0.2))
new_model.add(BatchNormalization())
new_model.add(Dense(128, activation='relu'))
new_model.add(Dropout(0.2))
new_model.add(BatchNormalization())
new_model.add(Dense(64, activation='relu'))
new_model.add(BatchNormalization())
new_model.add(Dropout(0.2))
new_model.add(Dense(len(classes), activation='softmax'))
print(new_model.summary())
new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Next, some callbacks are specified and the model is fit.

```Python
filepath = "weights_vgg16.hdf5"
callbacks = [ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
              ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min', min_lr=0.00001),
              EarlyStopping(monitor= 'val_loss', min_delta=1e-10, patience=15, verbose=1, restore_best_weights=True)]

# try "train_generator.n // train_generator.batch_size " for steps per epoch and validation steps
train_records = new_model.fit_generator(train_generator,
         epochs = 80,
         steps_per_epoch= 100,
         validation_data = test_generator,
         validation_steps= 100,
         callbacks = callbacks,
         verbose = 1)
```

The final section evaluates the classification model.

```Python
# visualize training loss
# declare important variables
training_acc = train_records.history['acc']
val_acc = train_records.history['val_acc']
training_loss = train_records.history['loss']
validation_loss = train_records.history['val_loss']

# gets the length of how long the model was trained for
num_epochs = range(1, len(training_acc) + 1)

# plot the loss across the number of epochs
plt.figure()
plt.plot(num_epochs, training_loss, label='Training Loss')
plt.plot(num_epochs, validation_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure()
plt.plot(num_epochs, training_acc, label='Training Accuracy')
plt.plot(num_epochs, val_acc, label='Training Accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# evaluate generator
score = new_model.evaluate_generator(test_generator, verbose=1, steps=16)
print('\nAchieved Accuracy:', score[1],'\n')
```

