---
layout:     post
title:      Keras fit, fit_generator & train_on_batch
subtitle:   When & How to use them
date:       2019-09-25
author:     LZY
header-img: img/lossfunction.jpeg
catalog: true
tags:
    - Keras
---

# When to use Keras fit, fit_generator & train_on_batch

[Reference](https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/)

## .fit

**Primary Assumptions:**

1. Our `entire training set` can fit into RAM

2. There is `no data augmentation` going on (i.e., there is no need for Keras generators)

Our network will be trained on the raw data.

The raw data itself will fit into memory — we have no need to move old batches of data out of RAM and move new batches of data into RAM.

Furthermore, we will not be manipulating the training data on the fly using data augmentation.

```python
model.fit(trainX, trainY, batch_size=32, epochs=50)
```

## .fit_generator

**Factors:**

1. Real-world datasets are often `too large` to fit into memory.

2. They also tend to be challenging, requiring us to perform `data augmentation` to avoid overfitting and increase the ability of our model to generalize.

```python
# initialize the number of epochs and batch size
EPOCHS = 100
BS = 32

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# train the network
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)
```

We then initialize aug , a Keras `ImageDataGenerator`  object that is used to apply data augmentation, `randomly translating, rotating, resizing, etc.` images on the fly.

Each new batch of data is randomly adjusted according to the parameters supplied to `ImageDataGenerator`.

Performing data augmentation is a form of regularization, enabling our model to generalize better.

However, applying data augmentation implies that our training data is `no longer “static”` — the data is constantly changing.

## Q & A

**Why do we need `steps_per_epoch` ?**

Keep in mind that a Keras data generator is meant to loop infinitely — it should never return or exit.

Since the function is intended to loop infinitely, Keras has no ability to determine when one epoch starts and a new epoch begins.

Therefore, we compute the `steps_per_epoch`  value as `the total number of training data points divided by the batch size`. Once Keras hits this step count it knows that it’s a new epoch.

## .train_on_batch

For deep learning practitioners looking for the `finest-grained` control over training your Keras models, you may wish to use the `.train_on_batch`  function which is suitable for an advanced deep learning practitioner/engineer.

```python
model.train_on_batch(batchX, batchY)
```

You’ll typically use the . `train_on_batch`  function when you have very explicit reasons for wanting `to maintain your own training data iterator, such as the data iteration process being extremely complex and requiring custom code.`