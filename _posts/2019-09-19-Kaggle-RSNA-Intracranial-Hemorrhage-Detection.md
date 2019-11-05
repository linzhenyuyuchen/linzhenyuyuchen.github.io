---
layout:     post
title:      Kaggle RSNA Intracranial Hemorrhage Detection
subtitle:   Identify acute intracranial hemorrhage and its subtypes
date:       2019-09-19
author:     LZY
header-img: img/hemorrhage-types.png
catalog: true
tags:
    - Kaggle
---

# RSNA Intracranial Hemorrhage Detection

[Reference](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)

Hemorrhage in the head (intracranial hemorrhage) is a relatively common condition that has many causes ranging from trauma, stroke, aneurysm, vascular malformations, high blood pressure, illicit drugs and blood clotting disorders. The neurologic consequences also vary extensively depending upon the size, type of hemorrhage and location ranging from headache to death. The role of the Radiologist is to `detect the hemorrhage`, `characterize the hemorrhage subtype`, `its size` and to determine if the hemorrhage might be jeopardizing critical areas of the brain that might require immediate surgery.

## Hemorrhage Types

While all acute (i.e. new) hemorrhages appear dense (i.e. white) on computed tomography (CT), the primary imaging features that help Radiologists determine the subtype of hemorrhage are the location, shape and proximity to other structures (see table).

![](/img/Meninges-en.png)

- `Intraparenchymal` (脑实质) hemorrhage is blood that is located completely within the brain itself

- `intraventricular` (脑室内) or `subarachnoid` (蛛网膜下) hemorrhage is blood that has leaked into the spaces of the brain that normally contain cerebrospinal fluid (the ventricles or subarachnoid cisterns)

- Extra-axial hemorrhages are blood that collects in the tissue coverings that surround the brain (e.g. `subdural` (硬膜下) or `epidural` (硬膜外) subtypes). ee figure.)

![](/img/subtypes-of-hemorrhage.png)

>Patients may exhibit more than one type of cerebral hemorrhage, which c may appear on the same image. While small hemorrhages are less morbid than large hemorrhages typically, even a small hemorrhage can lead to death because it is an indicator of another type of serious abnormality (e.g. cerebral aneurysm)

# DICOM

## 像素值（灰度值）转换为CT值

CT值的单位是Hounsfield，简称为Hu，范围是-1024-3071。用于衡量人体组织对X射线的吸收率，设定水的吸收率为0Hu

在DICOM图像读取的过程中，我们会发现图像的像素值有可能不是这个范围，通常是0-4096，这是我们常见到的像素值或者灰度值，这就需要我们在图像像素值（灰度值）转换为CT值

首先，需要读取两个DICOM Tag信息，（0028|1052）：rescale intercept和（0028|1053）：rescale slope.

`CT值`: Hu = pixel * slope + intercept


# .png

https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/110840#latest-646974

## 512x png images of training data

[Source](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/110223#latest-638116)

## 224x png images of training data

[Train](https://www.kaggle.com/taindow/rsna-train-stage-1-images-png-224x/download)

[Test](https://www.kaggle.com/teeyee314/rsnatest224)

## 128x png images of training data

[Source](https://www.kaggle.com/guiferviz/rsna_stage1_png_128)

# Gradient & Sigmoid Windowing

- No Windowing

- Brain Windowing

clip everything outside that range so that there's more contrast in the brain-range.

- Metadata Windowing

The DICOM images come with metadata specifying a window center and width. We could also use these values instead of the fixed range from above.

- One Window, Three Channels

Since we'd like to eventually export the scans as png files, we have 3 channels (R,G,B) to work with. If we're only going to use one window setting, we can try to improve the contrast by spreading it out across all 3 channels.

- Gradient Windowing

We can spread our a single window across channels a different way, by mapping the pixel values to a gradient.

- Brain + Subdural + Bone Windowing
- Exclusive Windowing
- Gradient (Brain + Subdural + Bone) Windowing

combine a few previous ideas by averaging 3 different window settings and then mapping the results to a gradient.

- Sigmoid Windowing
- Sigmoid (Brain + Subdural + Bone) Windowing
- Sigmoid Gradient (Brain + Subdural + Bone) Windowing

# See like a Radiologist with Systematic Windowing

1. Brain Matter window : W:80 L:40
2. Blood/subdural window: W:130-300 L:50-100
3. Soft tissue window: W:350–400 L:20–60
4. Bone window: W:2800 L:600
5. Grey-white differentiation window: W:8 L:32 or W:40 L:40

`L = window level or center`
`W = window width or range`

*Example*:

Brain Matter window

L = 40

W = 80

Voxels displayed range from 0 to 80

(  Lower limit = 40 - (80/2), upper limit = 40 + (80/2)  )

Voxel values outside this range will be completely black or white.

# Model

## DenseNet

[keras applications](https://github.com/keras-team/keras-applications/releases)

[denseNet Assets](https://github.com/keras-team/keras-applications/releases/tag/densenet)

# Image Data Augmentation

CenterCrop(200, 200)
Resize(224, 224)
HorizontalFlip()
RandomBrightnessContrast()
ShiftScaleRotate()
GaussianBlur()

# LB score

Model|Image Size|lr|Train Set|Batch Size|Epoch|Score
-|:-:|:-:|:-:|:-:|:-:|-:
DenseNet|128x|-|All|64|10|0.116
DenseNet|224x|2e-5|All|64|10|0.100
DenseNet|224x|2e-5|All|64|2|0.095
DenseNet|224x|2e-5|All|64|30|0.286
ResNext101_32x8d_wsl|224x|2e-5|All|64|2|0.085
ResNext101_32x32d_wsl|224x|2e-5|All|64|2|0.084
ResNext101_32x32d_wsl|224x|2e-5|All|64|6|0.090
EfficientNetB0|224x|2e-5|0.9|32|5|0.120
SeResnet50|224x|6e-5|All|28|3|0.090
SeResnet50|dicom3window|6e-5(multi)|All|28|3|0.075


# Inference



