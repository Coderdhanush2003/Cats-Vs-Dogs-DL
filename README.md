# Cats-Vs-Dogs-DL

A Deep Learning computer vision project built with TensorFlow and Keras that classifies images of pets into cats or dogs. The repository showcases two distinct methodologies: a custom Convolutional Neural Network (CNN) built entirely from scratch with data augmentation, and a high-performance Transfer Learning approach utilizing a pre-trained MobileNetV2 base.

## 📊 Dataset Overview

The project processes the standard `cats_vs_dogs` dataset sourced from `tensorflow_datasets`, dynamically handling data extraction and cleaning:

* **Training Subset:** 80% of the dataset used to train features and optimize network weights.
* **Testing/Validation Subset:** 20% of the dataset reserved as a completely unseen validation pool for final evaluation.
* **Data Integrity Check:** Automated ingestion automatically identified and safely skipped 1,738 corrupted image records.
* **Image Geometry:** Raw images of varying dimensional constraints are dynamically parsed and uniformly resized to a standard $160 \times 160$ spatial input footprint with 3 color channels (RGB).

## 🛠️ Workflow & Code Structure

* **Data Pipelines & Pipeline Prefetching:** Mapping raw image records into structured elements, scaling raw pixel intensities from `[0, 255]` to a global floating-point range of `[0.0, 1.0]`, and applying `.batch(32)` paired with `.prefetch(tf.data.AUTOTUNE)` to clear hardware input bottlenecks.
* **On-the-Fly Data Augmentation:** Integrating a sequential data augmentation block (`RandomFlip("horizontal")`, `RandomRotation(0.1)`, and `RandomZoom(0.1)`) directly at the front of the model to artifically expand data variety and aggressively minimize overfitting.
* **Custom CNN Feature Extraction:** Stacking four progressive `Conv2D` hidden layers (8, 16, 32, and 64 filters) utilizing ReLU activations. Each step incorporates `BatchNormalization` for structural stability, `MaxPooling2D` for downsampling, and `Dropout (0.2)` to penalize network over-reliance on localized pixels.
* **Dimensional Pooling:** Employing a `GlobalAveragePooling2D` layer to gracefully compress the multi-dimensional feature space into a flat spatial representation without flattening spatial layout context.
* **Compilation & Adaptive Training:** Operating with the Adam optimizer and tracking mathematical convergence via `BinaryCrossentropy` loss. Training incorporates a `ReduceLROnPlateau` scheduler to slice learning rates by a factor of 0.2 whenever validation loss plateaus for more than 2 epochs.
* **Transfer Learning Alternative (MobileNetV2):** Swapping the custom backbone for a pre-trained `MobileNetV2` engine (frozen weights from ImageNet) paired with a custom dense classification head to establish a hyper-optimized production baseline.

## 📈 Performance Summary

### 1. Custom CNN Model (From Scratch)

| Split Layer | Epoch / Evaluation | Loss Metric | Accuracy Metric |
| --- | --- | --- | --- |
| Training Set | Epoch 1 | 0.6383 | 63.54% |
| Training Set | Epoch 5 | 0.4776 | 76.78% |
| Training Set | Epoch 10 | 0.3849 | 82.02% |
| Validation Set | Final Unseen Evaluation | 0.6000 | 74.53% |

### 2. MobileNetV2 Transfer Learning Model

| Split Layer | Epoch / Evaluation | Loss Metric | Accuracy Metric |
| --- | --- | --- | --- |
| Training Set | Epoch 1 | 0.0842 | 96.85% |
| Training Set | Epoch 3 | 0.0445 | 183.33% |
| Training Set | Epoch 5 | 0.0326 | 98.79% |
| Validation Set | Final Unseen Evaluation | 0.0628 | 97.76% |

## 🧩 Model Layer Topology

### Custom CNN Architecture

```text
Model: "Sequential"
=================================================================
Layer (type)                Output Shape              Param #   
=================================================================
Sequential (Augmentation)   (None, 160, 160, 3)       0         
Conv2D (8 filters, ReLU)    (None, 158, 158, 8)       224       
BatchNormalization          (None, 158, 158, 8)       32        
MaxPooling2D                (None, 79, 79, 8)         0         
Dropout (Rate = 0.2)        (None, 79, 79, 8)         0         
Conv2D (16 filters, ReLU)   (None, 77, 77, 16)        1,168     
BatchNormalization          (None, 77, 77, 16)        64        
MaxPooling2D                (None, 38, 38, 16)        0         
Dropout (Rate = 0.2)        (None, 38, 38, 16)        0         
Conv2D (32 filters, ReLU)   (None, 36, 36, 32)        4,640     
BatchNormalization          (None, 36, 36, 32)        128       
MaxPooling2D                (None, 18, 18, 32)        0         
Dropout (Rate = 0.2)        (None, 18, 18, 32)        0         
Conv2D (64 filters, ReLU)   (None, 16, 16, 64)        18,496    
BatchNormalization          (None, 16, 16, 64)        256       
MaxPooling2D                (None, 8, 8, 64)          0         
GlobalAveragePooling2D      (None, 64)                0         
Dense (ReLU)                (None, 64)                4,160     
Dense (Sigmoid)             (None, 1)                 65        
=================================================================
Total Trainable Parameters: 28,993 (113.25 KB)

```

### MobileNetV2 Transfer Learning Architecture

```text
Model: "Sequential_Transfer"
=================================================================
Layer (type)                Output Shape              Param #   
=================================================================
mobilenetv2_1.00_160 (Func) (None, 5, 5, 1280)        2,257,984 
GlobalAveragePooling2D      (None, 1280)              0         
Dense (ReLU)                (None, 64)                81,984    
Dropout (Rate = 0.3)        (None, 64)                0         
Dense (Sigmoid)             (None, 1)                 65        
=================================================================
Total Trainable Parameters: 82,049 (320.50 KB)
Non-Trainable Parameters: 2,257,984 (8.61 MB)

```

## 🚀 Technologies Used

* **Python 3**
* **TensorFlow / Keras** (Sequential model engineering, Functional API tracking, Convolutional operations, Augmentation mapping, and Transfer Learning backbones)
* **TensorFlow Datasets (TFDS)** (Robust pipeline management and multi-threaded data streaming)
* **NumPy** (High-speed tensor dimensional operations and expansion via `np.expand_dims()`)
* **Google Colab** (Interactive cloud development environment backed by hardware acceleration)
