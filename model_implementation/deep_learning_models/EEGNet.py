import sys
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Load dataset
df_train = pd.read_csv("/kaggle/input/data-eeg/df_train.csv")
df_test = pd.read_csv("/kaggle/input/data-eeg/df_test.csv")


# Preprocessing data
def preprocess_data(df, scaler, encoder):
    features = df.drop("state", axis=1)
    labels = df["state"]
    X_scaled = scaler.fit_transform(features)
    y_encoded = encoder.fit_transform(labels)
    return X_scaled, y_encoded


scaler = MinMaxScaler()
encoder = LabelEncoder()

# Prepare training, validation, and test data
X_train_scaled, y_train_encoded = preprocess_data(df_train, scaler, encoder)
X_test, y_test = preprocess_data(df_test, scaler, encoder)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_scaled,
    y_train_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_train_encoded,
)


# Define and Train EEGNet Model
def EEGNet(
    nb_classes,
    Chans,
    Samples,
    kernLength,
    F1,
    D,
    F2,
    norm_rate,
    dropoutRate,
    dropoutType,
):
    """
    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Randomly remove some neurons to avoid overfitting, passed as a string.

    """
    if dropoutType == "SpatialDropout2D":
        dropoutType = SpatialDropout2D
    elif dropoutType == "Dropout":
        dropoutType = Dropout
    else:
        raise ValueError(
            "dropoutType must be one of SpatialDropout2D "
            "or Dropout, passed as a string."
        )

    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    # Block 1
    block1 = Conv2D(
        F1,
        (1, kernLength),
        padding="same",
        input_shape=(Chans, Samples, 1),
        use_bias=False,
    )(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D(
        (Chans, 1),
        use_bias=False,
        depth_multiplier=D,
        depthwise_constraint=max_norm(1.0),
    )(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation("elu")(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    # Block 2
    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding="same")(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation("elu")(block2)
    block2 = AveragePooling2D((1, 4))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    # Flatten, dense, softmax and output
    flatten = Flatten(name="flatten")(block2)
    dense = Dense(nb_classes, name="dense", kernel_constraint=max_norm(norm_rate))(
        flatten
    )
    softmax = Activation("softmax", name="softmax")(dense)

    return Model(inputs=input1, outputs=softmax)


# Model Parameters
Chans = len(
    ["ED_F7", "ED_F3", "ED_P7", "ED_O1", "ED_O2", "ED_P8", "ED_AF4"]
)  # useful channels
Samples = df_train.shape[1] // Chans  # Number of samples per channel

# Reshape data
X_train = X_train.reshape(-1, Chans, Samples, 1)
X_val = X_val.reshape(-1, Chans, Samples, 1)
X_test = X_test.reshape(-1, Chans, Samples, 1)

# Convert the data labels to one-hot encoded
y_train_onehot = to_categorical(y_train, num_classes=3)
y_val_onehot = to_categorical(y_val, num_classes=3)
y_test_onehot = to_categorical(y_test, num_classes=3)

# Initialize EEGNet model
eeg_net = EEGNet(
    nb_classes=3,  # Number of states: Drowsy (class 0), Focused (class 1), Unfocused (class 2)
    Chans=Chans,  # Number of EEG channels
    Samples=Samples,  # Samples per second
    kernLength=64,  # Kernel temporal: suitable for 128Hz
    F1=8,  # Number of temporal filters
    D=2,  # Spatial Filter
    F2=16,  # Total filter number
    norm_rate=0.25,  # Constraint level for the kernel of the Dense layer.
    dropoutRate=0.5,  # Dropout to avoid overfitting
    dropoutType="Dropout",
)

# Compile EEGNet model
eeg_net.compile(
    loss="categorical_crossentropy",  # Multi-layer classification
    optimizer=Adam(learning_rate=0.001),  # Adam optimizer with low learning rate
    metrics=["accuracy"],  # Evaluate by accuracy
)

# Train model
history = eeg_net.fit(
    X_train,
    y_train_onehot,
    epochs=50,
    batch_size=32,
    validation_data=(
        X_val,
        y_val_onehot,
    ),  # Use validation_data to validate in each epoch
)

# Evaluate the model on the test set
y_pred = np.argmax(eeg_net.predict(X_test), axis=1)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print(f"Classification Report:")
print(classification_report(y_pred, y_test))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ROC curve
y_pred_onehot = to_categorical(y_pred, num_classes=3)

n_classes = 3
fpr, tpr, roc_auc = {}, {}, {}

# Calculate ROC for each class
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_onehot[:, i], y_pred_onehot[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure()
colors = ["blue", "orange", "green"]
for i in range(n_classes):
    plt.plot(
        fpr[i],
        tpr[i],
        color=colors[i],
        lw=2,
        label=f"ROC curve for class {i} (AUC = {roc_auc[i]:.2f})",
    )

plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
