import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import seaborn as sns

# Paths to the data
PARENT_DIRNAME = os.path.expanduser("~/PRML-MidTerm-Project/")
TRAIN_PATH = os.path.join(PARENT_DIRNAME, "data", "df_train.csv")
TEST_PATH = os.path.join(PARENT_DIRNAME, "data", "df_test.csv")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the EEG dataset
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

# Prepare the features and labels
features = df_train.drop(columns=["state"]).values
labels = df_train["state"].values

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Train, validation, and test split
X_train, X_temp, y_train, y_temp = train_test_split(
    features, labels_encoded, test_size=0.4, random_state=42, stratify=labels_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for batch processing
batch_size = 64
train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False
)

# Define the MLP model
dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_prob=0.5):
        super(MLP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Training loop
def train_model(model, dataloaders, criterion, optimizer, num_epochs=30):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss /= len(dataloaders["train"].dataset)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= len(dataloaders["val"].dataset)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - "
            f"Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}"
        )

    return train_losses, val_losses, train_accuracies, val_accuracies

# Evaluate the model
def evaluate_model(model, dataloader, class_names):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            _, predicted = torch.max(outputs, 1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs)

    accuracy = total_correct / total_samples
    return accuracy, all_labels, all_predictions, np.array(all_probs)

def plot_results(train_losses, val_losses, train_accuracies, val_accuracies, true_labels, predicted_labels, predicted_probs, class_names):
    # Plot Loss and Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Training and Validation Loss")

    ax2.plot(train_accuracies, label="Train Accuracy")
    ax2.plot(val_accuracies, label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Training and Validation Accuracy")

    plt.tight_layout()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    num_classes = len(class_names)
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve((np.array(true_labels) == i).astype(int), predicted_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {class_names[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="best")
    plt.show()

if __name__ == "__main__":
    # Hyperparameters
    input_dim = X_train.shape[1]  # Number of input features
    hidden_dims = [512, 256, 128, 64, 32]  # Hidden layer dimensions
    output_dim = len(np.unique(y_train))  # Number of output classes
    learning_rate = 0.001
    num_epochs = 50

    # Initialize the model, loss function, and optimizer
    model = MLP(input_dim, hidden_dims, output_dim, dropout_prob=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, dataloaders, criterion, optimizer, num_epochs=num_epochs
    )

    # Evaluate on the test set
    test_accuracy, test_labels, test_predictions, test_probs = evaluate_model(
        model, dataloaders["test"], class_names=label_encoder.classes_
    )
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            test_labels, test_predictions, target_names=label_encoder.classes_
        )
    )

    # Plot all results (Loss, Accuracy, Confusion Matrix & ROC Curve)
    plot_results(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        test_labels,
        test_predictions,
        test_probs,
        label_encoder.classes_
    )