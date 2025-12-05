# Import pyplot module from matplotlib for plotting graphs
from matplotlib import pyplot as plt

# Import numpy for numerical operations
import numpy as np

# Import pandas for data manipulation and reading CSV files
import pandas as pd

# Import matplotlib itself to configure its backend
import matplotlib

# Set the backend for matplotlib to 'TkAgg' for GUI support (especially in PyCharm)
matplotlib.use('TkAgg')  # Ensure compatibility with PyCharm

# -----------------------------------
# Load and Visualize Dataset
# -----------------------------------

# Load the image data CSV file into a DataFrame
data = pd.read_csv("D:\DATASET\hmnist_28_28_RGB.csv")

# Load metadata CSV (labels and extra info) into another DataFrame
meta_df = pd.read_csv("D:\DATASET\HAM10000_metadata.csv")

# Count how many times each diagnosis (dx) appears in metadata
distribution = meta_df['dx'].value_counts()

# Plot the distribution of skin cancer classes as a graph
distribution.plot()

# Add title to the plot
plt.title('Distribution of Skin Cancer Classes')

# Display the plot; `block=True` keeps the plot open
plt.show(block=True)

# -----------------------------------
# Handle Class Imbalance
# -----------------------------------

# Import RandomOverSampler to balance class distribution by duplicating minority classes
from imblearn.over_sampling import RandomOverSampler

# Create an oversampling object
sampler = RandomOverSampler()

# Separate the image data and label column, then apply oversampling
XData, yData = sampler.fit_resample(data.drop(columns=['label']), data['label'])

# Convert oversampled data into numpy array, reshape into 28x28 RGB images, and normalize (0-1 range)
XData = np.array(XData).reshape((-1, 28, 28, 3)) / 255

# -----------------------------------
# Split Data for Training and Testing
# -----------------------------------

# Import method to split dataset
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80% training, 20% testing)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(XData, yData, test_size=0.2)

# -----------------------------------
# Build the CNN Model
# -----------------------------------

# Import necessary layers and tools from K
# eras for deep learning
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import tensorflow as tf

# Create a sequential model (layer by layer)
model = Sequential([
    # First convolutional layer with 32 filters of size 2x2, input is 28x28 RGB
    Conv2D(32, (2, 2), input_shape=(28, 28, 3)),
    Activation('swish'),  # Use swish activation for better non-linearity
    MaxPooling2D(pool_size=(2, 2)),  # Reduce dimensions using max pooling

    # Second convolutional block
    Conv2D(32, (2, 2)),
    Activation('swish'),
    MaxPooling2D(pool_size=(2, 2)),

    # Third convolutional block
    Conv2D(64, (2, 2)),
    Activation('swish'),
    MaxPooling2D(pool_size=(2, 2)),

    # Flatten the 3D output into 1D
    Flatten(),

    # Fully connected dense layer with 64 neurons
    Dense(64),
    Activation('swish'),

    # Dropout layer to prevent overfitting
    Dropout(0.5),

    # Output layer with 7 neurons (for 7 skin cancer classes) and softmax activation
    Dense(7),
    Activation('softmax')
])

# Compile the model with loss function, optimizer, and evaluation metric
model.compile(
    loss='sparse_categorical_crossentropy',  # Loss for integer labels
    optimizer='nadam',  # Optimizer that combines RMSProp and Nesterov momentum
    metrics=['accuracy']  # Track accuracy
)

# -----------------------------------
# Train the Model
# -----------------------------------

# Create a callback to save the model with the best validation accuracy
callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='skin2.h5',  # Save model to this file
    monitor='val_accuracy',  # Monitor validation accuracy
    mode='max',
    verbose=1  # Print when saving the model
)

# Create early stopping callback to stop training when validation accuracy stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=10,  # Stop after 10 non-improving epochs
    restore_best_weights=True  # Restore the best weights automatically
)

# Fit (train) the model using training data; validate using test set
history = model.fit(
    Xtrain, Ytrain,
    epochs=100,  # Train for up to 100 epochs
    validation_data=(Xtest, Ytest),  # Use test set for validation
    callbacks=[callback, early_stopping]  # Add both callbacks
)

# -----------------------------------
# Evaluate the Model
# -----------------------------------

# Evaluate accuracy and loss on the training set
model.evaluate(Xtrain, Ytrain)

# Evaluate accuracy and loss on the test set
model.evaluate(Xtest, Ytest)

# -----------------------------------
# Plot Training and Validation Loss
# -----------------------------------

# Extract training history (loss and accuracy)
history_dict = history.history

# Get training and validation loss lists
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

# Plot validation loss over epochs
plt.plot(range(1, len(loss_values) + 1), val_loss_values, label='Validation Loss', marker='+', linewidth=2.0)

# Plot training loss over epochs
plt.plot(range(1, len(loss_values) + 1), loss_values, label='Training Loss', marker='4', linewidth=2.0)

# Add plot title and labels
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show(block=True)

# -----------------------------------
# Plot Training and Validation Accuracy
# -----------------------------------

# Get training and validation accuracy lists
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

# Plot validation accuracy
plt.plot(range(1, len(acc_values) + 1), val_acc_values, label='Validation Accuracy', marker='+', linewidth=2.0)

# Plot training accuracy
plt.plot(range(1, len(acc_values) + 1), acc_values, label='Training Accuracy', marker='4', linewidth=2.0)

# Add plot title and labels
plt.title('Training and Validation Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show(block=True)

# -----------------------------------
# Confusion Matrix for Predictions
# -----------------------------------

# Import classification report and confusion matrix tools
from sklearn.metrics import classification_report, confusion_matrix

# Import seaborn for heatmap visualization
import seaborn as sns

# Predict the class probabilities for test set
y_pred_test = np.argmax(model.predict(Xtest), axis=1)  # Take argmax to get predicted class

# Generate confusion matrix
cm = confusion_matrix(Ytest, y_pred_test)

# Plot confusion matrix as heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show(block=True)

# -----------------------------------
# Frequency Bar Chart of Skin Cancer Types
# -----------------------------------

# Count frequency of each skin cancer type again from metadata
skin_cancer_counts = meta_df['dx'].value_counts()

# Set figure size for better display
plt.figure(figsize=(6, 4))

# Create bar chart of skin cancer types
skin_cancer_counts.plot(kind='bar', color='skyblue')

# Add chart title and axis labels
plt.title('Frequency of Different Types of Skin Cancer')
plt.xlabel('Skin Cancer Types')
plt.ylabel('Frequency')

# Rotate x-axis labels to prevent overlapping
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent clipping of labels
plt.tight_layout()

# Show the final plot
plt.show(block=True)