# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.


## DESIGN STEPS
# Data Preparation and Preprocessing
Collected and organized the dataset into train and test folders.

Applied image transformations such as resizing to 224×224 and converting to tensor for VGG19 compatibility.

Loaded the data using ImageFolder and DataLoader for efficient batching and shuffling.

# Model Design (Transfer Learning Setup)
Loaded the pre-trained VGG19 model from torchvision.models.

Replaced the final fully connected layer (classifier[6]) to match the number of output classes in the dataset.

Froze convolutional layers to retain pre-learned features and fine-tuned only the classifier part.

# Loss Function and Optimizer Selection
Used CrossEntropyLoss for multi-class image classification.

Optimized the classifier layer parameters using Adam optimizer with a learning rate of 0.001.

# Model Training and Validation
Trained the model for multiple epochs, recording both training and validation losses.

Visualized the loss curves to monitor convergence and detect overfitting or underfitting trends.

# Evaluation and Prediction
Evaluated the model performance using accuracy, confusion matrix, and classification report.

Tested predictions on single images to visually confirm model accuracy and reliability.


## RESULT
The pretrained VGG19 model was successfully fine-tuned on the custom image dataset.

The model achieved high training and testing accuracy, showing effective transfer learning performance.

The confusion matrix and classification report confirmed good class-wise prediction.

Sample test images were correctly classified, verifying the model’s ability to generalize well.
