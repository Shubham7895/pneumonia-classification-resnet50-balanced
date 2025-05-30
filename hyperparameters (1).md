# Hyperparameter Choices

- **Learning Rate:** 0.0001  
  Fine-tuning with a small learning rate to avoid large weight updates.

- **Batch Size:** 32  
  Balanced size for memory efficiency and convergence speed.

- **Epochs:** 10  
  Early stopping can be introduced if overfitting is detected.

- **Image Size:** 224x224  
  Required size for ResNet-50 input.

- **Optimizer:** Adam  
  Adaptive optimizer with good convergence behavior.

- **Loss Function:** Binary Crossentropy  
  Suitable for binary classification tasks.

- **Class Weights:** {0: 5.00, 1: 0.56}  
  Used to balance the dataset due to class imbalance.
