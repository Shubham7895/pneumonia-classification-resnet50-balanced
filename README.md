# Pneumonia Classification using ResNet-50

This project implements fine-tuning of a ResNet-50 model for classifying pneumonia from chest X-ray images using the MedMNIST Pneumonia dataset.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run the training

Place the `pneumoniamnist.npz` dataset file in the root directory, then run:

```bash
python train.py
```

## Evaluate the model

After training:

```bash
python evaluate.py
```

## Dataset

Dataset used: [MedMNIST PneumoniaMNIST](https://medmnist.com/)

## Output

- Trained ResNet-50 model saved as `resnet50_pneumonia.h5`
- Classification report and confusion matrix for evaluation
