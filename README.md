# Lung Nodule Classification with Vision Transformers

A notebook-based deep learning project for **binary classification of lung CT image patches** using a Vision Transformer (ViT) implemented with PyTorch and `vit-pytorch`.

The model classifies each input image as **nodule (`NOD`)** or **non-nodule (`NON_NOD`)**. The repository includes training, evaluation, and single-image inference workflows.

## Features

- Custom PyTorch dataset for loading images and binary labels.
- Image resizing, normalization, and stochastic augmentation.
- Vision Transformer training with GPU support when available.
- Training and validation loss and accuracy curves.
- Test accuracy, F1 score, confusion matrix, and ROC curve.
- Model checkpoint export and single-image inference.

## Technologies

Python · PyTorch · torchvision · vit-pytorch · scikit-learn · Pillow · Matplotlib · Seaborn · Jupyter

## Repository contents

| Path                         | Description                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------- |
| `Train.ipynb`                | Dataset loading, augmentation, splitting, model training, evaluation, and checkpoint saving |
| `test.ipynb`                 | Load a checkpoint and classify a single image                                               |
| `DATA_LIDC_IDRI/NOD/`        | 1,162 nodule JPEG images                                                                    |
| `DATA_LIDC_IDRI/NON_NOD/`    | 1,149 non-nodule JPEG images                                                                |


## Model configuration

| Parameter                   | Value               |
| --------------------------- | ------------------- |
| Input                       | RGB, 64 × 64 pixels |
| Patch size                  | 4 × 4 pixels        |
| Number of patches           | 256                 |
| Classes                     | 2                   |
| Embedding dimension         | 128                 |
| Transformer depth           | 8                   |
| Attention heads             | 8                   |
| MLP dimension               | 128                 |
| Dropout / embedding dropout | 0.1 / 0.1           |
| Loss                        | Cross-entropy       |
| Optimizer                   | Adam                |
| Learning rate               | 0.0001              |
| Batch size                  | 32                  |
| Epochs                      | 100                 |


## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/ShacMuSe/Lung-nodule-classification.git
cd Lung-nodule-classification
```

### 2. Create an environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install torch torchvision vit-pytorch scikit-learn pillow matplotlib seaborn jupyter
```

### 4. Configure local paths and hardware checks

Launch Jupyter from the repository root:

```bash
jupyter notebook
```

In `Train.ipynb`, replace the hard-coded Windows dataset path with:

```python
root_dir = 'DATA_LIDC_IDRI'
```

Keep that directory limited to the two class folders: the dataset loader assumes every entry in the root is a class directory containing images.

The notebook selects CUDA when available and otherwise uses CPU. However, its standalone `torch.cuda.get_device_name()` diagnostic is unconditional. On CPU-only machines, remove that cell or replace it with:

```python
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
```

### 5. Train and evaluate

Run the cells in `Train.ipynb` in order after applying the path and hardware adjustments. Training is configured for 100 epochs. The final cell saves the model state dictionary as:

```text
My_Ultimate_Model.pth
```

## Data preparation and splitting

Images are converted to RGB, resized to 64 × 64, converted to tensors, and normalized with mean and standard deviation of 0.5. The augmentation pipeline adds random rotations up to 15 degrees and random horizontal flips.

The notebook concatenates the original dataset with four stochastic augmented dataset views, then performs class-wise splits using `random_state=42`:

These augmented entries are views of the same underlying files, not independent images. This split is image/view-based and does not enforce patient separation.

## Single-image inference

Open `test.ipynb` and replace both hard-coded paths:

```python
model.load_state_dict(
    torch.load('My_Ultimate_Model.pth', map_location='cpu')
)
image_path = 'path/to/your/image.jpg'
```

Keep the model architecture identical to the training configuration. The notebook converts the image to RGB, applies the training resize and normalization, and computes softmax scores in evaluation mode.

| Predicted label | Meaning                |
| --------------- | ---------------------- |
| `0`             | `NON_NOD` — non-nodule |
| `1`             | `NOD` — nodule         |
