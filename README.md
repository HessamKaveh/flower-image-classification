# Flower Image Classification (Transfer Learning with ResNet18)

Fine-tuned ResNet18 (pretrained on ImageNet) to classify 5 flower species:
daisy, dandelion, roses, sunflowers, tulips.

## Pipeline
1. Load flower dataset (80/20 train/val split)
2. Data augmentation (random crop, horizontal flip)
3. Transfer learning: freeze ResNet18 backbone, fine-tune final FC layer
4. Train for 10 epochs, save best checkpoint by validation accuracy
5. Evaluate: confusion matrix, per-class precision/recall/F1

## Dataset
[TensorFlow flower_photos dataset](http://download.tensorflow.org/example_images/flower_photos.tgz)
(not included in repo due to size — download and extract into `data/`)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
cd src
python train.py
python evaluate.py
```

## Results
See `results/training_curves.png` and `results/confusion_matrix.png`

## Author
Hessam Kaveh — Research Fellow, Italian Institute of Technology
