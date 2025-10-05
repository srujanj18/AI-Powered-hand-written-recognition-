jt# AI-Powered Handwriting Recognition

This project implements an AI-powered handwriting recognition system using the IAM Handwriting Database. It uses a CRNN (CNN + RNN) model to convert handwritten images into digital text.

## Features

- Dataset: IAM Handwriting Database
- Preprocessing: Grayscale conversion, normalization, resizing with padding
- Model: CRNN architecture with CTC loss
- Evaluation: Character Error Rate (CER) and Word Error Rate (WER)
- Deployment: Flask web app for uploading handwritten images and getting text output

## Setup

1. Clone the repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. The IAM Handwriting Database will be downloaded automatically using KaggleHub when running the training script.

## Usage

### Training

Run the training script:
```
python src/train.py
```

### Evaluation

Run the evaluation script:
```
python src/evaluate.py
```

### Web App

Start the Flask web app:
```
python web/app.py
```

Open your browser at `http://127.0.0.1:5000` to upload handwritten images and get text output.

## Notes

- Training may take significant time depending on your hardware.
- The IAM dataset requires manual download due to licensing.

## License

MIT License
