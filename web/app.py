from flask import Flask, request, jsonify, render_template
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model import CRNN
from preprocess import preprocess_image
from evaluate import decode_predictions
from train import IAMDataset
import tempfile

app = Flask(__name__)

# Load model
model = CRNN()
model_path = 'models/crnn.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully.")
else:
    print("Model not found. Please train the model first.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        file.save(tmp.name)
        img_path = tmp.name

    try:
        img = preprocess_image(img_path)
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        with torch.no_grad():
            output = model(img)
        pred_text = decode_predictions(output, IAMDataset([]).idx_to_char)[0]
        os.unlink(img_path)  # Clean up
        return jsonify({'text': pred_text})
    except Exception as e:
        os.unlink(img_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
