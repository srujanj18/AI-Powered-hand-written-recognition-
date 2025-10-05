import torch
import editdistance
from model import CRNN
from preprocess import preprocess_image
from train import IAMDataset

def decode_predictions(outputs, idx_to_char):
    """
    Decode CTC outputs to text.
    """
    preds = outputs.argmax(2).cpu().numpy()
    decoded = []
    for pred in preds:
        text = []
        prev = -1
        for p in pred:
            if p != 0 and p != prev:
                text.append(idx_to_char.get(p, ''))
            prev = p
        decoded.append(''.join(text))
    return decoded

def calculate_cer_wer(preds, targets):
    """
    Calculate Character Error Rate (CER) and Word Error Rate (WER).
    """
    total_cer = 0
    total_wer = 0
    count = len(preds)
    for pred, target in zip(preds, targets):
        cer = editdistance.eval(pred, target) / max(len(target), 1)
        wer = editdistance.eval(pred.split(), target.split()) / max(len(target.split()), 1)
        total_cer += cer
        total_wer += wer
    return total_cer / count, total_wer / count

def evaluate_model(model_path='models/crnn.pth', data_dir='data/iam'):
    model = CRNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    pairs = IAMDataset(load_image_paths_and_labels(data_dir)).pairs[:100]  # Evaluate on first 100 for speed
    preds = []
    targets = []
    for img_path, label in pairs:
        img = preprocess_image(img_path)
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        with torch.no_grad():
            output = model(img)
        pred_text = decode_predictions(output, IAMDataset([]).idx_to_char)[0]
        preds.append(pred_text)
        targets.append(label)

    cer, wer = calculate_cer_wer(preds, targets)
    print(f"CER: {cer:.4f}, WER: {wer:.4f}")
    return cer, wer

if __name__ == '__main__':
    evaluate_model()
