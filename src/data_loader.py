import os
import kagglehub

def download_iam_dataset(data_dir='data/iam'):
    """
    Download the IAM Handwriting Database using KaggleHub.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Download the dataset
    path = kagglehub.dataset_download("naderabdalghani/iam-handwritten-forms-dataset")
    print("Path to dataset files:", path)
    # Assume the dataset is extracted to path, and we use that as data_dir
    return path

def load_image_paths_and_labels(data_dir='data/iam'):
    """
    Load image file paths and corresponding ground truth text labels.
    This function parses the IAM dataset ascii files to get line images and labels.
    """
    ascii_dir = os.path.join(data_dir, 'ascii')
    lines_file = os.path.join(ascii_dir, 'lines.txt')
    if not os.path.exists(lines_file):
        raise FileNotFoundError(f"Lines file not found: {lines_file}. "
                                "Please ensure IAM dataset is downloaded and extracted properly.")
    image_label_pairs = []
    with open(lines_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split(' ')
            if len(parts) < 9:
                continue
            image_id = parts[0]
            transcription = ' '.join(parts[8:])
            image_path = os.path.join(data_dir, 'lines', image_id + '.png')
            if os.path.exists(image_path):
                image_label_pairs.append((image_path, transcription))
    return image_label_pairs
