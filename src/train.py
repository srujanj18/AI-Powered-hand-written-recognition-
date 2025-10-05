import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from data_loader import load_image_paths_and_labels, download_iam_dataset
from preprocess import preprocess_image
from model import CRNN

class IAMDataset(Dataset):
    def __init__(self, image_label_pairs, img_height=32, img_width=128):
        self.pairs = image_label_pairs
        self.img_height = img_height
        self.img_width = img_width
        self.char_to_idx = {chr(i): i-96 for i in range(97, 123)}  # a-z: 1-26
        self.char_to_idx.update({str(i): i+27 for i in range(10)})  # 0-9: 27-36
        self.char_to_idx[' '] = 37  # space
        self.char_to_idx['<blank>'] = 0  # blank for CTC
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, label = self.pairs[idx]
        img = preprocess_image(img_path, self.img_height, self.img_width)
        img = torch.tensor(img).unsqueeze(0)  # (1, H, W)
        label_encoded = [self.char_to_idx.get(c, 0) for c in label.lower()]
        return img, torch.tensor(label_encoded), len(label_encoded)

def collate_fn(batch):
    imgs, labels, lengths = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.cat(labels)
    lengths = torch.tensor(lengths)
    return imgs, labels, lengths

def train_model(epochs=10, batch_size=32):
    data_dir = download_iam_dataset()
    pairs = load_image_paths_and_labels(data_dir)
    train_pairs = pairs[:int(0.8*len(pairs))]
    val_pairs = pairs[int(0.8*len(pairs)):]

    train_dataset = IAMDataset(train_pairs)
    val_dataset = IAMDataset(val_pairs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = CRNN()
    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for imgs, labels, lengths in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long)
            loss = criterion(outputs.log_softmax(2).permute(1, 0, 2), labels, input_lengths, lengths)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'models/crnn.pth')
    print("Model saved to models/crnn.pth")

if __name__ == '__main__':
    train_model()
