import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_height=32, num_classes=37):  # 26 letters + 10 digits + space + blank
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.ReLU()
        )
        self.rnn = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True, batch_first=True),
            nn.LSTM(256*2, 256, bidirectional=True, batch_first=True)
        )
        self.fc = nn.Linear(256*2, num_classes)

    def forward(self, x):
        # x: (batch, 1, H, W)
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.view(b, c, -1).permute(0, 2, 1)  # (b, w, c)
        rnn_out, _ = self.rnn(conv)
        out = self.fc(rnn_out)
        return out  # (b, seq_len, num_classes)
