from torch import nn
import torch
import torchaudio

SAMPLE_RATE = 48000
EMOTIONS_MAP = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised",
}


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size = x.size(0)
        sequence_length = x.size(1)

        # reshaped_x shape: (batch_size * sequence_length, channels, height, width)
        reshaped_x = x.view(batch_size * sequence_length, *x.size()[2:])

        out = self.module(reshaped_x)

        # out_shape: (batch_size, sequence_length, new_channels, new_height, new_width)
        out = out.view(batch_size, sequence_length, *out.size()[1:])

        return out


class EmotionalModel(nn.Module):
    def __init__(self, num_emotions: int):
        super().__init__()
        self.cnn_extractor = nn.Sequential(
            TimeDistributed(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)),
            TimeDistributed(nn.BatchNorm2d(16)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(kernel_size=2, stride=2)),
            TimeDistributed(nn.Dropout(p=0.3)),
            TimeDistributed(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
            TimeDistributed(nn.BatchNorm2d(32)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(kernel_size=4, stride=4)),
            TimeDistributed(nn.Dropout(p=0.3)),
            TimeDistributed(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            TimeDistributed(nn.BatchNorm2d(64)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(kernel_size=4, stride=4)),
            TimeDistributed(nn.Dropout(p=0.3)),
        )
        self.feature_proj = nn.Linear(1024, 256)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            bidirectional=True,
            batch_first=True,
        )

        self.lstm_dropout = nn.Dropout(p=0.4)

        self.attention_linear = nn.Linear(2 * 128, 1)
        self.output_linear = nn.Linear(2 * 128, num_emotions)

    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, sequence_length, channels, height, width)

        # 1. CNN Feature Extraction
        cnn_features = self.cnn_extractor(x)

        # Reshape for LSTM
        batch_size, seq_len, _, _, _ = cnn_features.size()
        lstm_input = cnn_features.view(batch_size, seq_len, -1)
        lstm_input = self.feature_proj(lstm_input)

        # 2. LSTM Processing
        lstm_output, _ = self.lstm(lstm_input)
        lstm_output = self.lstm_dropout(lstm_output)
        # lstm_output shape: (batch_size, seq_len, hidden_size * 2)

        # 3. Attention Mechanism
        attention_weights = self.attention_linear(
            lstm_output
        )  # (batch_size, seq_len, 1)
        attention_weights = nn.functional.softmax(attention_weights, dim=1)

        # Apply attention weights
        context_vector = torch.bmm(
            attention_weights.permute(0, 2, 1), lstm_output  # (batch_size, 1, seq_len)
        )  # (batch_size, 1, hidden_size * 2)
        context_vector = context_vector.squeeze(1)  # (batch_size, hidden_size * 2)

        # 4. Final Classification
        logits = self.output_linear(context_vector)

        # Softmax
        probabilities = nn.functional.softmax(logits, dim=1)

        return logits, probabilities


def load_raw_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)

    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=SAMPLE_RATE
        )
        waveform = resampler(waveform)
    return waveform


def format_audio(waveform):
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = waveform.squeeze(0)

    start_sample = int(0.5 * SAMPLE_RATE)
    waveform = waveform[start_sample:]

    # PAD/TRIM
    max_len = SAMPLE_RATE * 3
    waveform = waveform[:max_len]

    if waveform.size(0) < max_len:
        pad = torch.zeros(max_len)
        pad[: waveform.size(0)] = waveform
        waveform = pad
    return waveform


def get_mel_chunks(waveform):
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=128,
        n_fft=1024,
        hop_length=256,
        win_length=512,
        window_fn=torch.hamming_window,
        f_max=SAMPLE_RATE / 2,
    )(waveform).squeeze(0)
    mel_spec = torchaudio.functional.amplitude_to_DB(
        mel_spec, amin=1e-10, multiplier=10.0, db_multiplier=0.0
    )
    stride = 64
    win_size = 128
    chunks = []

    for i in range(mel_spec.size(1) // stride):
        start_idx = i * stride
        end_idx = start_idx + win_size
        if end_idx > mel_spec.size(1):
            break
        chunk = mel_spec[:, start_idx:end_idx]
        if chunk.size(1) == win_size:
            chunks.append(chunk)

    return torch.stack(chunks).unsqueeze(1)


def _prepare_data(file_path: str):
    waveform = load_raw_audio(file_path)
    waveform = format_audio(waveform)
    mel_chunks = get_mel_chunks(waveform)
    input_tensor = mel_chunks.to(torch.float32)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_tensor


def get_emotion_prediction(model: nn.Module, filename: str):
    input_tensor = _prepare_data(filename)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        _, probabilities = model(input_tensor)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return EMOTIONS_MAP[predicted_class]
