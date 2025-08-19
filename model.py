import torch
import torch.nn as nn

class SpatialEncoder(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        residual = out
        out = self.relu(self.bn2(self.conv2(out)))
        out = out + residual
        out = self.relu(self.bn3(self.conv3(out)))
        return out

class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim, nhead=2, num_layers=2, dropout=0.1, max_seq_len=100):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.transformer(x)
        return x

class MuSHiNet(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=4, seq_len=12, H=27, W=51):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(in_channels, hidden_channels)
        self.embed_dim = hidden_channels * H * W
        self.temporal_transformer = TemporalTransformer(embed_dim=self.embed_dim)
        self.mean_head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, H*W)
        )
        self.var_head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, H*W),
            nn.Softplus()
        )
        self.H, self.W = H, W

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        spatial_features = []
        for t in range(seq_len):
            out = self.spatial_encoder(x[:, t])
            out = out.view(batch_size, -1)
            spatial_features.append(out)
        seq_features = torch.stack(spatial_features, dim=1)
        temporal_features = self.temporal_transformer(seq_features)
        last_step = temporal_features[:, -1, :]
        mean = self.mean_head(last_step).view(batch_size, 1, H, W)
        var = self.var_head(last_step).view(batch_size, 1, H, W)
        return mean, var
