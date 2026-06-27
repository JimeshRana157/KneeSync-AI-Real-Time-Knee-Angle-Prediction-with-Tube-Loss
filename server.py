import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----- Model definition (DIC) -----
class ChannelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(inplace=True),
            nn.AvgPool1d(4),
        )
    def forward(self, x): return self.net(x)

class AttentionFusion(nn.Module):
    def __init__(self, emg_feat_dim=32, kin_hidden=64):
        super().__init__()
        self.emg_proj = nn.Linear(emg_feat_dim, kin_hidden)
        self.attn_fc  = nn.Linear(kin_hidden * 2, 1)
    def forward(self, kin_hidden, emg_feats):
        attn_weights, emg_proj_list = [], []
        for feat in emg_feats:
            f = feat.permute(0,2,1)
            f_proj = self.emg_proj(f)
            emg_proj_list.append(f_proj)
            score = self.attn_fc(torch.cat([kin_hidden, f_proj], -1)).squeeze(-1)
            attn_weights.append(torch.softmax(score, -1))
        emg_stacked  = torch.stack([f.permute(0,2,1) for f in emg_feats], 1)
        attn_stacked = torch.stack(attn_weights, 1)
        emg_weighted = (emg_stacked * attn_stacked.unsqueeze(-1)).sum(1)
        return torch.cat([kin_hidden, emg_weighted], -1)

class CNNLSTM_DIC_Tube(nn.Module):
    def __init__(self, predict_steps=1, dropout=0.3):
        super().__init__()
        self.emg_cnns = nn.ModuleList([ChannelCNN() for _ in range(4)])
        self.kin_lstm = nn.LSTM(4, 64, batch_first=True)
        self.attn_fusion = AttentionFusion()
        self.lstm1 = nn.LSTM(64+32, 64, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.head  = nn.Linear(32, 2 * predict_steps)
    def forward(self, emg, knee_hist):
        emg_feats = [cnn(emg[:, i:i+1, :]) for i, cnn in enumerate(self.emg_cnns)]
        knee_in   = knee_hist.view(-1, 50, 4)
        kin_out, _ = self.kin_lstm(knee_in)
        fused      = self.attn_fusion(kin_out, emg_feats)
        out, _     = self.lstm1(fused)
        out        = self.drop(out)
        out, _     = self.lstm2(out)
        return self.head(out[:, -1, :])
    def get_bounds(self, emg, knee_hist):
        out = self.forward(emg, knee_hist)
        mu1 = torch.min(out[:,0:1], out[:,1:2])
        mu2 = torch.max(out[:,0:1], out[:,1:2])
        return mu1, mu2

# ----- Scaler -----
class KneeScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = max(std, 1e-6)
    def transform(self, y): return (np.asarray(y, np.float64) - self.mean) / self.std
    def inverse(self, y): return np.asarray(y, np.float64) * self.std + self.mean

# ----- Load model -----
device = torch.device('cpu')
model = CNNLSTM_DIC_Tube(predict_steps=1, dropout=0.3).to(device)
checkpoint = torch.load('s2_dic_abnormal_loo10.pt', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint)
model.eval()

# ----- Scalers (from UCI abnormal data - approximate, replace if you have exact) -----
SCALER_MEAN = 14.2   # degrees (update after running helper)
SCALER_STD  = 8.7    # degrees
scaler = KneeScaler(SCALER_MEAN, SCALER_STD)

# ----- FastAPI -----
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class DICInput(BaseModel):
    emg: list[list[float]]
    knee_hist: list[float]

@app.post("/predict")
def predict(data: DICInput):
    emg_np   = np.array(data.emg, dtype=np.float32)
    knee_np  = np.array(data.knee_hist, dtype=np.float32)

    # per-window EMG normalisation
    emg_mean = emg_np.mean(1, keepdims=True)
    emg_std  = emg_np.std(1, keepdims=True) + 1e-8
    emg_np   = (emg_np - emg_mean) / emg_std

    x_emg  = torch.from_numpy(emg_np).unsqueeze(0).to(device)
    x_knee = torch.from_numpy(knee_np).unsqueeze(0).to(device)

    with torch.no_grad():
        mu1, mu2 = model.get_bounds(x_emg, x_knee)

    mu1_deg = scaler.inverse(mu1.cpu().numpy()).item()
    mu2_deg = scaler.inverse(mu2.cpu().numpy()).item()

    return {
        "mu1": round(mu1_deg, 2),
        "mu2": round(mu2_deg, 2),
        "midpoint": round((mu1_deg + mu2_deg) / 2, 2)
    }
