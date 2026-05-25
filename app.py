from flask import Flask, request, jsonify, send_file
import torch
import numpy as np
import torch.nn as nn

class PriceANN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x)

ckpt = torch.load('uso_ann_model.pth', map_location='cpu', weights_only=False)
model = PriceANN(ckpt['input_dim'])
model.load_state_dict(ckpt['model_state'])
model.eval()

s_mean  = np.array(ckpt['scaler_mean'])
s_scale = np.array(ckpt['scaler_scale'])

app = Flask(__name__)

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    d = request.json
    f = np.array([[
        float(d['high']), float(d['low']), float(d['open']), float(d['volume']),
        float(d['sp_open']), float(d['sp_close']), float(d['sp_high']),
        float(d['dj_open']), float(d['dj_close']),
        float(d['eg_close']), float(d['eu_price']),
        float(d['of_price']), float(d['os_price']),
        float(d['sf_price']), float(d['plt_price']), float(d['pld_price']),
        float(d['rho_price']), float(d['usdi_price']),
        float(d['gdx_close']), float(d['gdx_volume']),
        float(d['uso_open']), float(d['uso_high']), float(d['uso_low']), float(d['uso_volume'])
    ]])

    f = (f - s_mean) / s_scale

    with torch.no_grad():
        pred = model(torch.FloatTensor(f)).item()

    return jsonify({
        'predicted_uso_close': round(pred, 2),
        'model_r2': round(ckpt['r2_score'], 4),
        'model_rmse': round(ckpt['rmse'], 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
