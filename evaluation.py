import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_model import MLP
import joblib

# Load dataset
df = pd.read_csv('data/pricing_dataset.csv')
X = df[['S0','K','sigma','T','r']].values
y_true = df['price'].values

# Load trained model and scalers (no training here)
state = torch.load('models/mlp_weights.pt', map_location='cpu')
model = MLP()
model.load_state_dict(state)
model.eval()

sc_X = joblib.load('models/scaler_X.pkl')
sc_y = joblib.load('models/scaler_y.pkl')

# Predict
X_scaled = sc_X.transform(X)
with torch.no_grad():
    y_pred_scaled = model(torch.from_numpy(X_scaled).float()).detach().numpy()
y_pred = sc_y.inverse_transform(y_pred_scaled).ravel()

# 1. Scatter plot MC vs ML
plt.figure()
plt.scatter(y_true, y_pred, s=2)
m, M = float(np.min(y_true)), float(np.max(y_true))
plt.plot([m, M], [m, M], 'k--')
plt.xlabel('MC price'); plt.ylabel('ML price')
plt.title('MC vs ML Pricing')
plt.show()

# 2. Error histogram
errors = y_pred - y_true
plt.figure()
plt.hist(errors, bins=100)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Error (ML – MC)'); plt.ylabel('Count')
plt.show()

# 3. Heatmap of mean error over (K, T)
kbins, tbins = 50, 50
H, xedges, yedges = np.histogram2d(
    df['K'], df['T'], bins=[kbins, tbins], weights=errors, density=False
)
counts, _, _ = np.histogram2d(df['K'], df['T'], bins=[kbins, tbins])
mean_err = H / np.maximum(counts, 1)

plt.figure()
plt.imshow(
    mean_err.T, origin='lower',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    aspect='auto'
)
plt.colorbar(label='Mean Error')
plt.xlabel('Strike K'); plt.ylabel('Maturity T')
plt.title('Heatmap of ML – MC Error')
plt.show()

