import torch, numpy as np, pandas as pd
import torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Chargement
df = pd.read_csv('data/pricing_dataset.csv')
X, y = df[['S0','K','sigma','T','r']].values, df['price'].values.reshape(-1,1)

# 2. Normalisation
sc_X, sc_y = StandardScaler(), StandardScaler()
X, y = sc_X.fit_transform(X), sc_y.fit_transform(y)

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Modèle
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5,64), nn.ReLU(),
            nn.Linear(64,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self,x): return self.net(x)

model, loss_fn = MLP(), nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

# 4. Entraînement
data_tr = torch.utils.data.TensorDataset(
    torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float()
)
loader = torch.utils.data.DataLoader(data_tr, batch_size=1024, shuffle=True)

for epoch in range(1,51):
    model.train()
    l=0
    for xb,yb in loader:
        pred=model(xb); loss=loss_fn(pred,yb)
        opt.zero_grad(); loss.backward(); opt.step()
        l+=loss.item()*len(xb)
    if epoch%10==0:
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(torch.from_numpy(X_va).float()),
                        torch.from_numpy(y_va).float()).item()
        print(f"Epoch {epoch} – train={l/len(loader.dataset):.4f} – val={v:.4f}")

# 5. Sauvegarde
# 1) save model weights only
torch.save(model.state_dict(), 'models/mlp_weights.pt')
# 2) save scalers separately
joblib.dump(sc_X, 'models/scaler_X.pkl')
joblib.dump(sc_y, 'models/scaler_y.pkl')

print("Weights saved -> models/mlp_weights.pt")
print("Scalers saved  -> models/scaler_X.pkl, models/scaler_y.pkl")
