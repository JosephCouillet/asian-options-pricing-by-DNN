import torch
import torch.nn as nn

# Model only (no heavy code at import)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5,64), nn.ReLU(),
            nn.Linear(64,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self, x):
        return self.net(x)

# Optional: keep the original training here but under a guard
def train_and_save():
    import numpy as np, pandas as pd, joblib
    import torch.optim as optim
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # loading
    df = pd.read_csv('data/pricing_dataset.csv')
    X = df[['S0','K','sigma','T','r']].values
    y = df['price'].values.reshape(-1,1)

    # normalization (fits saved to disk)
    sc_X, sc_y = StandardScaler(), StandardScaler()
    X, y = sc_X.fit_transform(X), sc_y.fit_transform(y)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)

    # model + loss + opt
    model, loss_fn = MLP(), nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    # dataloader
    data_tr = torch.utils.data.TensorDataset(
        torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float()
    )
    loader = torch.utils.data.DataLoader(data_tr, batch_size=1024, shuffle=True)

    # training loop (identique à ton code)
    for epoch in range(1,51):
        model.train()
        total = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                v = loss_fn(
                    model(torch.from_numpy(X_va).float()),
                    torch.from_numpy(y_va).float()
                ).item()
            print(f"Epoch {epoch} – train={total/len(loader.dataset):.4f} – val={v:.4f}")

    # save artifacts (same paths)
    torch.save(model.state_dict(), 'models/mlp_weights.pt')
    joblib.dump(sc_X, 'models/scaler_X.pkl')
    joblib.dump(sc_y, 'models/scaler_y.pkl')
    print("Weights saved -> models/mlp_weights.pt")
    print("Scalers saved  -> models/scaler_X.pkl, models/scaler_y.pkl")

if __name__ == "__main__":
    train_and_save()
