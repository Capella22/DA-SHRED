# scripts/train_shred_with_sindy.py
import argparse, numpy as np, torch
from torch import optim
from models.shred import SHRED
from models.sindy_utils import build_library, stlsq
from tqdm import tqdm

def finite_diff(arr, dt=1.0):
    return (arr[1:] - arr[:-1]) / dt

def train(args):
    sim = np.load(args.sim)   # (T, n_full)
    sensor_map = np.load(args.sensor_map)  # (p, n_full)
    sensors = sim.dot(sensor_map.T)       # (T, p)
    window = args.window
    X_hist = []
    Y = []
    for t in range(window, sensors.shape[0]):
        X_hist.append(sensors[t-window:t])
        Y.append(sim[t])
    X_hist = np.stack(X_hist)
    Y = np.stack(Y)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SHRED(sensor_dim=X_hist.shape[2], latent_dim=args.latent_dim, enc_hidden=args.enc_hidden, dec_out_dim=Y.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for i in range(0, X_hist.shape[0], args.batch):
            xb = torch.tensor(X_hist[i:i+args.batch], dtype=torch.float32).to(device)
            yb = torch.tensor(Y[i:i+args.batch], dtype=torch.float32).to(device)
            pred, z = model(xb)
            loss_rec = mse(pred, yb)
            loss = loss_rec
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss_rec.item() * xb.size(0)
        # SINDy regularization and reporting every sindy_freq epochs
        if args.sindy_reg > 0 and (epoch+1) % args.sindy_reg_freq == 0:
            model.eval()
            with torch.no_grad():
                xb_full = torch.tensor(X_hist, dtype=torch.float32).to(device)
                _, Z = model(xb_full)
                Z_np = Z.cpu().numpy()
            if Z_np.shape[0] > 2:
                dzdt = finite_diff(Z_np, dt=args.dt)
                Z_t = Z_np[:-1]
                Theta, names = build_library(Z_t, poly_order=args.poly_order, include_sine=args.include_sine, include_cos=args.include_cos)
                # STLSQ scheduling: coarse->fine thresholds
                schedule = [args.sindy_lambda*2.0, args.sindy_lambda, args.sindy_lambda*0.5]
                Xi = stlsq(Theta, dzdt, lam=args.sindy_lambda, max_iter=args.sindy_max_iter, thresh_schedule=schedule, verbose=False)
                # print a summary
                for j in range(Xi.shape[1]):
                    nz = np.where(np.abs(Xi[:, j]) > 0)[0]
                    terms = [(names[i], float(Xi[i, j])) for i in nz]
                    print(f"Epoch {epoch+1} z{j}: {terms}")
        print(f"Epoch {epoch+1}/{args.epochs} loss {epoch_loss / X_hist.shape[0]:.6f}")
    torch.save(model.state_dict(), args.ckpt)
    print('Saved', args.ckpt)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', required=True)
    parser.add_argument('--sensor_map', required=True)
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--enc_hidden', type=int, default=128)
    parser.add_argument('--dec_out_dim', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ckpt', default='shred_ckpt.pth')
    parser.add_argument('--sindy_reg', type=float, default=0.0)
    parser.add_argument('--sindy_lambda', type=float, default=0.1)
    parser.add_argument('--sindy_max_iter', type=int, default=10)
    parser.add_argument('--sindy_reg_freq', type=int, default=1)
    parser.add_argument('--poly_order', type=int, default=2)
    parser.add_argument('--include_sine', action='store_true')
    parser.add_argument('--include_cos', action='store_true')
    parser.add_argument('--dt', type=float, default=1.0)
    args = parser.parse_args()
    train(args)
