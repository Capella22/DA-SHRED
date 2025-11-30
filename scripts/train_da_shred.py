# scripts/train_da_shred.py
import argparse, numpy as np, torch
from models.shred import SHRED
from models.dashred import DAShred
from models.sindy_utils import build_library, stlsq

def finite_diff(arr, dt=1.0):
    return (arr[1:] - arr[:-1]) / dt

def main(args):
    model = SHRED(sensor_dim=args.p, latent_dim=args.latent_dim, enc_hidden=args.enc_hidden, dec_out_dim=args.n_full)
    model.load_state_dict(torch.load(args.shred_ckpt, map_location='cpu'))
    sensor_map = np.load(args.sensor_map)
    sensors = np.load(args.sensors)
    das = DAShred(model, sensor_map, window=args.window)
    fields = das.assimilate_sequence(sensors, steps=args.da_steps, lr=args.da_lr)
    print('Assimilated fields shape', fields.shape)
    if args.sindy_discover:
        sensors_rec = fields.dot(sensor_map.T)
        H = []
        for t in range(args.window, sensors_rec.shape[0]):
            H.append(sensors_rec[t-args.window:t])
        H = np.stack(H)
        with torch.no_grad():
            xb = torch.tensor(H, dtype=torch.float32)
            _, Z = model(xb)
        Z = Z.numpy()
        dZ = finite_diff(Z, dt=args.dt)
        Z_t = Z[:-1]
        Theta, names = build_library(Z_t, poly_order=args.poly_order, include_sine=args.include_sine, include_cos=args.include_cos)
        Xi = stlsq(Theta, dZ, lam=args.lambda_thr, max_iter=args.sindy_max_iter)
        for j in range(Xi.shape[1]):
            nz = np.where(np.abs(Xi[:, j]) > 0)[0]
            terms = [(names[i], float(Xi[i, j])) for i in nz]
            print(f"z{j}: {terms}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shred_ckpt', required=True)
    parser.add_argument('--sensor_map', required=True)
    parser.add_argument('--sensors', required=True)
    parser.add_argument('--p', type=int, required=True)
    parser.add_argument('--n_full', type=int, required=True)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--enc_hidden', type=int, default=128)
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--da_steps', type=int, default=20)
    parser.add_argument('--da_lr', type=float, default=1e-2)
    parser.add_argument('--poly_order', type=int, default=2)
    parser.add_argument('--include_sine', action='store_true')
    parser.add_argument('--include_cos', action='store_true')
    parser.add_argument('--lambda_thr', type=float, default=0.1)
    parser.add_argument('--sindy_max_iter', type=int, default=10)
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--sindy_discover', action='store_true')
    args = parser.parse_args()
    main(args)
