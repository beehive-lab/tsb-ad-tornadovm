"""Runs TSB-AD's CPU detectors on TSB-AD files and writes their scores and VUS metrics.

    python run_reference.py --dataset_dir DIR --file_list LIST.csv --out_dir REF [--detectors ...] [--repeat 1]

Needs the TSB-AD package (pip install TSB-AD, or an editable checkout). Writes REF/DET/STEM.f64 (raw
scores, little-endian float64) and appends to REF/results.csv: run time, and VUS-PR / VUS-ROC of the
min-max normalised scores computed by TSB-AD's get_metrics code (version 'opt'), with its run time.
"""
import argparse, contextlib, io, os, random, time, warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
from TSB_AD.evaluation.basic_metrics import generate_curve
from TSB_AD.model_wrapper import run_Unsupervise_AD
from TSB_AD.utils.slidingWindows import find_length_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--file_list')
    ap.add_argument('--files', nargs='*')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--detectors', nargs='+', default=['Sub_PCA', 'Sub_KNN', 'KMeansAD_U', 'MatrixProfile'])
    ap.add_argument('--repeat', type=int, default=1, help='runs per detector; the fastest is reported')
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--vus_every', type=int, default=1, help="run TSB-AD's (slow) VUS on every n-th file only")
    args = ap.parse_args()
    files = args.files or pd.read_csv(args.file_list)['file_name'].tolist()
    os.makedirs(args.out_dir, exist_ok=True)
    csv = os.path.join(args.out_dir, 'results.csv')
    done = set()
    if os.path.exists(csv):
        prev = pd.read_csv(csv)
        done = set(zip(prev['file'], prev['detector']))
    for fi, f in enumerate(files):
        df = pd.read_csv(os.path.join(args.dataset_dir, f)).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        sw = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        stem = os.path.splitext(f)[0]
        for det in args.detectors:
            if (f, det) in done:
                continue
            best = float('inf')
            for _ in range(args.repeat):
                np.random.seed(args.seed)
                random.seed(args.seed)
                t = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    out = run_Unsupervise_AD(det, data, **Optimal_Uni_algo_HP_dict[det])
                best = min(best, time.perf_counter() - t)
            if isinstance(out, str):
                print(f, det, 'failed:', out)
                continue
            os.makedirs(os.path.join(args.out_dir, det), exist_ok=True)
            np.asarray(out, dtype='<f8').tofile(os.path.join(args.out_dir, det, stem + '.f64'))
            row = dict(file=f, detector=det, n=len(label), eval_window=sw, cpu_ms=best * 1e3,
                       vus_ms=np.nan, vus_roc=np.nan, vus_pr=np.nan)
            if fi % args.vus_every == 0:
                s = MinMaxScaler().fit_transform(np.asarray(out, float).reshape(-1, 1)).ravel()
                t = time.perf_counter()
                r = generate_curve(label, s, sw, 'opt', 250)
                row.update(vus_ms=(time.perf_counter() - t) * 1e3, vus_roc=r[6], vus_pr=r[7])
            pd.DataFrame([row]).to_csv(csv, mode='a', header=not os.path.exists(csv), index=False)
            print(f"[ref] {f} {det} n={len(label)} cpu {best:.2f}s VUS-PR {row['vus_pr']:.6f}", flush=True)


if __name__ == '__main__':
    main()
