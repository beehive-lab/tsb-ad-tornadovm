"""Compares the TornadoVM results with the CPU reference.

    python compare.py --ref REF --tvm OUT [--summary summary.csv]

Per detector: agreement of the scores (Pearson correlation of the min-max normalised scores), VUS-PR /
VUS-ROC of each implementation, the VUS metric error (TornadoVM VUS of the reference scores against
TSB-AD's own VUS of the same scores) and run-time speedups.
"""
import argparse, os

import numpy as np
import pandas as pd


def norm(a):
    r = a.max() - a.min()
    return (a - a.min()) / r if r > 0 else np.zeros_like(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--tvm', required=True)
    ap.add_argument('--summary')
    args = ap.parse_args()
    ref = pd.read_csv(os.path.join(args.ref, 'results.csv')).drop_duplicates(['file', 'detector'], keep='last')
    tvm = pd.read_csv(os.path.join(args.tvm, 'results.csv')).drop_duplicates(['file', 'detector'], keep='last')
    m = ref.merge(tvm, on=['file', 'detector'], suffixes=('_cpu', '_tvm'))
    corr = []
    for _, r in m.iterrows():
        stem = os.path.splitext(r.file)[0]
        a = np.fromfile(os.path.join(args.ref, r.detector, stem + '.f64'), '<f8')
        b = np.fromfile(os.path.join(args.tvm, r.detector, stem + '.f64'), '<f8')
        a, b = norm(a), norm(b)
        corr.append(np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan)
    m['corr'] = corr
    m['vus_metric_err'] = np.maximum((m.ref_vus_pr - m.vus_pr_cpu).abs(), (m.ref_vus_roc - m.vus_roc_cpu).abs())
    # VUS of both implementations' scores, both by the TornadoVM metric (shown exact by vus_metric_err)
    m['vus_pr_diff'] = m.vus_pr_tvm - m.ref_vus_pr
    m['vus_pr_diff_r6'] = m.vus_pr_r6 - m.ref_vus_pr_r6
    m['speedup_warm'] = m.cpu_ms / m.warm_ms
    m['speedup_first'] = m.cpu_ms / m.first_ms
    rows = []
    for det, g in m.groupby('detector'):
        rows.append(dict(detector=det, files=len(g), corr_min=g['corr'].min(), corr_median=g['corr'].median(),
                         vus_pr_cpu_mean=g.ref_vus_pr.mean(), vus_pr_tvm_mean=g.vus_pr_tvm.mean(),
                         vus_pr_absdiff_mean=g.vus_pr_diff.abs().mean(), vus_pr_absdiff_max=g.vus_pr_diff.abs().max(),
                         files_absdiff_gt_0_01=int((g.vus_pr_diff.abs() > 0.01).sum()),
                         vus_pr_r6_absdiff_max=g.vus_pr_diff_r6.abs().max(),
                         vus_checked=int(g.vus_metric_err.notna().sum()), vus_metric_err_max=g.vus_metric_err.max(),
                         cpu_s_total=g.cpu_ms.sum() / 1e3, tvm_warm_s_total=g.warm_ms.sum() / 1e3,
                         tvm_first_s_total=g.first_ms.sum() / 1e3,
                         speedup_total_warm=g.cpu_ms.sum() / g.warm_ms.sum(),
                         speedup_total_first=g.cpu_ms.sum() / g.first_ms.sum(),
                         vus_speedup_total=g.vus_ms_cpu.sum() / g.vus_ms_tvm[g.vus_ms_cpu.notna()].sum()))
    s = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print(s.to_string(index=False, float_format=lambda v: f'{v:.6g}'))
    if args.summary:
        s.to_csv(args.summary, index=False)
        m.to_csv(os.path.splitext(args.summary)[0] + '_per_file.csv', index=False)


if __name__ == '__main__':
    main()
