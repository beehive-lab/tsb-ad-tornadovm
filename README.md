# TSB-AD on GPUs with TornadoVM

GPU implementations, in Java with [TornadoVM](https://github.com/beehive-lab/TornadoVM), of the most
expensive parts of the [TSB-AD](https://github.com/TheDatumOrg/TSB-AD) time-series anomaly detection
benchmark:

| Component | TSB-AD counterpart | Device work |
|---|---|---|
| `Vus` | `get_metrics` VUS-ROC / VUS-PR (version `opt`, 250 thresholds) | extended labels for every buffer window, window x threshold accumulation |
| `Detectors.MatrixProfile` | `MatrixProfile` (stumpy self-join) | all-pairs z-normalised distances by diagonal (SCAMP) recurrences |
| `Detectors.SubPca` | `Sub_PCA` | window scoring against all principal directions |
| `Detectors.KMeansAd` | `KMeansAD_U` | k-means++ distances, Lloyd assignment and centroid sums |
| `Detectors.SubKnn` | `Sub_KNN` (method `largest`) | exact brute-force k nearest windows |

The detectors take TSB-AD's tuned univariate hyperparameters and reproduce TSB-AD's windowing,
normalisation, padding and scoring, so their scores can be compared value by value with the Python
originals. A PyTorch GPU version of the same pieces (CUDA or Apple MPS) is on the
[`gpu-acceleration` branch of beehive-lab/TSB-AD](https://github.com/beehive-lab/TSB-AD/tree/gpu-acceleration).

All kernels use TornadoVM's `KernelContext` API with explicit work-group sizes.

## Build and run

Needs JDK 21+, TornadoVM 8.1 (any backend: OpenCL, PTX, SPIR-V or Metal) and, for validation, Python with
the `TSB-AD` package.

```bash
./mvnw -q compile
java @"$TORNADOVM_HOME/tornado-argfile" -cp target/classes org.beehive.tsbad.TsbadBench \
  --dataset-dir TSB-AD-U --files 001_NAB_id_1_Facility_tr_1007_1st_2014.csv --out-dir out
```

`TsbadBench` reads TSB-AD CSV files, runs the detectors (`--detectors`, default all four), writes the scores
to `out/DET/FILE.f64` (little-endian float64) and one line per file and detector to `out/results.csv`:
first-run time (includes JIT compilation), best warm time (`--repeat`), VUS time and VUS-PR / VUS-ROC.
With `--ref-dir REF` it also scores `REF/DET/FILE.f64`, the output of another implementation, so the metric
can be checked on identical input.

The full comparison with TSB-AD:

```bash
PY=python3 scripts/bench.sh TSB-AD-U TSB-AD/Datasets/File_List/TSB-AD-U-Eva.csv results
```

runs the CPU detectors (`python/run_reference.py`), the TornadoVM ones, and `python/compare.py`, which
writes `results/summary.csv` and `results/summary_per_file.csv`.

## How it works

Every subsequence detector reads its windows straight from the series (window `i` is
`x[i .. i + m - 1]`), so no window matrix is stored on the device.

**VUS.** For buffer window `w` and threshold `thr`, the reference's loops over segments reduce to
`TP = #(anomalous points predicted) + S` and `N_labels = #(anomalous points) + S`, where
`S = sum over normal points of ext_w(t) * [score(t) >= thr]` and `ext_w` is the extended label. One kernel
builds `ext_w` for all windows over the points near anomalies; a tiled kernel accumulates `S` for the
whole (window x threshold) grid. Whether a point passes a threshold is decided on the host in double
precision (as an integer threshold rank), so ties behave exactly as in TSB-AD. The host computes the
existence term from per-segment score maxima (sparse table) and the areas.

**MatrixProfile.** A work-group owns 64 rows and sweeps all diagonals in bands of 64. Each thread walks one
diagonal, seeds the centred dot product directly and advances it with the SCAMP recurrence
`cov[i+1, j+1] = cov[i, j] + df[i+1] dg[j+1] + df[j+1] dg[i+1]`; correlations go to local memory, where each
thread reduces one row and keeps the best column. Two measures keep float32 accurate: the series and the
subsequence means are passed as (value, remainder) float pairs for the seeds, and a running bound on the
recurrence's rounding error triggers a fresh seed (flat subsequences after large jumps). The host
recomputes the distance to the selected match in double precision and applies stumpy's rules for constant
subsequences.

**Sub-PCA.** The covariance of the standardised window matrix is built on the host in double precision in
O(N m) using its Hankel structure (`G[a][b] = G[a-1][b-1] - x[a-1] x[b-1] + x[a-1+N] x[b-1+N]`), followed
by a symmetric eigendecomposition. Double precision matters here: with all components kept, the smallest
eigenvalues (down to 1e-10 of the largest) carry the largest weights. The device scores every window
against all components.

**KMeansAD.** k-means++ seeding with local trials and Lloyd iterations as in scikit-learn (tolerance
1e-4 of the mean variance); the device computes candidate distances, assignments and per-chunk centroid
sums. The random seeding differs from NumPy's, so results agree statistically, not value by value.

**Sub-KNN.** A work-group owns 32 query windows; candidate windows come in tiles whose series stretch sits in
local memory with the queries' stretch. Each thread keeps its sorted k best (distance and index) in local
memory and abandons a candidate once its partial sum passes the current k-th. The host recomputes the last
few of the k nearest in double precision. Windows up to 320 points and k up to 100.

## Validation and benchmark

RESULTS

## Limitations

- Univariate series only (TSB-AD-U). The PyTorch branch also covers the multivariate detectors.
- Sub-KNN computes exact neighbours by brute force; its cost grows with N^2 m.
- KMeansAD's random seeding differs from scikit-learn's.

## License

MIT. TSB-AD is Apache-2.0; this project reimplements its algorithms and does not include its code.
