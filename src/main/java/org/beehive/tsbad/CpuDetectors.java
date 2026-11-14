/*
 * Copyright (c) 2026 Beehive lab
 * SPDX-License-Identifier: MIT
 */
package org.beehive.tsbad;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * CPU versions of the {@link Detectors}, in double precision, for when the device fails: the same definitions
 * (TSB-AD's windowing, normalisation, scoring and padding), with the device kernels replaced by loops over all
 * cores. They are a reference and a fallback when the device fails, not a fast path: Sub-KNN and MatrixProfile
 * do O(N^2 m) and O(N^2) work. They use the detectors' host code (period estimate, padding, eigendecomposition)
 * unchanged.
 */
public final class CpuDetectors {

    private CpuDetectors() {
    }

    /** Sub-PCA, as {@link Detectors.SubPca}: all components, weighted by the explained variance. */
    public static double[] subPca(double[] x, int m) {
        int n = x.length;
        int N = n - m + 1;
        double shift = Detectors.mean(x);
        double[] xd = new double[n];
        for (int i = 0; i < n; i++) {
            xd[i] = x[i] - shift;
        }
        double[] cmu = new double[m];
        double[] csd = new double[m];
        double[][] G = new double[m][m];
        IntStream.range(0, m).parallel().forEach(b -> {
            double s = 0;
            double g0 = 0;
            for (int i = 0; i < N; i++) {
                s += xd[i + b];
                g0 += xd[i] * xd[i + b];
            }
            cmu[b] = s / N;
            double q = 0;
            for (int i = 0; i < N; i++) {
                double d = xd[i + b] - cmu[b];
                q += d * d;
            }
            csd[b] = Math.sqrt(q / N);
            G[0][b] = g0;
        });
        for (int a = 1; a < m; a++) {
            for (int b = a; b < m; b++) {
                G[a][b] = G[a - 1][b - 1] - xd[a - 1] * xd[b - 1] + xd[a - 1 + N] * xd[b - 1 + N];
            }
        }
        int[] keep = new int[m];
        int mk = 0;
        for (int a = 0; a < m; a++) {
            if (csd[a] > 0) {
                keep[mk++] = a;
            }
        }
        double[][] C = new double[mk][mk];
        for (int p = 0; p < mk; p++) {
            for (int q = p; q < mk; q++) {
                int a = keep[p];
                int b = keep[q];
                double g = a <= b ? G[a][b] : G[b][a];
                double c = (g - N * cmu[a] * cmu[b]) / (csd[a] * csd[b]) / (N - 1);
                C[p][q] = c;
                C[q][p] = c;
            }
        }
        double[] evals = new double[mk];
        double[][] vecs = SymmetricEigen.decompose(C, evals);
        double total = 0;
        for (int c = 0; c < mk; c++) {
            evals[c] = Math.max(evals[c], 0);
            total += evals[c];
        }
        double[][] v = new double[mk][m];
        double[] invW = new double[mk];
        for (int c = 0; c < mk; c++) {
            double[] e = vecs[c];
            int arg = 0;
            for (int p = 1; p < mk; p++) {
                if (Math.abs(e[p]) > Math.abs(e[arg])) {
                    arg = p;
                }
            }
            double sgn = e[arg] < 0 ? -1 : 1;
            for (int p = 0; p < mk; p++) {
                v[c][keep[p]] = sgn * e[p];
            }
            invW[c] = total / evals[c];
        }
        double[] colInv = new double[m];
        for (int a = 0; a < m; a++) {
            colInv[a] = csd[a] > 0 ? 1 / csd[a] : 0;
        }
        int comps = mk;
        double[] s = new double[N];
        IntStream.range(0, N).parallel().forEach(i -> {
            double[] z = new double[m];
            double nrm = 0;
            for (int a = 0; a < m; a++) {
                z[a] = (xd[i + a] - cmu[a]) * colInv[a];
                nrm += z[a] * z[a];
            }
            double score = 0;
            for (int c = 0; c < comps; c++) {
                double dot = 0;
                for (int a = 0; a < m; a++) {
                    dot += z[a] * v[c][a];
                }
                score += invW[c] * Math.sqrt(Math.max(nrm + 1.0 - 2.0 * dot, 0.0));
            }
            s[i] = score;
        });
        return Detectors.pad(s, m, n);
    }

    /** Sub-KNN, as {@link Detectors.SubKnn}: distance to the k-th nearest other window, column-scaled. */
    public static double[] subKnn(double[] x, int m, int k) {
        int n = x.length;
        int N = n - m + 1;
        double g = Detectors.mean(x);
        double[] xd = new double[n];
        for (int i = 0; i < n; i++) {
            xd[i] = x[i] - g;
        }
        double[] inv2 = new double[m];
        IntStream.range(0, m).parallel().forEach(a -> {
            double s = 0;
            for (int i = 0; i < N; i++) {
                s += xd[i + a];
            }
            double cm = s / N;
            double q = 0;
            for (int i = 0; i < N; i++) {
                double d = xd[i + a] - cm;
                q += d * d;
            }
            double var = q / N;
            inv2[a] = var > 0 ? 1 / var : 0;
        });
        double[] s = new double[N];
        IntStream.range(0, N).parallel().forEach(q -> {
            // the k smallest distances, as a sorted array
            double[] best = new double[k];
            Arrays.fill(best, Double.POSITIVE_INFINITY);
            for (int j = 0; j < N; j++) {
                if (j == q) {
                    continue;
                }
                double d = 0;
                double worst = best[k - 1];
                for (int a = 0; a < m && d < worst; a++) {
                    double diff = xd[q + a] - xd[j + a];
                    d += inv2[a] * diff * diff;
                }
                if (d < worst) {
                    int p = k - 1;
                    while (p > 0 && best[p - 1] > d) {
                        best[p] = best[p - 1];
                        p--;
                    }
                    best[p] = d;
                }
            }
            s[q] = Math.sqrt(best[k - 1]);
        });
        return Detectors.pad(s, m, n);
    }

    /**
     * KMeansAD (stride 1), as {@link Detectors.KMeansAd}: the same seeding sequence (k-means++ with local
     * trials from {@code new Random(seed)}) and Lloyd iterations, in double precision.
     */
    public static double[] kmeansAd(double[] x, int m, int k, long seed) {
        int n = x.length;
        int N = n - m + 1;
        double g = Detectors.mean(x);
        double[] xd = new double[n];
        for (int i = 0; i < n; i++) {
            xd[i] = x[i] - g;
        }
        double[] mu = new double[N];
        double[] inv = new double[N];
        IntStream.range(0, N).parallel().forEach(i -> {
            double s = 0;
            double mn = Double.POSITIVE_INFINITY;
            double mx = Double.NEGATIVE_INFINITY;
            for (int a = 0; a < m; a++) {
                double v = xd[i + a];
                s += v;
                mn = Math.min(mn, v);
                mx = Math.max(mx, v);
            }
            if (mn == mx) {
                mu[i] = mn;
                inv[i] = 0;
                return;
            }
            double mi = s / m;
            double q = 0;
            for (int a = 0; a < m; a++) {
                double d = xd[i + a] - mi;
                q += d * d;
            }
            double sd = Math.sqrt(q / (m - 1));
            mu[i] = mi;
            inv[i] = sd > 0 ? 1 / sd : 0;
        });
        double[] colVar = IntStream.range(0, m).parallel().mapToDouble(a -> {
            double s = 0;
            double q = 0;
            for (int i = 0; i < N; i++) {
                double z = (xd[i + a] - mu[i]) * inv[i];
                s += z;
                q += z * z;
            }
            double cm = s / N;
            return q / N - cm * cm;
        }).toArray();
        double tv = 0;
        for (double c : colVar) {
            tv += c;
        }
        double tol = 1e-4 * tv / m;
        // squared distance of window i to the centre c
        java.util.function.BiFunction<Integer, double[], Double> dist2 = (i, c) -> {
            double d = 0;
            for (int a = 0; a < m; a++) {
                double diff = (xd[i + a] - mu[i]) * inv[i] - c[a];
                d += diff * diff;
            }
            return d;
        };
        java.util.function.IntFunction<double[]> window = i -> {
            double[] w = new double[m];
            for (int a = 0; a < m; a++) {
                w[a] = (xd[i + a] - mu[i]) * inv[i];
            }
            return w;
        };
        Random rng = new Random(seed);
        int trials = Math.min(8, 2 + (int) Math.log(k));
        int[] chosen = new int[k];
        chosen[0] = rng.nextInt(N);
        double[] c0 = window.apply(chosen[0]);
        double[] closest = IntStream.range(0, N).parallel().mapToDouble(i -> dist2.apply(i, c0)).toArray();
        double pot = Arrays.stream(closest).sum();
        double[] cum = new double[N];
        for (int c = 1; c < k; c++) {
            double run = 0;
            for (int i = 0; i < N; i++) {
                run += closest[i];
                cum[i] = run;
            }
            int[] ids = new int[trials];
            for (int t = 0; t < trials; t++) {
                double r = rng.nextDouble() * pot;
                int lo = 0;
                int hi = N - 1;
                while (lo < hi) {
                    int mid = (lo + hi) >>> 1;
                    if (cum[mid] < r) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                ids[t] = lo;
            }
            int bestT = 0;
            double bestPot = Double.POSITIVE_INFINITY;
            double[] bestD = null;
            for (int t = 0; t < trials; t++) {
                double[] ct = window.apply(ids[t]);
                double[] d = IntStream.range(0, N).parallel().mapToDouble(i -> dist2.apply(i, ct)).toArray();
                double p = 0;
                for (int i = 0; i < N; i++) {
                    p += Math.min(closest[i], d[i]);
                }
                if (p < bestPot) {
                    bestPot = p;
                    bestT = t;
                    bestD = d;
                }
            }
            chosen[c] = ids[bestT];
            for (int i = 0; i < N; i++) {
                closest[i] = Math.min(closest[i], bestD[i]);
            }
            pot = bestPot;
        }
        double[][] C = new double[k][];
        for (int c = 0; c < k; c++) {
            C[c] = window.apply(chosen[c]);
        }
        int[] lab = new int[N];
        double[] dist = new double[N];
        int[] prev = null;
        for (int it = 0; it < 300; it++) {
            double[][] cc = C;
            IntStream.range(0, N).parallel().forEach(i -> {
                int arg = 0;
                double best = Double.POSITIVE_INFINITY;
                for (int c = 0; c < k; c++) {
                    double d = dist2.apply(i, cc[c]);
                    if (d < best) {
                        best = d;
                        arg = c;
                    }
                }
                lab[i] = arg;
                dist[i] = best;
            });
            if (prev != null && Arrays.equals(prev, lab)) {
                break;
            }
            prev = lab.clone();
            double[][] sum = new double[k][m];
            int[] counts = new int[k];
            for (int i = 0; i < N; i++) {
                counts[lab[i]]++;
                for (int a = 0; a < m; a++) {
                    sum[lab[i]][a] += (xd[i + a] - mu[i]) * inv[i];
                }
            }
            double shift = 0;
            double[][] next = new double[k][m];
            for (int c = 0; c < k; c++) {
                for (int a = 0; a < m; a++) {
                    next[c][a] = counts[c] > 0 ? sum[c][a] / counts[c] : C[c][a];
                    double d = next[c][a] - C[c][a];
                    shift += d * d;
                }
            }
            C = next;
            if (shift <= tol) {
                double[][] cf = C;
                IntStream.range(0, N).parallel().forEach(i -> {
                    double best = Double.POSITIVE_INFINITY;
                    for (int c = 0; c < k; c++) {
                        best = Math.min(best, dist2.apply(i, cf[c]));
                    }
                    dist[i] = best;
                });
                break;
            }
        }
        double[] cs = new double[N + 1];
        for (int i = 0; i < N; i++) {
            cs[i + 1] = cs[i] + Math.sqrt(dist[i]);
        }
        double[] res = new double[n];
        for (int t = 0; t < n; t++) {
            int lo = Math.max(t - m + 1, 0);
            int hi = Math.min(t, N - 1);
            res[t] = (cs[hi + 1] - cs[lo]) / (hi - lo + 1);
        }
        return res;
    }

    /**
     * MatrixProfile, as {@link Detectors.MatrixProfile} (stumpy's self-join, exclusion zone ceil(m / 4)): the
     * best Pearson correlation of each subsequence, by diagonals with the sliding dot product in double
     * precision, then stumpy's rules for constant subsequences.
     */
    public static double[] matrixProfile(double[] x, int m) {
        int n = x.length;
        int N = n - m + 1;
        double g = Detectors.mean(x);
        double[] xd = new double[n];
        for (int i = 0; i < n; i++) {
            xd[i] = x[i] - g;
        }
        double[] mu = new double[N];
        double[] inv = new double[N];
        boolean[] constant = new boolean[N];
        IntStream.range(0, N).parallel().forEach(i -> {
            double s = 0;
            double mn = Double.POSITIVE_INFINITY;
            double mx = Double.NEGATIVE_INFINITY;
            for (int a = 0; a < m; a++) {
                double v = xd[i + a];
                s += v;
                mn = Math.min(mn, v);
                mx = Math.max(mx, v);
            }
            double mi = s / m;
            double q = 0;
            for (int a = 0; a < m; a++) {
                double v = xd[i + a] - mi;
                q += v * v;
            }
            mu[i] = mi;
            constant[i] = mn == mx;
            inv[i] = constant[i] || q == 0 ? 0 : 1 / Math.sqrt(q);
        });
        int excl = (int) Math.ceil(m / 4.0);
        int[] cum = new int[N + 1];
        for (int i = 0; i < N; i++) {
            cum[i + 1] = cum[i] + (constant[i] ? 1 : 0);
        }
        // per chunk of diagonals, the best correlation of every row; merged at the end
        int chunks = Math.max(1, Math.min(4 * Runtime.getRuntime().availableProcessors(), N - excl - 1));
        double[][] partial = new double[chunks][];
        IntStream.range(0, chunks).parallel().forEach(ch -> {
            double[] best = new double[N];
            Arrays.fill(best, Double.NEGATIVE_INFINITY);
            for (int d = excl + 1 + ch; d < N; d += chunks) {
                // dot product of the centred subsequences i and i + d, slid down the diagonal
                double dot = 0;
                for (int a = 0; a < m; a++) {
                    dot += xd[a] * xd[d + a];
                }
                for (int i = 0; i + d < N; i++) {
                    if (i > 0) {
                        dot += xd[i + m - 1] * xd[i + d + m - 1] - xd[i - 1] * xd[i + d - 1];
                    }
                    int j = i + d;
                    double c = (dot - m * mu[i] * mu[j]) * inv[i] * inv[j];
                    if (c > best[i]) {
                        best[i] = c;
                    }
                    if (c > best[j]) {
                        best[j] = c;
                    }
                }
            }
            partial[ch] = best;
        });
        double[] corr = new double[N];
        Arrays.fill(corr, Double.NEGATIVE_INFINITY);
        for (double[] p : partial) {
            for (int i = 0; i < N; i++) {
                corr[i] = Math.max(corr[i], p[i]);
            }
        }
        double[] prof = new double[N];
        double min = Double.POSITIVE_INFINITY;
        for (int i = 0; i < N; i++) {
            double c = corr[i] == Double.NEGATIVE_INFINITY ? 0 : corr[i];
            double d = Math.sqrt(Math.max(2.0 * m * (1 - c), 0));
            int lo = Math.max(0, i - excl);
            int hi = Math.min(N, i + excl + 1);
            boolean constFar = cum[N] - (cum[hi] - cum[lo]) > 0;
            if (constFar) {
                d = Math.min(d, Math.sqrt(m));
                if (constant[i]) {
                    d = 0;
                }
            }
            prof[i] = d;
            min = Math.min(min, d);
        }
        double[] res = new double[n];
        Arrays.fill(res, min);
        System.arraycopy(prof, 0, res, m / 2, N);
        return res;
    }
}
