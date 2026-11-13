/*
 * Copyright (c) 2026 Beehive lab
 * SPDX-License-Identifier: MIT
 */
package org.beehive.tsbad;

import java.util.Random;
import java.util.stream.IntStream;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * TornadoVM versions of TSB-AD's univariate subsequence detectors. Each one keeps its execution plan, so
 * that {@link Detector#score()} can be repeated on the same series without recompiling.
 */
public final class Detectors {

    private Detectors() {
    }

    private static final int LOCAL = 256;

    /**
     * A detector bound to one series. Sub-PCA, Sub-KNN and MatrixProfile can be rebound to another series of the
     * same length, which reuses their device buffers and compiled execution plan.
     */
    public interface Detector extends AutoCloseable {
        /** Point anomaly scores, one per sample, like the CPU detector's decision_scores_. */
        double[] score();

        /** Whether {@link #rebind} is supported. */
        default boolean reusable() {
            return false;
        }

        /** Binds the detector to another series of the same length; its next {@link #score()} scores that one. */
        default void rebind(double[] x) {
            throw new UnsupportedOperationException(getClass().getSimpleName() + " cannot be rebound");
        }

        @Override
        void close();
    }

    static TornadoExecutionPlan plan(GridScheduler gs, TaskGraph... graphs) {
        var snaps = new uk.ac.manchester.tornado.api.ImmutableTaskGraph[graphs.length];
        for (int i = 0; i < graphs.length; i++) {
            snaps[i] = graphs[i].snapshot();
        }
        TornadoExecutionPlan p = new TornadoExecutionPlan(snaps);
        p.withGridScheduler(gs);
        return p;
    }

    static void closePlan(TornadoExecutionPlan p) {
        try {
            p.close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    static double mean(double[] x) {
        double s = 0;
        for (double v : x) {
            s += v;
        }
        return s / x.length;
    }

    /** Remainders v - (float) v, the low half of a float pair. */
    static FloatArray remainder(double[] v) {
        FloatArray a = new FloatArray(v.length);
        for (int i = 0; i < v.length; i++) {
            a.set(i, (float) (v[i] - (float) v[i]));
        }
        return a;
    }

    /** Series minus its mean, as float (the detectors are shift invariant). */
    static FloatArray centred(double[] x, double mean) {
        return centredInto(new FloatArray(x.length), x, mean);
    }

    static FloatArray centredInto(FloatArray a, double[] x, double mean) {
        for (int i = 0; i < x.length; i++) {
            a.set(i, (float) (x[i] - mean));
        }
        return a;
    }

    static void checkLength(double[] x, int n) {
        if (x.length != n) {
            throw new IllegalArgumentException("Rebinding needs a series of " + n + " points; got " + x.length + ".");
        }
    }

    /** Pads window scores to n points: first value ceil((w - 1) / 2) times in front, last (w - 1) / 2 behind. */
    static double[] pad(double[] s, int w, int n) {
        if (s.length >= n) {
            return s;
        }
        double[] out = new double[n];
        int front = (w - 1 + 1) / 2;
        for (int i = 0; i < n; i++) {
            int j = Math.min(Math.max(i - front, 0), s.length - 1);
            out[i] = s[j];
        }
        return out;
    }

    // ------------------------------------------------------------------ MatrixProfile

    /** stumpy.stump self-join as wrapped by TSB-AD's MatrixProfile (exclusion zone ceil(m / 4)). */
    public static final class MatrixProfile implements Detector {
        private final int n;
        private final int m;
        private final int N;
        private final int excl;
        private final boolean[] constant;
        private final boolean[] constFar;
        private final FloatArray best;
        private final IntArray bestIdx;
        private final double[] xd;
        private final double[] mu;
        private final double[] inv;
        private final FloatArray xc;
        private final FloatArray xlo;
        private final FloatArray muLo;
        private final FloatArray muD;
        private final FloatArray invD;
        private final FloatArray df;
        private final FloatArray dg;
        private final TornadoExecutionPlan plan;

        public MatrixProfile(double[] x, int m) {
            this.n = x.length;
            this.m = m;
            this.N = n - m + 1;
            this.excl = (int) Math.ceil(m / 4.0);
            xd = new double[n];
            mu = new double[N];
            inv = new double[N];
            constant = new boolean[N];
            constFar = new boolean[N];
            xc = new FloatArray(n);
            xlo = new FloatArray(n);
            muLo = new FloatArray(N);
            muD = new FloatArray(N);
            invD = new FloatArray(N);
            df = new FloatArray(N);
            dg = new FloatArray(N);
            best = new FloatArray(N);
            bestIdx = new IntArray(N);
            load(x);
            TaskGraph tg = new TaskGraph("mp") //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, xc, xlo, muD, muLo, invD, df, dg) //
                    .task("rowmax", Kernels::mpRowMax, new KernelContext(), xc, xlo, muD, muLo, invD, df, dg, best, bestIdx, N, m, excl) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, best, bestIdx);
            GridScheduler gs = new GridScheduler();
            gs.addWorkerGrid("mp.rowmax", Grids.grid1(Grids.roundUp(N, Kernels.MP_T), Kernels.MP_T));
            plan = plan(gs, tg);
        }

        @Override
        public boolean reusable() {
            return true;
        }

        @Override
        public void rebind(double[] x) {
            checkLength(x, n);
            load(x);
        }

        private void load(double[] x) {
            double g = mean(x);
            for (int i = 0; i < n; i++) {
                xd[i] = x[i] - g;
            }
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
            // a constant subsequence is at distance 0 from another constant one and sqrt(m) from the rest
            int[] cum = new int[N + 1];
            for (int i = 0; i < N; i++) {
                cum[i + 1] = cum[i] + (constant[i] ? 1 : 0);
            }
            for (int i = 0; i < N; i++) {
                int lo = Math.max(0, i - excl);
                int hi = Math.min(N, i + excl + 1);
                constFar[i] = cum[N] - (cum[hi] - cum[lo]) > 0;
            }
            centredInto(xc, xd, 0);
            for (int i = 0; i < n; i++) {
                xlo.set(i, (float) (xd[i] - (float) xd[i]));
            }
            for (int i = 0; i < N; i++) {
                muLo.set(i, (float) (mu[i] - (float) mu[i]));
                muD.set(i, (float) mu[i]);
                invD.set(i, (float) inv[i]);
                if (i > 0) {
                    df.set(i, (float) ((xd[i + m - 1] - xd[i - 1]) / 2));
                    dg.set(i, (float) ((xd[i + m - 1] - mu[i]) + (xd[i - 1] - mu[i - 1])));
                } else {
                    df.set(0, 0f);
                    dg.set(0, 0f);
                }
            }
        }

        @Override
        public double[] score() {
            plan.execute();
            // the correlation with the best match found in float32, again in float64
            double[] corr = new double[N];
            IntStream.range(0, N).parallel().forEach(i -> {
                int j = bestIdx.get(i);
                if (j < 0) {
                    corr[i] = best.get(i);
                    return;
                }
                double num = 0;
                double qi = 0;
                double qj = 0;
                for (int a = 0; a < m; a++) {
                    double ci = xd[i + a] - mu[i];
                    double cj = xd[j + a] - mu[j];
                    num += ci * cj;
                    qi += ci * ci;
                    qj += cj * cj;
                }
                double den = Math.sqrt(qi * qj);
                corr[i] = den > 0 ? num / den : 0;
            });
            double[] prof = new double[N];
            double min = Double.POSITIVE_INFINITY;
            for (int i = 0; i < N; i++) {
                double c = corr[i];
                double d = Math.sqrt(Math.max(2.0 * m * (1 - c), 0));
                if (constFar[i]) {
                    d = Math.min(d, Math.sqrt(m));
                    if (constant[i]) {
                        d = 0;
                    }
                }
                prof[i] = d;
                min = Math.min(min, d);
            }
            // res[m // 2 : -m // 2 + 1] = profile, the rest the profile minimum
            double[] res = new double[n];
            java.util.Arrays.fill(res, min);
            System.arraycopy(prof, 0, res, m / 2, N);
            return res;
        }

        @Override
        public void close() {
            closePlan(plan);
        }
    }

    // ------------------------------------------------------------------ Sub-PCA

    /** TSB-AD's Sub-PCA (models.PCA with its defaults): all components, weighted by explained variance. */
    public static final class SubPca implements Detector {
        private double[] x;
        private final int n;
        private final int m;
        private final int N;
        private final FloatArray xc;
        private final FloatArray colMu;
        private final FloatArray colInv;
        private final FloatArray v;
        private final FloatArray invW;
        private final FloatArray out;
        private final TornadoExecutionPlan plan;
        private double shift;

        public SubPca(double[] x, int m) {
            this.x = x;
            this.n = x.length;
            this.m = m;
            this.N = n - m + 1;
            this.shift = mean(x);
            xc = centred(x, shift);
            colMu = new FloatArray(m);
            colInv = new FloatArray(m);
            v = new FloatArray(m * m);
            invW = new FloatArray(m);
            out = new FloatArray(N);
            TaskGraph tg = new TaskGraph("pca") //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, xc, colMu, colInv, v, invW) //
                    .task("score", Kernels::pcaScore, new KernelContext(), xc, colMu, colInv, v, invW, out, N, m, m) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            GridScheduler gs = new GridScheduler();
            gs.addWorkerGrid("pca.score", Grids.grid1(Grids.roundUp(N, LOCAL), LOCAL));
            plan = plan(gs, tg);
        }

        @Override
        public boolean reusable() {
            return true;
        }

        @Override
        public void rebind(double[] x) {
            checkLength(x, n);
            this.x = x;
            this.shift = mean(x);
            centredInto(xc, x, shift);
        }

        @Override
        public double[] score() {
            // column statistics and the Gram matrix of the window matrix in double precision; the window
            // (Hankel) structure gives G[a][b] = G[a-1][b-1] - x[a-1] x[b-1] + x[a-1+N] x[b-1+N]
            double[] xd = new double[n];
            for (int i = 0; i < n; i++) {
                xd[i] = x[i] - shift;
            }
            double[] cmu = new double[m];
            double[] csd = new double[m];
            double[][] G = new double[m][m];
            IntStream.range(0, m).parallel().forEach(b -> {
                double s = 0;
                double q = 0;
                double g0 = 0;
                for (int i = 0; i < N; i++) {
                    double vb = xd[i + b];
                    s += vb;
                    g0 += xd[i] * vb;
                }
                cmu[b] = s / N;
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
            // zero pruning: constant columns standardise to 0 and are dropped
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
            for (int a = 0; a < m; a++) {
                colMu.set(a, (float) cmu[a]);
                colInv.set(a, csd[a] > 0 ? (float) (1 / csd[a]) : 0f);
            }
            for (int i = 0; i < m * m; i++) {
                v.set(i, 0f);
            }
            for (int c = 0; c < m; c++) {
                invW.set(c, 0f);
            }
            for (int c = 0; c < mk; c++) {
                double[] e = vecs[c];
                // svd_flip(u_based_decision=False): largest |entry| of each component positive
                int arg = 0;
                for (int p = 1; p < mk; p++) {
                    if (Math.abs(e[p]) > Math.abs(e[arg])) {
                        arg = p;
                    }
                }
                double sgn = e[arg] < 0 ? -1 : 1;
                for (int p = 0; p < mk; p++) {
                    v.set(c * m + keep[p], (float) (sgn * e[p]));
                }
                invW.set(c, (float) (total / evals[c]));
            }
            plan.execute();
            double[] s = new double[N];
            for (int i = 0; i < N; i++) {
                s[i] = out.get(i);
            }
            return pad(s, m, n);
        }

        @Override
        public void close() {
            closePlan(plan);
        }
    }

    // ------------------------------------------------------------------ KMeansAD

    /**
     * TSB-AD's KMeansAD (stride 1): z-normalised windows (ddof 1), k-means++ seeding and Lloyd iterations as
     * in sklearn's KMeans (max_iter 300, tol 1e-4), window score = distance to its centre, point score =
     * mean over the windows that cover it.
     */
    public static final class KMeansAd implements Detector {
        private static final int TRIALS_MAX = 8;
        private static final int CHUNKS = 64;
        private final int n;
        private final int m;
        private final int N;
        private final int k;
        private final long seed;
        private final float[] rowMuH;
        private final float[] rowInvH;
        private final double tol;
        private final double[] xd;
        private final double[] mu;
        private final double[] inv;
        private final FloatArray xc;
        private final FloatArray xlo;
        private final FloatArray rowMu;
        private final FloatArray rowMuLo;
        private final FloatArray rowInv;
        private final FloatArray cand;
        private final FloatArray candDist;
        private final FloatArray centres;
        private final IntArray labels;
        private final FloatArray dist;
        private final FloatArray partial;
        private final TornadoExecutionPlan plan;
        public int iterations;

        public KMeansAd(double[] x, int window, int k, long seed) {
            this.n = x.length;
            this.m = window;
            this.N = n - m + 1;
            this.k = k;
            this.seed = seed;
            double g = mean(x);
            xd = new double[n];
            for (int i = 0; i < n; i++) {
                xd[i] = x[i] - g;
            }
            rowMuH = new float[N];
            rowInvH = new float[N];
            mu = new double[N];
            inv = new double[N];
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
                    // constant window: z-scores are 0 (not rounding noise divided by a rounding-noise std)
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
            // tol = 1e-4 * mean of the column variances of the normalised windows
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
            tol = 1e-4 * tv / m;
            xc = centred(xd, 0);
            xlo = remainder(xd);
            rowMuLo = remainder(mu);
            rowMu = new FloatArray(N);
            rowInv = new FloatArray(N);
            for (int i = 0; i < N; i++) {
                rowMuH[i] = (float) mu[i];
                rowInvH[i] = (float) inv[i];
                rowMu.set(i, rowMuH[i]);
                rowInv.set(i, rowInvH[i]);
            }
            cand = new FloatArray(TRIALS_MAX * m);
            candDist = new FloatArray(TRIALS_MAX * N);
            centres = new FloatArray(k * m);
            labels = new IntArray(N);
            dist = new FloatArray(N);
            partial = new FloatArray(CHUNKS * k * m);

            TaskGraph pp = new TaskGraph("kmpp") //
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, xc, xlo, rowMu, rowMuLo, rowInv) //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, cand) //
                    .task("dist", Kernels::kmDist, new KernelContext(), xc, xlo, rowMu, rowMuLo, rowInv, cand, candDist, N, m, TRIALS_MAX) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, candDist);
            TaskGraph ll = new TaskGraph("lloyd") //
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, xc, xlo, rowMu, rowMuLo, rowInv) //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, centres) //
                    .task("assign", Kernels::kmAssign, new KernelContext(), xc, xlo, rowMu, rowMuLo, rowInv, centres, labels, dist, N, m, k) //
                    .task("sums", Kernels::kmSums, new KernelContext(), xc, xlo, rowMu, rowMuLo, rowInv, labels, partial, N, m, k, CHUNKS) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, labels, dist, partial);
            GridScheduler gs = new GridScheduler();
            gs.addWorkerGrid("kmpp.dist", Grids.grid1(Grids.roundUp(N, LOCAL), LOCAL));
            gs.addWorkerGrid("lloyd.assign", Grids.grid1(Grids.roundUp(N, LOCAL), LOCAL));
            gs.addWorkerGrid("lloyd.sums", Grids.grid1(Grids.roundUp((long) CHUNKS * k * m, LOCAL), LOCAL));
            plan = plan(gs, pp, ll);
        }

        private void window(int i, float[] dst, int off) {
            for (int a = 0; a < m; a++) {
                dst[off + a] = (float) ((xd[i + a] - mu[i]) * inv[i]);
            }
        }

        /** Squared distances of all windows to the given windows (at most TRIALS_MAX), [t][i]. */
        private float[][] distTo(int[] ids) {
            float[] buf = new float[TRIALS_MAX * m];
            for (int t = 0; t < TRIALS_MAX; t++) {
                window(ids[Math.min(t, ids.length - 1)], buf, t * m);
            }
            for (int i = 0; i < buf.length; i++) {
                cand.set(i, buf[i]);
            }
            plan.withGraph(0).execute();
            float[][] d = new float[ids.length][N];
            for (int t = 0; t < ids.length; t++) {
                for (int i = 0; i < N; i++) {
                    d[t][i] = candDist.get(t * N + i);
                }
            }
            return d;
        }

        @Override
        public double[] score() {
            Random rng = new Random(seed);
            int trials = Math.min(TRIALS_MAX, 2 + (int) Math.log(k));
            // k-means++ with local trials (sklearn's _kmeans_plusplus)
            int[] chosen = new int[k];
            chosen[0] = rng.nextInt(N);
            float[] closest = distTo(new int[] {chosen[0]})[0];
            double pot = 0;
            for (float c : closest) {
                pot += c;
            }
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
                float[][] d = distTo(ids);
                int bestT = 0;
                double bestPot = Double.POSITIVE_INFINITY;
                for (int t = 0; t < trials; t++) {
                    double p = 0;
                    for (int i = 0; i < N; i++) {
                        p += Math.min(closest[i], d[t][i]);
                    }
                    if (p < bestPot) {
                        bestPot = p;
                        bestT = t;
                    }
                }
                chosen[c] = ids[bestT];
                for (int i = 0; i < N; i++) {
                    closest[i] = Math.min(closest[i], d[bestT][i]);
                }
                pot = bestPot;
            }
            float[] C = new float[k * m];
            for (int c = 0; c < k; c++) {
                window(chosen[c], C, c * m);
            }
            // Lloyd
            int[] prev = null;
            iterations = 0;
            for (int it = 0; it < 300; it++) {
                for (int i = 0; i < k * m; i++) {
                    centres.set(i, C[i]);
                }
                plan.withGraph(1).execute();
                iterations++;
                int[] lab = labels.toHeapArray();
                if (prev != null && java.util.Arrays.equals(prev, lab)) {
                    break;
                }
                prev = lab;
                int[] counts = new int[k];
                for (int l : lab) {
                    counts[l]++;
                }
                double shift = 0;
                for (int c = 0; c < k; c++) {
                    for (int a = 0; a < m; a++) {
                        double s = 0;
                        for (int p = 0; p < CHUNKS; p++) {
                            s += partial.get((p * k + c) * m + a);
                        }
                        float nv = counts[c] > 0 ? (float) (s / counts[c]) : C[c * m + a];
                        double dlt = nv - C[c * m + a];
                        shift += dlt * dlt;
                        C[c * m + a] = nv;
                    }
                }
                if (shift <= tol) {
                    for (int i = 0; i < k * m; i++) {
                        centres.set(i, C[i]);
                    }
                    plan.withGraph(1).execute();
                    break;
                }
            }
            // reverse windowing: mean distance of the windows covering each point
            double[] cs = new double[N + 1];
            for (int i = 0; i < N; i++) {
                cs[i + 1] = cs[i] + Math.sqrt(dist.get(i));
            }
            double[] res = new double[n];
            for (int t = 0; t < n; t++) {
                int lo = Math.max(t - m + 1, 0);
                int hi = Math.min(t, N - 1);
                res[t] = (cs[hi + 1] - cs[lo]) / (hi - lo + 1);
            }
            return res;
        }

        @Override
        public void close() {
            closePlan(plan);
        }
    }

    // ------------------------------------------------------------------ Sub-KNN

    /** TSB-AD's Sub-KNN (models.KNN, method 'largest'): distance to the k-th nearest other window. */
    public static final class SubKnn implements Detector {
        private static final int REFINE = 4;
        private final int n;
        private final int m;
        private final int N;
        private final int k;
        private final FloatArray out;
        private final IntArray outIdx;
        private final double[] xd;
        private final double[] inv2d;
        private final FloatArray xc;
        private final FloatArray inv2;
        private final TornadoExecutionPlan plan;

        public SubKnn(double[] x, int m, int k) {
            if (m > Kernels.KNN_MAX_M || k > Kernels.KNN_MAX_K) {
                throw new IllegalArgumentException("Sub-KNN supports windows up to " + Kernels.KNN_MAX_M + " and k up to " + Kernels.KNN_MAX_K);
            }
            this.n = x.length;
            this.m = m;
            this.N = n - m + 1;
            this.k = k;
            xd = new double[n];
            inv2d = new double[m];
            xc = new FloatArray(n);
            inv2 = new FloatArray(m);
            load(x);
            out = new FloatArray(N * k);
            outIdx = new IntArray(N * k);
            TaskGraph tg = new TaskGraph("knn") //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, xc, inv2) //
                    .task("topk", Kernels::knnTopK, new KernelContext(), xc, inv2, out, outIdx, N, m, k) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, out, outIdx);
            GridScheduler gs = new GridScheduler();
            gs.addWorkerGrid("knn.topk", Grids.grid1(Grids.roundUp(N, Kernels.KNN_T), Kernels.KNN_T));
            plan = plan(gs, tg);
        }

        @Override
        public boolean reusable() {
            return true;
        }

        @Override
        public void rebind(double[] x) {
            checkLength(x, n);
            load(x);
        }

        private void load(double[] x) {
            double g = mean(x);
            for (int i = 0; i < n; i++) {
                xd[i] = x[i] - g;
            }
            // column z-score (axis 0, ddof 0) of the window matrix: only the column scales matter
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
                inv2d[a] = var > 0 ? 1 / var : 0;
                inv2.set(a, (float) inv2d[a]);
            });
            centredInto(xc, xd, 0);
        }

        @Override
        public double[] score() {
            plan.execute();
            // the k-th distance is the largest of the k nearest: recompute the last few in float64 so that
            // ties and near ties come out as on the CPU
            double[] s = new double[N];
            int lo = Math.max(0, k - 1 - REFINE);
            IntStream.range(0, N).parallel().forEach(q -> {
                double worst = 0;
                for (int r = lo; r < k; r++) {
                    int j = outIdx.get(q * k + r);
                    if (j < 0) {
                        worst = Double.POSITIVE_INFINITY;
                        continue;
                    }
                    double d = 0;
                    for (int a = 0; a < m; a++) {
                        double diff = xd[q + a] - xd[j + a];
                        d += inv2d[a] * diff * diff;
                    }
                    worst = Math.max(worst, d);
                }
                s[q] = Math.sqrt(worst);
            });
            return pad(s, m, n);
        }

        @Override
        public void close() {
            closePlan(plan);
        }
    }
}
