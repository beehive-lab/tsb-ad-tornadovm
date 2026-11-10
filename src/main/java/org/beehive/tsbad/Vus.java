/*
 * Copyright (c) 2026 Beehive lab
 * SPDX-License-Identifier: MIT
 */
package org.beehive.tsbad;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * VUS-ROC and VUS-PR (TSB-AD's {@code get_metrics}, version 'opt', 250 thresholds) with the window x
 * threshold grid on the device.
 *
 * <p>For buffer window w and threshold thr the reference loop reduces to TP = #(anomaly points predicted)
 * + S and N_labels = #(anomaly points) + S, with S = sum over normal points of ext_w(t) * [score(t) >= thr]
 * and ext_w the extended label. The device computes ext for every window and S for the whole grid; the
 * host finds the thresholds, the existence term (per-segment score maxima) and the areas.
 */
public final class Vus {

    public final double vusRoc;
    public final double vusPr;
    public final long nanos;

    private Vus(double roc, double pr, long nanos) {
        this.vusRoc = roc;
        this.vusPr = pr;
        this.nanos = nanos;
    }

    private static final int LOCAL = 256;

    private record Runs(int[] s, int[] e) {
    }

    static Runs runs(int[] label) {
        List<Integer> s = new ArrayList<>();
        List<Integer> e = new ArrayList<>();
        int n = label.length;
        for (int t = 0; t < n; t++) {
            if (label[t] == 1 && (t == 0 || label[t - 1] != 1)) {
                s.add(t);
            }
            if (label[t] == 1 && (t == n - 1 || label[t + 1] != 1)) {
                e.add(t);
            }
        }
        return new Runs(s.stream().mapToInt(Integer::intValue).toArray(), e.stream().mapToInt(Integer::intValue).toArray());
    }

    /** new_sequence(): runs extended by window / 2 and merged where they overlap; ends inclusive. */
    static int[][] merged(Runs r, int window, int n) {
        int h = window / 2;
        int K = r.s.length;
        List<int[]> out = new ArrayList<>();
        int a = Math.max(r.s[0] - h, 0);
        for (int k = 0; k < K - 1; k++) {
            if (r.e[k] + h < r.s[k + 1] - h) {
                out.add(new int[] {a, r.e[k] + h});
                a = r.s[k + 1] - h;
            }
        }
        out.add(new int[] {a, Math.min(r.e[K - 1] + h, n - 1)});
        return out.toArray(new int[0][]);
    }

    public static Vus compute(int[] label, double[] score, int slidingWindow) {
        return compute(label, score, slidingWindow, 250);
    }

    public static Vus compute(int[] label, double[] score, int slidingWindow, int thre) {
        long t0 = System.nanoTime();
        int n = score.length;
        int W = slidingWindow + 1;
        Runs runs = runs(label);
        if (runs.s.length == 0) {
            throw new IllegalArgumentException("no anomalies in the labels");
        }
        double P = 0;
        for (int v : label) {
            P += v;
        }

        // thresholds: score_sorted[linspace(0, n - 1, thre).astype(int)], descending
        double[] asc = score.clone();
        Arrays.sort(asc);
        double[] thr = new double[thre];
        double[] nPred = new double[thre];
        double[] aPred = new double[thre];
        double[] anom = new double[(int) P];
        for (int t = 0, q = 0; t < n; t++) {
            if (label[t] == 1) {
                anom[q++] = score[t];
            }
        }
        Arrays.sort(anom);
        double step = (double) (n - 1) / (thre - 1);
        for (int j = 0; j < thre; j++) {
            // np.linspace(0, n - 1, thre).astype(int)
            int idx = j == thre - 1 ? n - 1 : (int) (j * step);
            thr[j] = asc[n - 1 - idx];
            nPred[j] = n - lowerBound(asc, thr[j]);
            aPred[j] = anom.length - lowerBound(anom, thr[j]);
        }

        // points of the l segments (largest window): the only ones that can carry an extended label
        int[][] l = merged(runs, slidingWindow, n);
        int R = 0;
        for (int[] seg : l) {
            R += seg[1] - seg[0] + 1;
        }
        int hMax = slidingWindow / 2;
        double[][] S = new double[W][thre];
        if (hMax > 0 && R > 0) {
            int K = runs.s.length;
            IntArray segS = IntArray.fromArray(runs.s);
            IntArray segE = IntArray.fromArray(runs.e);
            IntArray rIdx = new IntArray(R);
            IntArray normal = new IntArray(R);
            IntArray segLo = new IntArray(R);
            IntArray firstJ = new IntArray(R);
            int r = 0;
            for (int[] seg : l) {
                for (int t = seg[0]; t <= seg[1]; t++) {
                    rIdx.set(r, t);
                    normal.set(r, label[t] == 1 ? 0 : 1);
                    // first run whose end is within hMax before t
                    int lo = 0;
                    int hi = K;
                    while (lo < hi) {
                        int mid = (lo + hi) >>> 1;
                        if (runs.e[mid] < t - hMax) {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    segLo.set(r, lo);
                    firstJ.set(r, firstAtOrBelow(thr, score[t]));
                    r++;
                }
            }
            FloatArray ext = new FloatArray(W * R);
            FloatArray acc = new FloatArray(W * thre);

            TaskGraph g = new TaskGraph("vus") //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, segS, segE, rIdx, normal, segLo, firstJ) //
                    .task("ext", Kernels::vusExt, new KernelContext(), segS, segE, rIdx, normal, segLo, ext, K, W, R, hMax) //
                    .task("acc", Kernels::vusAccumulate, new KernelContext(), ext, firstJ, acc, W, R, thre) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, acc);
            GridScheduler gs = new GridScheduler();
            gs.addWorkerGrid("vus.ext", Grids.grid1(Grids.roundUp((long) W * R, LOCAL), LOCAL));
            WorkerGrid g2 = new WorkerGrid2D(Grids.roundUpInt(thre, Kernels.VUS_TILE), Grids.roundUpInt(W, Kernels.VUS_TILE));
            g2.setLocalWork(Kernels.VUS_TILE, Kernels.VUS_TILE, 1);
            gs.addWorkerGrid("vus.acc", g2);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
                plan.withGridScheduler(gs).execute();
            } catch (Exception ex) {
                throw new RuntimeException(ex);
            }
            for (int w = 0; w < W; w++) {
                for (int j = 0; j < thre; j++) {
                    S[w][j] = acc.get(w * thre + j);
                }
            }
        }

        // existence: merged segments of each window holding a predicted point (score max >= threshold)
        SparseMax sm = new SparseMax(score);
        double rocSum = 0;
        double prSum = 0;
        double[] tpr = new double[thre + 2];
        double[] fpr = new double[thre + 2];
        double[] prec = new double[thre + 1];
        for (int w = 0; w < W; w++) {
            int[][] L = merged(runs, w, n);
            double[] segMax = new double[L.length];
            for (int g = 0; g < L.length; g++) {
                segMax[g] = sm.max(L[g][0], L[g][1]);
            }
            Arrays.sort(segMax);
            tpr[0] = 0;
            fpr[0] = 0;
            prec[0] = 1;
            for (int j = 0; j < thre; j++) {
                double existence = segMax.length - lowerBound(segMax, thr[j]);
                double tp = aPred[j] + S[w][j];
                double nLabels = P + S[w][j];
                double fp = nPred[j] - tp;
                double pNew = (P + nLabels) / 2;
                double recall = Math.min(tp / pNew, 1);
                tpr[j + 1] = recall * existence / L.length;
                fpr[j + 1] = fp / (n - pNew);
                prec[j + 1] = tp / nPred[j];
            }
            tpr[thre + 1] = 1;
            fpr[thre + 1] = 1;
            double auc = 0;
            for (int j = 0; j <= thre; j++) {
                auc += (fpr[j + 1] - fpr[j]) * (tpr[j + 1] + tpr[j]) / 2;
            }
            double ap = 0;
            for (int j = 1; j <= thre; j++) {
                ap += (tpr[j] - tpr[j - 1]) * prec[j];
            }
            rocSum += auc;
            prSum += ap;
        }
        return new Vus(rocSum / W, prSum / W, System.nanoTime() - t0);
    }

    /** First index j of the descending thresholds with thr[j] <= v (thr.length if none). */
    static int firstAtOrBelow(double[] thr, double v) {
        int lo = 0;
        int hi = thr.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (thr[mid] > v) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    /** Number of entries of the ascending array a below v. */
    static int lowerBound(double[] a, double v) {
        int lo = 0;
        int hi = a.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (a[mid] < v) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    /** Range maxima in O(1) after O(n log n) preprocessing. */
    static final class SparseMax {
        private final double[][] table;

        SparseMax(double[] a) {
            int n = a.length;
            int levels = 1;
            while ((1 << levels) <= n) {
                levels++;
            }
            table = new double[levels][];
            table[0] = a;
            for (int k = 1; k < levels; k++) {
                int half = 1 << (k - 1);
                double[] prev = table[k - 1];
                double[] cur = new double[prev.length - half];
                for (int i = 0; i < cur.length; i++) {
                    cur[i] = Math.max(prev[i], prev[i + half]);
                }
                table[k] = cur;
            }
        }

        double max(int a, int b) {
            int len = b - a + 1;
            int k = 31 - Integer.numberOfLeadingZeros(len);
            return Math.max(table[k][a], table[k][b - (1 << k) + 1]);
        }
    }
}
