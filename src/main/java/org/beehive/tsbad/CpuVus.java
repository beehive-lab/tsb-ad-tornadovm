/*
 * Copyright (c) 2026 Beehive lab
 * SPDX-License-Identifier: MIT
 */
package org.beehive.tsbad;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * VUS-ROC and VUS-PR as {@link Vus#compute(int[], double[], int)}, with its device step (the extended labels of
 * every buffer window and their accumulation over the thresholds) on the CPU in double precision: a reference, and
 * a fallback when the device fails. It uses the host code of {@link Vus} unchanged.
 */
public final class CpuVus {

    private CpuVus() {
    }

    /** {@code {vusRoc, vusPr}}. */
    public static double[] compute(int[] label, double[] score, int slidingWindow) {
        int thre = 250;
        int n = score.length;
        int W = slidingWindow + 1;
        int[][] runs = runs(label);
        int[] rs = runs[0];
        int[] re = runs[1];
        if (rs.length == 0) {
            throw new IllegalArgumentException("no anomalies in the labels");
        }
        double P = 0;
        for (int v : label) {
            P += v;
        }
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
            int idx = j == thre - 1 ? n - 1 : (int) (j * step);
            thr[j] = asc[n - 1 - idx];
            nPred[j] = n - Vus.lowerBound(asc, thr[j]);
            aPred[j] = anom.length - Vus.lowerBound(anom, thr[j]);
        }

        // S[w][j]: sum over normal points t of ext_w(t) * [score(t) >= thr(j)]
        int hMax = slidingWindow / 2;
        double[][] S = new double[W][thre];
        if (hMax > 0) {
            List<Integer> pts = new ArrayList<>();
            for (int[] seg : merged(rs, re, slidingWindow, n)) {
                for (int t = seg[0]; t <= seg[1]; t++) {
                    if (label[t] != 1) {
                        pts.add(t);
                    }
                }
            }
            int K = rs.length;
            IntStream.range(2, W).parallel().forEach(w -> {
                int hw = w / 2;
                double[] acc = new double[thre + 1];
                for (int t : pts) {
                    // runs within hMax of t
                    int lo = 0;
                    int hi = K;
                    while (lo < hi) {
                        int mid = (lo + hi) >>> 1;
                        if (re[mid] < t - hMax) {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    double sum = 0;
                    for (int k = lo; k < K && rs[k] <= t + hMax; k++) {
                        if (re[k] < t && t - re[k] <= hw) {
                            sum += Math.sqrt(1.0 - (double) (t - re[k]) / w);
                        }
                        if (rs[k] > t && rs[k] - t <= hw) {
                            sum += Math.sqrt(1.0 - (double) (rs[k] - t) / w);
                        }
                    }
                    if (sum > 0) {
                        // predicted at the thresholds j >= firstAtOrBelow(thr, score[t])
                        acc[Vus.firstAtOrBelow(thr, score[t])] += Math.min(sum, 1.0);
                    }
                }
                double run = 0;
                for (int j = 0; j < thre; j++) {
                    run += acc[j];
                    S[w][j] = run;
                }
            });
        }

        Vus.SparseMax sm = new Vus.SparseMax(score);
        double rocSum = 0;
        double prSum = 0;
        double[] tpr = new double[thre + 2];
        double[] fpr = new double[thre + 2];
        double[] prec = new double[thre + 1];
        for (int w = 0; w < W; w++) {
            int[][] L = merged(rs, re, w, n);
            double[] segMax = new double[L.length];
            for (int g = 0; g < L.length; g++) {
                segMax[g] = sm.max(L[g][0], L[g][1]);
            }
            Arrays.sort(segMax);
            tpr[0] = 0;
            fpr[0] = 0;
            prec[0] = 1;
            for (int j = 0; j < thre; j++) {
                double existence = segMax.length - Vus.lowerBound(segMax, thr[j]);
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
        return new double[] {rocSum / W, prSum / W};
    }

    /** Starts and ends of the runs of anomalous points. */
    static int[][] runs(int[] label) {
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
        return new int[][] {s.stream().mapToInt(Integer::intValue).toArray(), e.stream().mapToInt(Integer::intValue).toArray()};
    }

    /** The runs extended by window / 2 and merged where they overlap (TSB-AD's new_sequence); ends inclusive. */
    static int[][] merged(int[] rs, int[] re, int window, int n) {
        int h = window / 2;
        int K = rs.length;
        List<int[]> out = new ArrayList<>();
        int a = Math.max(rs[0] - h, 0);
        for (int k = 0; k < K - 1; k++) {
            if (re[k] + h < rs[k + 1] - h) {
                out.add(new int[] {a, re[k] + h});
                a = rs[k + 1] - h;
            }
        }
        out.add(new int[] {a, Math.min(re[K - 1] + h, n - 1)});
        return out.toArray(new int[0][]);
    }
}
