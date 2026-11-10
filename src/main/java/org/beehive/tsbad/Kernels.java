/*
 * Copyright (c) 2026 Beehive lab
 * SPDX-License-Identifier: MIT
 */
package org.beehive.tsbad;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Device kernels. Every subsequence detector reads its windows straight from the series: window {@code i}
 * of length {@code m} is {@code x[i .. i + m - 1]}, so no window matrix is ever stored.
 */
public final class Kernels {

    /** Tile edge of the VUS accumulation. */
    public static final int VUS_TILE = 16;
    /** Diagonals (threads) and rows per matrix-profile workgroup. */
    public static final int MP_T = 64;
    /** Query rows (threads) and candidate columns per k-NN tile. */
    public static final int KNN_T = 32;
    /** float32 unit roundoff, and the largest accumulated recurrence error (in correlation) before a reseed. */
    public static final float MP_EPS = 6.0e-8f;
    public static final float MP_TOL = 1.0e-5f;
    /** Largest window the k-NN kernel's local buffers hold. */
    public static final int KNN_MAX_M = 320;
    /** Largest k of the k-NN kernel. */
    public static final int KNN_MAX_K = 100;

    private Kernels() {
    }

    // ---------------------------------------------------------------- VUS

    /**
     * Extended labels of the normal points near anomalies, for every buffer window w = 0 .. W-1:
     * ext[w, r] = min(1, sum over anomaly runs of sqrt(1 - d / w)) for the distance d <= w / 2 of point
     * rIdx[r] past the end or before the start of the run; 0 for anomalous points.
     */
    public static void vusExt(KernelContext ctx, IntArray segS, IntArray segE, IntArray rIdx, IntArray normal, IntArray segLo, FloatArray ext, int nSeg, int W, int R, int hMax) {
        int id = ctx.globalIdx;
        if (id < W * R) {
            int w = id / R;
            int r = id - w * R;
            float v = 0.0f;
            if (normal.get(r) == 1 && w > 1) {
                int t = rIdx.get(r);
                int hw = w / 2;
                float fw = (float) w;
                float sum = 0.0f;
                int k = segLo.get(r);
                while (k < nSeg && segS.get(k) <= t + hMax) {
                    int e = segE.get(k);
                    int s = segS.get(k);
                    if (e < t && t - e <= hw) {
                        sum += TornadoMath.sqrt(1.0f - (float) (t - e) / fw);
                    }
                    if (s > t && s - t <= hw) {
                        sum += TornadoMath.sqrt(1.0f - (float) (s - t) / fw);
                    }
                    k++;
                }
                v = TornadoMath.min(sum, 1.0f);
            }
            ext.set(id, v);
        }
    }

    /**
     * acc[w, j] = sum_r ext[w, r] * [score(r) >= thr(j)], tiled over r in local memory. Thresholds descend
     * with j, so the prediction is j >= firstJ[r], the first threshold at or below the point's score
     * (found on the host in double precision).
     */
    public static void vusAccumulate(KernelContext ctx, FloatArray ext, IntArray firstJ, FloatArray acc, int W, int R, int J) {
        float[] eTile = ctx.allocateFloatLocalArray(VUS_TILE * VUS_TILE);
        int[] jTile = ctx.allocateIntLocalArray(VUS_TILE);
        int lj = ctx.localIdx;
        int lw = ctx.localIdy;
        int j = ctx.globalIdx;
        int w = ctx.globalIdy;
        float sum = 0.0f;
        for (int r0 = 0; r0 < R; r0 += VUS_TILE) {
            int r = r0 + lj;
            float e = 0.0f;
            if (w < W && r < R) {
                e = ext.get(w * R + r);
            }
            eTile[lw * VUS_TILE + lj] = e;
            if (lw == 0) {
                int f = J;
                if (r < R) {
                    f = firstJ.get(r);
                }
                jTile[lj] = f;
            }
            ctx.localBarrier();
            for (int u = 0; u < VUS_TILE; u++) {
                if (j >= jTile[u]) {
                    sum += eTile[lw * VUS_TILE + u];
                }
            }
            ctx.localBarrier();
        }
        if (w < W && j < J) {
            acc.set(w * J + j, sum);
        }
    }

    // ---------------------------------------------------------------- matrix profile

    /**
     * Largest Pearson correlation of each subsequence with any subsequence outside its exclusion zone.
     * A workgroup owns MP_T consecutive rows and sweeps all diagonals in bands of MP_T: thread t walks
     * diagonal d down the rows of the block, seeding the centred dot product directly and updating it with
     * the SCAMP recurrence cov[i+1, j+1] = cov[i, j] + df[i+1] dg[j+1] + df[j+1] dg[i+1]. A running bound
     * on the recurrence's rounding error triggers a fresh seed where float32 would lose the correlation
     * (flat subsequences after large jumps). The correlations
     * of a band go to local memory, where each thread then reduces one row and keeps the best column.
     *
     * @param xc series minus its mean, as float; xlo the remainder (x - xc), so that subsequences with a
     *     spread far below the series' magnitude keep their precision
     * @param mu subsequence means (of xc), with remainders mulo
     * @param inv 1 / ||subsequence - mean||, 0 for constant subsequences
     */
    public static void mpRowMax(KernelContext ctx, FloatArray xc, FloatArray xlo, FloatArray mu, FloatArray mulo, FloatArray inv, FloatArray df, FloatArray dg, FloatArray best, IntArray bestIdx, int N, int m, int excl) {
        float[] tile = ctx.allocateFloatLocalArray(MP_T * (MP_T + 1));
        int t = ctx.localIdx;
        int i0 = ctx.groupIdx * MP_T;
        int rows = TornadoMath.min(MP_T, N - i0);
        float rowBest = -2.0f;
        int rowArg = -1;
        // diagonals that meet a row of this block: d in [-(i0 + rows - 1), N - 1 - i0]
        int dFirst = -(i0 + rows - 1);
        int dLast = N - 1 - i0;
        for (int dBand = dFirst; dBand <= dLast; dBand += MP_T) {
            int d = dBand + t;
            // rows s of the block where column i0 + s + d is inside the series
            int sLo = TornadoMath.max(0, -d - i0);
            int sHi = TornadoMath.min(rows, N - i0 - d);
            boolean live = d <= dLast && (d > excl || d < -excl) && sLo < sHi;
            float cov = 0.0f;
            float err = 0.0f;
            for (int s = 0; s < MP_T; s++) {
                float c = -2.0f;
                if (live && s >= sLo && s < sHi) {
                    int i = i0 + s;
                    int j = i + d;
                    float ii = inv.get(i);
                    float ij = inv.get(j);
                    float inc = 0.0f;
                    if (s > sLo) {
                        inc = df.get(i) * dg.get(j) + df.get(j) * dg.get(i);
                    }
                    // rounding error of the recurrence so far, in correlation units; reseed when too large
                    float e = err + MP_EPS * (TornadoMath.abs(cov) + TornadoMath.abs(inc));
                    if (s == sLo || e * ii * ij > MP_TOL) {
                        float mi = mu.get(i);
                        float mj = mu.get(j);
                        float li = mulo.get(i);
                        float lj = mulo.get(j);
                        float acc = 0.0f;
                        for (int a = 0; a < m; a++) {
                            float ci = (xc.get(i + a) - mi) + (xlo.get(i + a) - li);
                            float cj = (xc.get(j + a) - mj) + (xlo.get(j + a) - lj);
                            acc += ci * cj;
                        }
                        cov = acc;
                        err = 0.0f;
                    } else {
                        cov += inc;
                        err = e;
                    }
                    c = cov * ii * ij;
                }
                tile[s * (MP_T + 1) + t] = c;
            }
            ctx.localBarrier();
            for (int u = 0; u < MP_T; u++) {
                float c = tile[t * (MP_T + 1) + u];
                if (c > rowBest) {
                    rowBest = c;
                    rowArg = i0 + t + dBand + u;
                }
            }
            ctx.localBarrier();
        }
        if (t < rows) {
            best.set(i0 + t, rowBest);
            bestIdx.set(i0 + t, rowArg);
        }
    }

    // ---------------------------------------------------------------- Sub-PCA

    /**
     * Sub-PCA score of window i: sum_c invW[c] * ||z_i - v_c|| with z_i the column-standardised window
     * (colInv = 0 drops a column) and v_c the unit principal directions.
     */
    public static void pcaScore(KernelContext ctx, FloatArray xc, FloatArray colMu, FloatArray colInv, FloatArray v, FloatArray invW, FloatArray out, int N, int m, int k) {
        int i = ctx.globalIdx;
        if (i < N) {
            float nrm = 0.0f;
            for (int a = 0; a < m; a++) {
                float z = (xc.get(i + a) - colMu.get(a)) * colInv.get(a);
                nrm += z * z;
            }
            float score = 0.0f;
            for (int c = 0; c < k; c++) {
                float dot = 0.0f;
                for (int a = 0; a < m; a++) {
                    float z = (xc.get(i + a) - colMu.get(a)) * colInv.get(a);
                    dot += z * v.get(c * m + a);
                }
                score += invW.get(c) * TornadoMath.sqrt(TornadoMath.max(nrm + 1.0f - 2.0f * dot, 0.0f));
            }
            out.set(i, score);
        }
    }

    // ---------------------------------------------------------------- KMeansAD

    /**
     * Squared distance of every row-standardised window to each of nc centres: out[c * N + i]. Series and
     * window means come as float pairs (value, remainder), see {@link #mpRowMax}.
     */
    public static void kmDist(KernelContext ctx, FloatArray xc, FloatArray xlo, FloatArray rowMu, FloatArray rowMuLo, FloatArray rowInv, FloatArray centres, FloatArray out, int N, int m, int nc) {
        int i = ctx.globalIdx;
        if (i < N) {
            float mi = rowMu.get(i);
            float li = rowMuLo.get(i);
            float si = rowInv.get(i);
            for (int c = 0; c < nc; c++) {
                float d = 0.0f;
                for (int a = 0; a < m; a++) {
                    float diff = ((xc.get(i + a) - mi) + (xlo.get(i + a) - li)) * si - centres.get(c * m + a);
                    d += diff * diff;
                }
                out.set(c * N + i, d);
            }
        }
    }

    /** Index and squared distance of the closest of k centres for every window. */
    public static void kmAssign(KernelContext ctx, FloatArray xc, FloatArray xlo, FloatArray rowMu, FloatArray rowMuLo, FloatArray rowInv, FloatArray centres, IntArray labels, FloatArray dist, int N, int m, int k) {
        int i = ctx.globalIdx;
        if (i < N) {
            float mi = rowMu.get(i);
            float li = rowMuLo.get(i);
            float si = rowInv.get(i);
            float bd = Float.POSITIVE_INFINITY;
            int bc = 0;
            for (int c = 0; c < k; c++) {
                float d = 0.0f;
                for (int a = 0; a < m; a++) {
                    float diff = ((xc.get(i + a) - mi) + (xlo.get(i + a) - li)) * si - centres.get(c * m + a);
                    d += diff * diff;
                }
                if (d < bd) {
                    bd = d;
                    bc = c;
                }
            }
            labels.set(i, bc);
            dist.set(i, bd);
        }
    }

    /** Per-chunk sums of the windows of each cluster: partial[(p * k + c) * m + a]. */
    public static void kmSums(KernelContext ctx, FloatArray xc, FloatArray xlo, FloatArray rowMu, FloatArray rowMuLo, FloatArray rowInv, IntArray labels, FloatArray partial, int N, int m, int k, int chunks) {
        int id = ctx.globalIdx;
        if (id < chunks * k * m) {
            int p = id / (k * m);
            int rem = id - p * k * m;
            int c = rem / m;
            int a = rem - c * m;
            int len = (N + chunks - 1) / chunks;
            int lo = p * len;
            int hi = TornadoMath.min(N, lo + len);
            float s = 0.0f;
            for (int i = lo; i < hi; i++) {
                if (labels.get(i) == c) {
                    s += ((xc.get(i + a) - rowMu.get(i)) + (xlo.get(i + a) - rowMuLo.get(i))) * rowInv.get(i);
                }
            }
            partial.set(id, s);
        }
    }

    // ---------------------------------------------------------------- Sub-KNN

    /**
     * Squared distances of each window to its k nearest other windows (ascending), in the column-
     * standardised space: d^2(q, j) = sum_a inv2[a] (x[q + a] - x[j + a])^2. A workgroup owns KNN_T
     * consecutive query windows; candidate windows come in tiles of KNN_T, whose series stretch (KNN_T +
     * m - 1 points) sits in local memory together with the stretch of the queries. Each thread keeps its
     * sorted k best (distance and index) in local memory and abandons a candidate once its partial sum
 * passes the current k-th.
     */
    public static void knnTopK(KernelContext ctx, FloatArray xc, FloatArray inv2, FloatArray out, IntArray outIdx, int N, int m, int k) {
        float[] qs = ctx.allocateFloatLocalArray(KNN_T + KNN_MAX_M);
        float[] cs = ctx.allocateFloatLocalArray(KNN_T + KNN_MAX_M);
        float[] w2 = ctx.allocateFloatLocalArray(KNN_MAX_M);
        float[] top = ctx.allocateFloatLocalArray(KNN_T * KNN_MAX_K);
        int[] topIdx = ctx.allocateIntLocalArray(KNN_T * KNN_MAX_K);
        int t = ctx.localIdx;
        int q0 = ctx.groupIdx * KNN_T;
        int q = q0 + t;
        int span = KNN_T + m - 1;
        int len = N + m - 1;
        for (int u = t; u < span; u += KNN_T) {
            float v = 0.0f;
            if (q0 + u < len) {
                v = xc.get(q0 + u);
            }
            qs[u] = v;
        }
        for (int u = t; u < m; u += KNN_T) {
            w2[u] = inv2.get(u);
        }
        for (int u = 0; u < k; u++) {
            top[t * KNN_MAX_K + u] = Float.POSITIVE_INFINITY;
            topIdx[t * KNN_MAX_K + u] = -1;
        }
        float kth = Float.POSITIVE_INFINITY;
        for (int j0 = 0; j0 < N; j0 += KNN_T) {
            ctx.localBarrier();
            for (int u = t; u < span; u += KNN_T) {
                float v = 0.0f;
                if (j0 + u < len) {
                    v = xc.get(j0 + u);
                }
                cs[u] = v;
            }
            ctx.localBarrier();
            if (q < N) {
                int cols = TornadoMath.min(KNN_T, N - j0);
                for (int jj = 0; jj < cols; jj++) {
                    if (j0 + jj != q) {
                        float d = 0.0f;
                        int a = 0;
                        while (a < m) {
                            int stop = TornadoMath.min(m, a + 16);
                            for (int b = a; b < stop; b++) {
                                float diff = qs[t + b] - cs[jj + b];
                                d += w2[b] * diff * diff;
                            }
                            a = stop;
                            if (d >= kth) {
                                a = m + 1;
                            }
                        }
                        if (a == m && d < kth) {
                            // insert into the sorted list, dropping the old k-th
                            int pos = k - 1;
                            while (pos > 0 && top[t * KNN_MAX_K + pos - 1] > d) {
                                top[t * KNN_MAX_K + pos] = top[t * KNN_MAX_K + pos - 1];
                                topIdx[t * KNN_MAX_K + pos] = topIdx[t * KNN_MAX_K + pos - 1];
                                pos--;
                            }
                            top[t * KNN_MAX_K + pos] = d;
                            topIdx[t * KNN_MAX_K + pos] = j0 + jj;
                            kth = top[t * KNN_MAX_K + k - 1];
                        }
                    }
                }
            }
        }
        if (q < N) {
            for (int u = 0; u < k; u++) {
                out.set(q * k + u, top[t * KNN_MAX_K + u]);
                outIdx.set(q * k + u, topIdx[t * KNN_MAX_K + u]);
            }
        }
    }
}
