/*
 * Copyright (c) 2026 Beehive lab
 * SPDX-License-Identifier: MIT
 */
package org.beehive.tsbad;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** A TSB-AD series: values of the first column and the 0/1 labels, as read by TSB-AD's scripts. */
public final class Series {

    public final double[] x;
    public final int[] label;

    public Series(double[] x, int[] label) {
        this.x = x;
        this.label = label;
    }

    /** Reads a TSB-AD CSV file (header, value columns, Label last); rows with a missing value are dropped. */
    public static Series readCsv(Path file) throws IOException {
        List<String> lines = Files.readAllLines(file);
        double[] x = new double[lines.size()];
        int[] y = new int[lines.size()];
        int n = 0;
        for (int i = 1; i < lines.size(); i++) {
            String[] f = lines.get(i).split(",", -1);
            if (f.length < 2 || f[0].isEmpty() || f[f.length - 1].isEmpty()) {
                continue;
            }
            boolean missing = false;
            for (String s : f) {
                if (s.isEmpty() || s.equalsIgnoreCase("nan")) {
                    missing = true;
                    break;
                }
            }
            if (missing) {
                continue;
            }
            x[n] = Double.parseDouble(f[0]);
            y[n] = (int) Double.parseDouble(f[f.length - 1]);
            n++;
        }
        return new Series(Arrays.copyOf(x, n), Arrays.copyOf(y, n));
    }

    /**
     * Period estimate of TSB-AD's {@code find_length_rank}: local maxima of the autocorrelation (lags 3 to
     * 400 of the first 20000 points), ranked by height; 125 when none is found or the period exceeds 300.
     */
    public static int findLengthRank(double[] data, int rank) {
        if (rank == 0) {
            return 1;
        }
        int n = Math.min(20000, data.length);
        int base = 3;
        int nlags = Math.min(400, n - 1);
        double mean = 0;
        for (int i = 0; i < n; i++) {
            mean += data[i];
        }
        mean /= n;
        double[] c = new double[n];
        for (int i = 0; i < n; i++) {
            c[i] = data[i] - mean;
        }
        double[] acf = new double[nlags + 1];
        for (int k = 0; k <= nlags; k++) {
            double s = 0;
            for (int t = 0; t + k < n; t++) {
                s += c[t] * c[t + k];
            }
            acf[k] = s;
        }
        double c0 = acf[0];
        int len = nlags + 1 - base;
        double[] auto = new double[Math.max(len, 0)];
        for (int k = 0; k < len; k++) {
            auto[k] = acf[k + base] / c0;
        }
        List<Integer> localMax = new ArrayList<>();
        for (int k = 1; k + 1 < auto.length; k++) {
            if (auto[k] > auto[k - 1] && auto[k] > auto[k + 1]) {
                localMax.add(k);
            }
        }
        if (localMax.isEmpty()) {
            return 125;
        }
        // np.argsort(values)[::-1]: indices into localMax by descending autocorrelation
        Integer[] order = new Integer[localMax.size()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (p, q) -> {
            int cmp = Double.compare(auto[localMax.get(q)], auto[localMax.get(p)]);
            return cmp != 0 ? cmp : Integer.compare(q, p);
        });
        int pick = order[0];
        if (rank == 2) {
            for (int i = 1; i < order.length; i++) {
                if (order[i] > order[0]) {
                    pick = order[i];
                    break;
                }
            }
        } else if (rank == 3) {
            int idTmp = -1;
            for (int i = 1; i < order.length; i++) {
                if (order[i] > order[0]) {
                    idTmp = order[i];
                    break;
                }
            }
            if (idTmp < 0 || idTmp >= order.length) {
                return 125;
            }
            for (int i = idTmp; i < order.length; i++) {
                if (order[i] > order[idTmp]) {
                    pick = order[i];
                    break;
                }
            }
        }
        int period = localMax.get(pick) + base;
        return period > 300 ? 125 : period;
    }
}
