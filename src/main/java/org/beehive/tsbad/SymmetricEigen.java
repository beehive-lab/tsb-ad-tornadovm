/*
 * Copyright (c) 2026 Beehive lab
 * SPDX-License-Identifier: MIT
 */
package org.beehive.tsbad;

/**
 * All eigenpairs of a small dense symmetric matrix in double precision: Householder tridiagonalisation and
 * implicit QL (the public-domain JAMA algorithm, after EISPACK tred2/tql2, with the orthogonal matrix
 * stored transposed so that the inner loops run along rows).
 */
final class SymmetricEigen {

    private SymmetricEigen() {
    }

    /**
     * @param a symmetric n x n matrix (overwritten)
     * @param values receives the eigenvalues in descending order
     * @return the unit eigenvectors as rows, in the order of {@code values}
     */
    static double[][] decompose(double[][] a, double[] values) {
        int n = a.length;
        double[] d = new double[n];
        double[] e = new double[n];
        tred2(a, d, e, n);
        tql2(a, d, e, n, true);
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (x, y) -> Double.compare(d[y], d[x]));
        double[][] out = new double[n][];
        for (int i = 0; i < n; i++) {
            values[i] = d[order[i]];
            out[i] = a[order[i]].clone();
        }
        return out;
    }

    private static void tred2(double[][] V, double[] d, double[] e, int n) {
        for (int j = 0; j < n; j++) {
            d[j] = V[j][n - 1];
        }
        for (int i = n - 1; i > 0; i--) {
            double scale = 0.0;
            double h = 0.0;
            for (int k = 0; k < i; k++) {
                scale += Math.abs(d[k]);
            }
            if (scale == 0.0) {
                e[i] = d[i - 1];
                for (int j = 0; j < i; j++) {
                    d[j] = V[j][i - 1];
                    V[j][i] = 0.0;
                    V[i][j] = 0.0;
                }
            } else {
                for (int k = 0; k < i; k++) {
                    d[k] /= scale;
                    h += d[k] * d[k];
                }
                double f = d[i - 1];
                double g = Math.sqrt(h);
                if (f > 0) {
                    g = -g;
                }
                e[i] = scale * g;
                h = h - f * g;
                d[i - 1] = f - g;
                for (int j = 0; j < i; j++) {
                    e[j] = 0.0;
                }
                for (int j = 0; j < i; j++) {
                    f = d[j];
                    V[i][j] = f;
                    g = e[j] + V[j][j] * f;
                    for (int k = j + 1; k <= i - 1; k++) {
                        g += V[j][k] * d[k];
                        e[k] += V[j][k] * f;
                    }
                    e[j] = g;
                }
                f = 0.0;
                for (int j = 0; j < i; j++) {
                    e[j] /= h;
                    f += e[j] * d[j];
                }
                double hh = f / (h + h);
                for (int j = 0; j < i; j++) {
                    e[j] -= hh * d[j];
                }
                for (int j = 0; j < i; j++) {
                    f = d[j];
                    g = e[j];
                    for (int k = j; k <= i - 1; k++) {
                        V[j][k] -= (f * e[k] + g * d[k]);
                    }
                    d[j] = V[j][i - 1];
                    V[j][i] = 0.0;
                }
            }
            d[i] = h;
        }
        for (int i = 0; i < n - 1; i++) {
            V[i][n - 1] = V[i][i];
            V[i][i] = 1.0;
            double h = d[i + 1];
            if (h != 0.0) {
                for (int k = 0; k <= i; k++) {
                    d[k] = V[i + 1][k] / h;
                }
                for (int j = 0; j <= i; j++) {
                    double g = 0.0;
                    for (int k = 0; k <= i; k++) {
                        g += V[i + 1][k] * V[j][k];
                    }
                    for (int k = 0; k <= i; k++) {
                        V[j][k] -= g * d[k];
                    }
                }
            }
            for (int k = 0; k <= i; k++) {
                V[i + 1][k] = 0.0;
            }
        }
        for (int j = 0; j < n; j++) {
            d[j] = V[j][n - 1];
            V[j][n - 1] = 0.0;
        }
        V[n - 1][n - 1] = 1.0;
        e[0] = 0.0;
    }

    private static void tql2(double[][] V, double[] d, double[] e, int n, boolean vectors) {
        for (int i = 1; i < n; i++) {
            e[i - 1] = e[i];
        }
        e[n - 1] = 0.0;
        double f = 0.0;
        double tst1 = 0.0;
        double eps = Math.pow(2.0, -52.0);
        for (int l = 0; l < n; l++) {
            tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
            int m = l;
            while (m < n) {
                if (Math.abs(e[m]) <= eps * tst1) {
                    break;
                }
                m++;
            }
            if (m == n) {
                m = n - 1;
            }
            if (m > l) {
                int iter = 0;
                do {
                    iter++;
                    double g = d[l];
                    double p = (d[l + 1] - g) / (2.0 * e[l]);
                    double r = Math.hypot(p, 1.0);
                    if (p < 0) {
                        r = -r;
                    }
                    d[l] = e[l] / (p + r);
                    d[l + 1] = e[l] * (p + r);
                    double dl1 = d[l + 1];
                    double h = g - d[l];
                    for (int i = l + 2; i < n; i++) {
                        d[i] -= h;
                    }
                    f = f + h;
                    p = d[m];
                    double c = 1.0;
                    double c2 = c;
                    double c3 = c;
                    double el1 = e[l + 1];
                    double s = 0.0;
                    double s2 = 0.0;
                    for (int i = m - 1; i >= l; i--) {
                        c3 = c2;
                        c2 = c;
                        s2 = s;
                        g = c * e[i];
                        h = c * p;
                        r = Math.hypot(p, e[i]);
                        e[i + 1] = s * r;
                        s = e[i] / r;
                        c = p / r;
                        p = c * d[i] - s * g;
                        d[i + 1] = h + s * (c * g + s * d[i]);
                        for (int k = 0; vectors && k < n; k++) {
                            h = V[i + 1][k];
                            V[i + 1][k] = s * V[i][k] + c * h;
                            V[i][k] = c * V[i][k] - s * h;
                        }
                    }
                    p = -s * s2 * c3 * el1 * e[l] / dl1;
                    e[l] = s * p;
                    d[l] = c * p;
                } while (Math.abs(e[l]) > eps * tst1 && iter < 300);
            }
            d[l] = d[l] + f;
            e[l] = 0.0;
        }
    }
}
