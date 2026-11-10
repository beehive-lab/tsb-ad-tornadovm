/*
 * Copyright (c) 2026 Beehive lab
 * SPDX-License-Identifier: MIT
 */
package org.beehive.tsbad;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Runs the TornadoVM detectors and VUS metrics over TSB-AD files.
 *
 * <pre>
 * TsbadBench --dataset-dir DIR (--files a.csv,b.csv | --file-list list.csv) --out-dir OUT
 *            [--detectors Sub_PCA,Sub_KNN,KMeansAD_U,MatrixProfile] [--repeat 3] [--ref-dir REF] [--seed 2024]
 * </pre>
 *
 * Hyperparameters are TSB-AD's tuned univariate ones. For each file and detector: OUT/DET/STEM.f64 holds
 * the scores (little-endian float64) and a line of OUT/results.csv the timings and VUS-PR / VUS-ROC of the
 * min-max normalised scores. With --ref-dir, the VUS of REF/DET/STEM.f64 (scores of another
 * implementation) is computed too, so that the metric can be checked on identical input. The *_r6 columns
 * give VUS-PR after rounding the normalised scores to 6 decimals: VUS jumps when tied scores are split, so
 * rounding noise alone can move it on series with repeated patterns.
 */
public final class TsbadBench {

    public static void main(String[] args) throws IOException {
        Path dir = null;
        Path out = null;
        Path ref = null;
        List<String> files = new ArrayList<>();
        List<String> dets = List.of("Sub_PCA", "Sub_KNN", "KMeansAD_U", "MatrixProfile");
        int repeat = 3;
        long seed = 2024;
        for (int a = 0; a < args.length; a++) {
            switch (args[a]) {
                case "--dataset-dir" -> dir = Path.of(args[++a]);
                case "--out-dir" -> out = Path.of(args[++a]);
                case "--ref-dir" -> ref = Path.of(args[++a]);
                case "--files" -> files.addAll(Arrays.asList(args[++a].split(",")));
                case "--file-list" -> {
                    List<String> lines = Files.readAllLines(Path.of(args[++a]));
                    for (String l : lines.subList(1, lines.size())) {
                        if (!l.isBlank()) {
                            files.add(l.split(",")[0].trim());
                        }
                    }
                }
                case "--detectors" -> dets = Arrays.asList(args[++a].split(","));
                case "--repeat" -> repeat = Integer.parseInt(args[++a]);
                case "--seed" -> seed = Long.parseLong(args[++a]);
                default -> throw new IllegalArgumentException("unknown argument " + args[a]);
            }
        }
        if (dir == null || out == null || files.isEmpty()) {
            throw new IllegalArgumentException("--dataset-dir, --out-dir and --files or --file-list are required");
        }
        Files.createDirectories(out);
        Path csv = out.resolve("results.csv");
        boolean header = !Files.exists(csv);
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(csv, java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND))) {
            if (header) {
                w.println("file,detector,n,window,eval_window,first_ms,warm_ms,vus_ms,vus_roc,vus_pr,ref_vus_roc,ref_vus_pr,vus_pr_r6,ref_vus_pr_r6");
            }
            for (String f : files) {
                Series s = Series.readCsv(dir.resolve(f));
                String stem = f.endsWith(".csv") ? f.substring(0, f.length() - 4) : f;
                int evalWindow = Series.findLengthRank(s.x, 1);
                for (String det : dets) {
                    int window;
                    long t0 = System.nanoTime();
                    Detectors.Detector d;
                    switch (det) {
                        case "Sub_PCA" -> {
                            window = Series.findLengthRank(s.x, 1);
                            d = new Detectors.SubPca(s.x, window);
                        }
                        case "Sub_KNN" -> {
                            window = Series.findLengthRank(s.x, 2);
                            d = new Detectors.SubKnn(s.x, window, 50);
                        }
                        case "KMeansAD_U" -> {
                            window = Series.findLengthRank(s.x, 2);
                            d = new Detectors.KMeansAd(s.x, window, 10, seed);
                        }
                        case "MatrixProfile" -> {
                            window = Series.findLengthRank(s.x, 1);
                            d = new Detectors.MatrixProfile(s.x, window);
                        }
                        default -> throw new IllegalArgumentException("unknown detector " + det);
                    }
                    double[] score = d.score();
                    long first = System.nanoTime() - t0;
                    long warm = Long.MAX_VALUE;
                    for (int r = 1; r < repeat; r++) {
                        long t1 = System.nanoTime();
                        score = d.score();
                        warm = Math.min(warm, System.nanoTime() - t1);
                    }
                    d.close();
                    Path dd = out.resolve(det);
                    Files.createDirectories(dd);
                    write(dd.resolve(stem + ".f64"), score);
                    Vus v = Vus.compute(s.label, minMax(score), evalWindow);
                    String refCols = ",";
                    String r6 = String.format(Locale.ROOT, "%.10f,", Vus.compute(s.label, round6(minMax(score)), evalWindow).vusPr);
                    if (ref != null) {
                        Path rf = ref.resolve(det).resolve(stem + ".f64");
                        if (Files.exists(rf)) {
                            double[] rs = minMax(read(rf));
                            Vus rv = Vus.compute(s.label, rs, evalWindow);
                            refCols = String.format(Locale.ROOT, "%.10f,%.10f", rv.vusRoc, rv.vusPr);
                            r6 += String.format(Locale.ROOT, "%.10f", Vus.compute(s.label, round6(rs), evalWindow).vusPr);
                        }
                    }
                    String line = String.format(Locale.ROOT, "%s,%s,%d,%d,%d,%.2f,%.2f,%.2f,%.10f,%.10f,%s,%s", f, det, s.x.length, window, evalWindow, first / 1e6,
                            warm == Long.MAX_VALUE ? Double.NaN : warm / 1e6, v.nanos / 1e6, v.vusRoc, v.vusPr, refCols, r6);
                    w.println(line);
                    w.flush();
                    System.out.println("[tsbad] " + line);
                }
            }
        }
    }

    static double[] minMax(double[] s) {
        double lo = Double.POSITIVE_INFINITY;
        double hi = Double.NEGATIVE_INFINITY;
        for (double v : s) {
            lo = Math.min(lo, v);
            hi = Math.max(hi, v);
        }
        // sklearn's MinMaxScaler arithmetic, so that near ties stay ties exactly as in TSB-AD
        double range = hi - lo;
        double scale = 1.0 / (range == 0 ? 1.0 : range);
        double min = 0.0 - lo * scale;
        double[] o = new double[s.length];
        for (int i = 0; i < s.length; i++) {
            o[i] = s[i] * scale + min;
        }
        return o;
    }

    /** Rounds to 6 decimals (np.round), which merges scores that differ only by rounding noise. */
    static double[] round6(double[] s) {
        double[] o = new double[s.length];
        for (int i = 0; i < s.length; i++) {
            o[i] = Math.rint(s[i] * 1e6) / 1e6;
        }
        return o;
    }

    static void write(Path p, double[] a) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(8 * a.length).order(ByteOrder.LITTLE_ENDIAN);
        bb.asDoubleBuffer().put(a);
        Files.write(p, bb.array());
    }

    static double[] read(Path p) throws IOException {
        ByteBuffer bb = ByteBuffer.wrap(Files.readAllBytes(p)).order(ByteOrder.LITTLE_ENDIAN);
        double[] a = new double[bb.remaining() / 8];
        bb.asDoubleBuffer().get(a);
        return a;
    }
}
