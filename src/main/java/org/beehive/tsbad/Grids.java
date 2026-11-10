/*
 * Copyright (c) 2026 Beehive lab
 * SPDX-License-Identifier: MIT
 */
package org.beehive.tsbad;

import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;

final class Grids {

    private Grids() {
    }

    static WorkerGrid grid1(long global, int local) {
        WorkerGrid g = new WorkerGrid1D(Math.toIntExact(global));
        g.setLocalWork(local, 1, 1);
        return g;
    }

    static long roundUp(long n, int local) {
        return ((n + local - 1) / local) * local;
    }

    static int roundUpInt(int n, int local) {
        return Math.toIntExact(roundUp(n, local));
    }
}
