package com.mcoder.jge.util;

import java.util.concurrent.*;

public class ThreadPool {
    private static final int numThreads = 12; //Runtime.getRuntime().availableProcessors();
    private static final ExecutorService executor = Executors.newFixedThreadPool(numThreads);

    public static void executeInParallel(int n, Task task) {
        Future<?>[] futures = new Future[numThreads];
        final int blockSize = n/numThreads;
        final int carry = n%numThreads;

        for (int i = 0; i < numThreads; i++) {
            final int j = i;
            futures[i] = executor.submit(() -> {
                int block = blockSize;
                int step;
                if (j < carry) {
                    block++;
                    step = 0;
                } else step = carry;

                for (int k = 0; k < block; k++)
                    task.run(k+block*j+step);
            });
        }

        try {
            for (Future<?> future : futures)
                future.get();
        } catch (ExecutionException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    public interface Task {
        void run(int i);
    }
}
