package com.mcoder.jge.util;

import java.util.LinkedList;
import java.util.concurrent.*;

public class ThreadPool {
    private static ThreadPool instance;

    private final int numThreads;
    private final ExecutorService executor;
    private final LinkedList<Future<?>> futures;

    private ThreadPool() {
        numThreads = Runtime.getRuntime().availableProcessors();
        executor = Executors.newFixedThreadPool(numThreads);
        futures = new LinkedList<>();
    }

    public void executeInParallel(int start, int end, Task task) {
        int n = end-start;
        final int blockSize = n/numThreads;
        final int carry = n%numThreads;

        for (int i = 0; i < numThreads; i++) {
            final int j = i;
            futures.add(executor.submit(() -> {
                int block = blockSize;
                int step;
                if (j < carry) {
                    block++;
                    step = 0;
                } else step = carry;

                for (int k = 0; k < block; k++)
                    task.run(k + block * j + step + start);
            }));
        }

        waitForLastSubmittedTasks(numThreads);
    }

    public void executeInParallel(int n, Task task) {
        executeInParallel(0, n, task);
    }

    public void execute(Runnable runnable) {
        futures.add(executor.submit(runnable));
    }

    public void waitForAllTasks() {
        waitForLastSubmittedTasks(futures.size());
    }

    private void waitForLastSubmittedTasks(int n) {
        for (int i = 0; i < n; i++)
            try {
                futures.removeLast().get();
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException(e);
            }
    }

    public static ThreadPool getInstance() {
        if (instance == null)
            instance = new ThreadPool();
        return instance;
    }

    public interface Task {
        void run(int i);
    }
}
