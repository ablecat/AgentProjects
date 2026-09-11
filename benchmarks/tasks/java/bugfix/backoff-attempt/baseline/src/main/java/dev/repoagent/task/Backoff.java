package dev.repoagent.task;

public final class Backoff {
    private Backoff() {}

    public static long delayMillis(long baseMillis, int attempt) {
        if (baseMillis < 0 || attempt < 0 || attempt > 10) {
            throw new IllegalArgumentException("invalid backoff input");
        }
        return Math.multiplyExact(baseMillis, 1L << attempt);
    }
}
