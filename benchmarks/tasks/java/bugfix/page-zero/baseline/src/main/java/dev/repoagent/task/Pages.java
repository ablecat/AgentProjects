package dev.repoagent.task;

public final class Pages {
    private Pages() {}

    public static int offset(int pageNumber, int pageSize) {
        if (pageNumber < 1 || pageSize < 1) {
            throw new IllegalArgumentException("page number and size must be positive");
        }
        return Math.multiplyExact(pageNumber - 1, pageSize);
    }
}
