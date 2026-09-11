package dev.repoagent.task;

import java.util.Map;

public final class RetryAfterParser {
    private RetryAfterParser() {}

    public static int retryAfterSeconds(Map<String, String> headers, int fallback) {
        String raw = headers.get("Retry-After");
        if (raw == null) {
            return fallback;
        }
        final int seconds;
        try {
            seconds = Integer.parseInt(raw);
        } catch (NumberFormatException exception) {
            throw new IllegalArgumentException("Retry-After must be an integer", exception);
        }
        if (seconds < 0) {
            throw new IllegalArgumentException("Retry-After must not be negative");
        }
        return seconds;
    }
}
