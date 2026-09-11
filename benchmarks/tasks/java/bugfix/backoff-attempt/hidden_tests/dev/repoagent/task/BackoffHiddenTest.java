package dev.repoagent.task;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class BackoffHiddenTest {
    @Test
    void laterAttemptUsesMatchingExponent() {
        assertEquals(2_000, Backoff.delayMillis(250, 3));
    }
}
