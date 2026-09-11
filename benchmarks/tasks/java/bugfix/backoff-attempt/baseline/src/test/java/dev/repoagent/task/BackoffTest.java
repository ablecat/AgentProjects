package dev.repoagent.task;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class BackoffTest {
    @Test
    void firstAttemptUsesBaseDelay() {
        assertEquals(250, Backoff.delayMillis(250, 0));
    }

    @Test
    void rejectsNegativeAttempt() {
        assertThrows(IllegalArgumentException.class, () -> Backoff.delayMillis(250, -1));
    }
}
