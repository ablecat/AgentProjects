package dev.repoagent.task;

import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import org.junit.jupiter.api.Test;

class RetryAfterParserHiddenTest {
    @Test
    void malformedValueIsRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> RetryAfterParser.retryAfterSeconds(Map.of("Retry-After", "soon"), 5));
    }

    @Test
    void negativeValueIsRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> RetryAfterParser.retryAfterSeconds(Map.of("Retry-After", "-1"), 5));
    }
}
