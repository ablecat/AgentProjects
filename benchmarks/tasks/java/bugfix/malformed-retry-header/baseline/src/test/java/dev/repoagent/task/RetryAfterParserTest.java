package dev.repoagent.task;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Map;
import org.junit.jupiter.api.Test;

class RetryAfterParserTest {
    @Test
    void parsesValidHeader() {
        assertEquals(12, RetryAfterParser.retryAfterSeconds(Map.of("Retry-After", "12"), 3));
    }

    @Test
    void usesFallbackWhenHeaderIsMissing() {
        assertEquals(3, RetryAfterParser.retryAfterSeconds(Map.of(), 3));
    }
}
