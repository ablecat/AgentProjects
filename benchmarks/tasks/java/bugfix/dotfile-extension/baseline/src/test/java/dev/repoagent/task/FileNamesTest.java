package dev.repoagent.task;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class FileNamesTest {
    @Test
    void returnsLowercaseExtension() {
        assertEquals("json", FileNames.extension("report.JSON").orElseThrow());
    }

    @Test
    void returnsEmptyWhenNoDotExists() {
        assertTrue(FileNames.extension("README").isEmpty());
    }
}
