package dev.repoagent.task;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class FileNamesHiddenTest {
    @Test
    void dotfileHasNoExtension() {
        assertTrue(FileNames.extension(".env").isEmpty());
    }

    @Test
    void trailingDotHasNoExtension() {
        assertTrue(FileNames.extension("report.").isEmpty());
    }
}
