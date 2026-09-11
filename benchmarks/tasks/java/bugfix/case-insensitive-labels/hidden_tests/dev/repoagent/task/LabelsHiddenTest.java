package dev.repoagent.task;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;

class LabelsHiddenTest {
    @Test
    void duplicateLabelsIgnoreCase() {
        assertEquals(
                List.of("Alpha", "beta"),
                Labels.uniqueNormalized(List.of("Alpha", "alpha", "beta", "BETA")));
    }
}
