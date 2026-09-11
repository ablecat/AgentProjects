package dev.repoagent.task;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;

class LabelsTest {
    @Test
    void trimsLabelsAndDropsBlankValues() {
        assertEquals(
                List.of("alpha", "beta"),
                Labels.uniqueNormalized(List.of(" alpha ", "", "  ", "beta")));
    }

    @Test
    void preservesDistinctInputOrder() {
        assertEquals(
                List.of("second", "first"),
                Labels.uniqueNormalized(List.of("second", "first")));
    }
}
