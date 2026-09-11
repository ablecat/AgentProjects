package dev.repoagent.task;

import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class PagesHiddenTest {
    @Test
    void zeroPageNumberIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> Pages.offset(0, 25));
    }
}
