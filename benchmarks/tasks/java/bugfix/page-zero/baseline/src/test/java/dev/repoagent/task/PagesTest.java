package dev.repoagent.task;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class PagesTest {
    @Test
    void firstPageStartsAtZero() {
        assertEquals(0, Pages.offset(1, 25));
    }

    @Test
    void laterPageUsesOneBasedNumbering() {
        assertEquals(50, Pages.offset(3, 25));
    }

    @Test
    void rejectsZeroPageSize() {
        assertThrows(IllegalArgumentException.class, () -> Pages.offset(1, 0));
    }
}
