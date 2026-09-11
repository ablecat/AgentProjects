package dev.repoagent.task;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;

class IssueDraftHiddenTest {
    @Test
    void draftsDoNotShareMutableLabels() {
        IssueDraft first = new IssueDraft();
        first.addLabel("urgent");
        IssueDraft second = new IssueDraft();
        assertEquals(List.of("triage"), second.labels());
    }
}
