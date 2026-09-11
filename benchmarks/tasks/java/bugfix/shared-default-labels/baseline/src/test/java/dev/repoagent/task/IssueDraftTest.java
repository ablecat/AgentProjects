package dev.repoagent.task;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;

class IssueDraftTest {
    @Test
    void startsWithTriageAndAcceptsAnotherLabel() {
        IssueDraft draft = new IssueDraft();
        draft.addLabel("bug");
        assertEquals(List.of("triage", "bug"), draft.labels());
    }
}
