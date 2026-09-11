package dev.repoagent.task;

import java.util.ArrayList;
import java.util.List;

public final class IssueDraft {
    private final List<String> labels = new ArrayList<>(Defaults.LABELS);

    public void addLabel(String label) {
        labels.add(label);
    }

    public List<String> labels() {
        return List.copyOf(labels);
    }
}
