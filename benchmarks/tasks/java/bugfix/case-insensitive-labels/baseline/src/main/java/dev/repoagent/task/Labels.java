package dev.repoagent.task;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

public final class Labels {
    private Labels() {}

    public static List<String> uniqueNormalized(List<String> labels) {
        List<String> result = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        for (String label : labels) {
            String normalized = label.strip();
            if (!normalized.isEmpty() && seen.add(normalized.toLowerCase(Locale.ROOT))) {
                result.add(normalized);
            }
        }
        return List.copyOf(result);
    }
}
