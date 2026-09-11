package dev.repoagent.task;

import java.util.Locale;
import java.util.Objects;
import java.util.Optional;

public final class FileNames {
    private FileNames() {}

    public static Optional<String> extension(String name) {
        Objects.requireNonNull(name, "name");
        int dot = name.lastIndexOf('.');
        if (dot <= 0 || dot == name.length() - 1) {
            return Optional.empty();
        }
        return Optional.of(name.substring(dot + 1).toLowerCase(Locale.ROOT));
    }
}
