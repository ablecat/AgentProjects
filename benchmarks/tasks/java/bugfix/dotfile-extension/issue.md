# Dotfiles are treated as file extensions

`FileNames.extension()` reports `env` for `.env` and can report an empty string
for names ending in a dot. A leading dot that is the first character is a
dotfile marker, not an extension separator, and a trailing separator has no
extension after it.

Correct these boundary cases and add focused regression coverage without
changing ordinary mixed-case extension handling.
