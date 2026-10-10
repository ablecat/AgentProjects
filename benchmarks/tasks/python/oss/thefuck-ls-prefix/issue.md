# The ls rule rewrites unrelated commands

Upstream issue: https://github.com/nvbn/thefuck/issues/271

The `ls_lah` rule currently matches any command containing the substring `ls`.
For example, a failed `pacman -S binutils` command is incorrectly rewritten as
an `ls -lah` command. Commands such as `lsof` have the same false positive.

Restrict the rule to an `ls` command or an `ls` command with arguments, while
preserving the existing exclusion for calls that already contain ls options.
Add focused regression coverage for unrelated commands.
