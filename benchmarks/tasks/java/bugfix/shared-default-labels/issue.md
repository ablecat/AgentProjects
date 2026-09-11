# Issue drafts share mutable default labels

Adding a label to one `IssueDraft` changes the starting labels of drafts created
later. `Defaults` defines the initial values, but every draft must own an
independent mutable copy. The public `labels()` snapshot behavior must remain
unchanged.

Remove the cross-instance state leak and add a regression test that creates two
drafts. Do not replace the list API or add a dependency.
