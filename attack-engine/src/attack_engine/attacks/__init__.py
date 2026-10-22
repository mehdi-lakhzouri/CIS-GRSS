"""Attack transformations and in-memory execution (Phases 3-4).

Transformations turn a clean observation window into candidate modified values and
report which positions actually changed. They never touch the filesystem, the CLI
or global state, and never mutate the source data. Dataset export, ground-truth
files and provenance bundles are deferred to a later phase.
"""
