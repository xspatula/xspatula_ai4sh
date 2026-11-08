<!-- Loaded only when working under src/. Moved from .claude/CLAUDE.md on 2026-11-08. -->

# Composed FK names

An FK parameter pointing at an auto-named record (e.g. `observation_log_id__observation_log_name`) may give the
parts of the composed name as alias or tag instead of name (`ai4sh-dk-foulum 2024@ai4sh-metabarcod` for
`…@ai4sh metabarcoding`). When the direct lookup fails, `_Resolve_composed_name` in `src/postgres/pg_common.py`
splits the value by every `manage_<table>` rule in `process.process_parameter_auto_name`, resolves each FK part
(name/tag/alias/unique column, recursively for nested composed parts), rebuilds and re-searches. Generic: any
table with an auto_name rule is covered. Hooked into `_Check_get_foreign_key` and `find_record`
(`src/lib/structure.py`); success prints the canonical name at verbose ≥ 1, more than one match is an error.
Separators `_`, `@` and `#` are reserved in name/alias/tag. Optional concat arguments (`site?`) are dropped with their
separator when empty, `'none'` or 0 — e.g. observation_log name `'%s@%s#%s#%s'` (sampling_log@provision#preparation alias#rescan_nr; tag uses `.` instead of `#`,
added 2026-10-17). `pg_common.py` and `structure.py` are kept byte-identical with `xspatula_lucas` and `xspatula_core`.
