"""Convert the AI4SH eDNA master workbook to xspatula import files.

Usage (from any folder):

    python edna_final_results_to_xspatula.py [source.xlsx] [--out-dir DIR] [--report-dir DIR]

Input
-----
source   master workbook as delivered by the eDNA laboratory, 3 sheets:
         Summary          - 3 header rows, one row per sample, 19 indicators
         ASV_prokaryotes  - one row per ASV (lineage label), one column per sample,
                            relative abundance; plus derived rows (simpson_alpha,
                            shannon_alpha) that are not ASVs
         ASV_fungi        - as ASV_prokaryotes
         Default: ../eDNA/source/AI4SH_eDNA_Final results.xlsx - put an updated master
         there (same file name) and rerun.

edna/    translators and template, maintained by hand (the lab's sample names are
         "home made" and must be mapped to xspatula names):
         eDNA_sample_name_translator.csv   summary_name -> sampling log, sample name,
                                           observed_at (analysis date), exclude (reason;
                                           blank = include)
         eDNA_indicator_translator.csv     Summary header (3 rows) -> xspatula column
         eDNA_observation_log_template.csv values shared by every eDNA observation log

Output (--out-dir, default ../eDNA/excel)
------
AI4SH_observation_log_eDNA.xlsx     one observation log per sampling log
AI4SH_observation_eDNA.xlsx         one row per sample, 19 indicator columns
taxa_prokaryotes_singlecolumn.xlsx  lineages of all prokaryote ASVs (Stage 4a)
taxa_fungi_singlecolumn.xlsx        lineages of all fungal ASVs (Stage 4a)
AI4SH_asv_prokaryotes_eDNA.xlsx     ASV abundance per sample, non-zero only (Stage 4b)
AI4SH_asv_fungi_eDNA.xlsx           as above for fungi

The ASV files are in long format: observation_log_id__observation_log_name,
sample_id__sample_name, asv, taxa, rel_abundance, read_count. The source has no ASV
ids or sequences, so asv is <group>-<row number in the source sheet>; it is only
stable as long as the laboratory keeps the row order. read_count is the rarefied
read count (rel_abundance x rarefaction depth), filled only when every abundance of
the sample is an exact multiple of 1/depth.

Checks (--report-dir, default ../eDNA/report)
------
Every diversity index in the Summary is recomputed from the ASV sheets (observed
richness, chao1, dominance, pielou evenness, shannon (log2), simpson) and compared,
as are the derived rows of the ASV sheets. Functional predictions (FAPROTAX /
FUNGuild) cannot be recomputed from the delivered data and are not checked.
Differences are reported, not fatal. The run aborts (exit code 1, no files written)
if a sample or indicator cannot be mapped.
"""

import argparse
import csv
import math
import os
import sys
from collections import OrderedDict

import numpy as np
import openpyxl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSLATOR_DIR = os.path.join(SCRIPT_DIR, "edna")
DEFAULT_SOURCE = os.path.join(SCRIPT_DIR, "edna", "excel", "AI4SH_eDNA_Final results.xlsx")
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "..", "eDNA", "excel")
DEFAULT_REPORT_DIR = os.path.join(SCRIPT_DIR, "edna", "report")

SAMPLE_TRANSLATOR_CSV = "eDNA_sample_name_translator.csv"
INDICATOR_TRANSLATOR_CSV = "eDNA_indicator_translator.csv"
LOG_TEMPLATE_CSV = "eDNA_observation_log_template.csv"

SUMMARY_SHEET = "Summary"
SUMMARY_HEADER_ROWS = 3
SUMMARY_NAME_COL = 4
SUMMARY_FIRST_INDICATOR_COL = 5

# group -> (source sheet, output stem); group names as in the Summary header row 1
ASV_GROUPS = OrderedDict([
    ("prokaryotes", "ASV_prokaryotes"),
    ("fungi", "ASV_fungi"),
])

# a formatted sheet can report ~1M empty rows - stop after this many empty rows
EMPTY_ROW_STOP = 50

LOG_COL = "observation_log_id__observation_log_name"
SAMPLING_LOG_COL = "sampling_log_id__sampling_log_name"
SAMPLE_COL = "sample_id__sample_name"
PROVISION_COL = "provision_id__provision_name"

# Summary values are rounded to 3 decimals by the laboratory
TOLERANCE = 0.0011

# indicators recomputed from the ASV sheets, keyed by the normalised Summary indicator name
RECOMPUTED = ("observed richness", "chao1 estimated richness", "dominance", "pielou e", "shannon", "simpson")


class InputError(Exception):
    """A mapping problem that makes the output unusable."""


def norm(text):
    """Lowercase, underscores to spaces, collapsed whitespace, 'funtional' typo fixed."""
    text = str(text).strip().lower().replace("_", " ").replace("funtional", "functional")
    return " ".join(text.split())


def read_csv_dicts(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def iter_sheet_rows(ws):
    """Yield rows until EMPTY_ROW_STOP consecutive rows without a first-column value."""
    n_empty = 0
    for row in ws.iter_rows(values_only=True):
        if not row or all(c is None for c in row):
            n_empty += 1
            if n_empty >= EMPTY_ROW_STOP:
                return
            continue
        n_empty = 0
        yield row


# ---------------------------------------------------------------------------
# translators
# ---------------------------------------------------------------------------

def load_translators(translator_dir):
    samples = OrderedDict()
    for r in read_csv_dicts(os.path.join(translator_dir, SAMPLE_TRANSLATOR_CSV)):
        name = r["summary_name"].strip()
        if name in samples:
            raise InputError(f"{SAMPLE_TRANSLATOR_CSV}: summary_name {name} occurs twice")
        samples[name] = {
            "sampling_log": r[SAMPLING_LOG_COL].strip(),
            "sample": r["sample_name"].strip(),
            "observed_at": r["observed_at"].strip(),
            "exclude": r["exclude"].strip(),
        }

    indicators = OrderedDict()
    for r in read_csv_dicts(os.path.join(translator_dir, INDICATOR_TRANSLATOR_CSV)):
        key = (norm(r["summary_group"]), norm(r["summary_category"]), norm(r["summary_indicator"]))
        indicators[key] = r["xspatula_column"].strip()

    template = read_csv_dicts(os.path.join(translator_dir, LOG_TEMPLATE_CSV))
    if len(template) != 1:
        raise InputError(f"{LOG_TEMPLATE_CSV} must have exactly one data row")

    return samples, indicators, template[0]


# ---------------------------------------------------------------------------
# source workbook
# ---------------------------------------------------------------------------

def parse_number(value):
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip().replace(",", "."))


def read_summary(ws, indicator_translator):
    """Return (columns, rows): columns is [(group, indicator key, xspatula column)] in
    sheet order, rows is [(summary_name, [value or None per column])]."""
    raw = list(iter_sheet_rows(ws))
    header = [list(r) for r in raw[:SUMMARY_HEADER_ROWS]]
    width = max(len(r) for r in header)
    for h in header:
        h.extend([None] * (width - len(h)))

    # merged header cells arrive as None - fill forward along rows 1 and 2
    for level in (0, 1):
        last = None
        for i in range(SUMMARY_FIRST_INDICATOR_COL, width):
            if header[level][i] is None:
                header[level][i] = last
            last = header[level][i]

    columns = []
    for i in range(SUMMARY_FIRST_INDICATOR_COL, width):
        if header[2][i] is None:
            continue
        key = (norm(header[0][i]), norm(header[1][i]), norm(header[2][i]))
        if key not in indicator_translator:
            raise InputError(f"Summary column {i + 1} {key} is not in {INDICATOR_TRANSLATOR_CSV}")
        columns.append((i, key, indicator_translator[key]))

    unused = set(indicator_translator.values()) - {c[2] for c in columns}
    if unused:
        raise InputError(f"indicators in {INDICATOR_TRANSLATOR_CSV} missing from the Summary: {sorted(unused)}")

    rows = []
    for r in raw[SUMMARY_HEADER_ROWS:]:
        if len(r) <= SUMMARY_NAME_COL or r[SUMMARY_NAME_COL] is None:
            continue
        rows.append((str(r[SUMMARY_NAME_COL]).strip(),
                     [parse_number(r[i]) if i < len(r) else None for i, _, _ in columns]))
    return columns, rows


def read_asv_sheet(ws):
    """Return (samples, labels, matrix, derived): matrix is ASVs x samples relative
    abundance; derived holds the non-ASV rows (label without '__', e.g. simpson_alpha)."""
    rows = iter_sheet_rows(ws)
    header = next(rows)
    samples = [str(c).strip() for c in header[1:] if c is not None]
    n = len(samples)
    labels, values, derived = [], [], {}
    for r in rows:
        label = r[0]
        if label is None:
            continue
        label = str(label).strip()
        vec = [parse_number(v) or 0.0 for v in r[1:n + 1]]
        vec.extend([0.0] * (n - len(vec)))
        if "__" in label:
            labels.append(label)
            values.append(vec)
        else:
            derived[label] = np.array(vec)
    return samples, labels, np.array(values), derived


# ---------------------------------------------------------------------------
# diversity indices
# ---------------------------------------------------------------------------

def rarefaction_depth(p):
    """Smallest depth N (from 1/min non-zero p) for which every p*N is an integer, else None."""
    nonzero = p[p > 0]
    if nonzero.size == 0:
        return None
    depth = round(1 / nonzero.min())
    counts = p * depth
    return depth if np.abs(counts - np.round(counts)).max() < 1e-6 else None


def diversity(p, depth):
    """Indices of one sample from relative abundances p (QIIME2 definitions)."""
    p = p[p > 0]
    s = p.size
    shannon = float(-(p * np.log2(p)).sum())
    dominance = float((p ** 2).sum())
    result = {
        "observed richness": float(s),
        "dominance": dominance,
        "simpson": 1 - dominance,
        "shannon": shannon,
        "pielou e": shannon / math.log2(s) if s > 1 else None,
        "chao1 estimated richness": None,
    }
    if depth:
        counts = np.round(p * depth)
        f1 = int((counts == 1).sum())
        f2 = int((counts == 2).sum())
        # bias-corrected chao1, as used by scikit-bio / QIIME2 when f2 == 0 or requested
        result["chao1 estimated richness"] = s + f1 * (f1 - 1) / (2 * (f2 + 1))
        result["chao1 classic"] = s + f1 ** 2 / (2 * f2) if f2 > 0 else None
    return result


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def cell(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def write_xlsx(path, header, rows):
    wb = openpyxl.Workbook(write_only=True)
    ws = wb.create_sheet("Sheet1")
    ws.append(header)
    for r in rows:
        ws.append([cell(v) for v in r])
    wb.save(path)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build(source, translator_dir):
    """Read and check everything; return the output tables and the check report.
    Nothing is written here."""
    samples, indicator_translator, log_template = load_translators(translator_dir)
    provision = log_template[PROVISION_COL].strip()

    wb = openpyxl.load_workbook(source, read_only=True, data_only=True)
    columns, summary_rows = read_summary(wb[SUMMARY_SHEET], indicator_translator)

    unknown = [name for name, _ in summary_rows if name not in samples]
    if unknown:
        raise InputError(f"Summary samples not in {SAMPLE_TRANSLATOR_CSV}: {unknown}")

    def log_name(t):
        return f"{t['sampling_log']}@{provision}"

    notes = []
    excluded = [(name, samples[name]["exclude"]) for name, _ in summary_rows if samples[name]["exclude"]]
    missing_date = [name for name, _ in summary_rows
                    if not samples[name]["exclude"] and not samples[name]["observed_at"]]
    if missing_date:
        raise InputError(f"no observed_at in {SAMPLE_TRANSLATOR_CSV} for: {missing_date}")

    # observation log + observation
    obs_header = [LOG_COL, SAMPLE_COL, "observed_at"] + [c[2] for c in columns]
    obs_rows = []
    for name, values in summary_rows:
        t = samples[name]
        if t["exclude"]:
            continue
        obs_rows.append([log_name(t), t["sample"], int(t["observed_at"])] + values)

    # same column order as the manage_observation_log excel: name (auto) before abstract
    log_fields = [k for k in log_template if k != PROVISION_COL]
    log_fields.insert(log_fields.index("abstract") if "abstract" in log_fields else len(log_fields), "name")
    log_header = [SAMPLING_LOG_COL, PROVISION_COL] + log_fields
    log_template = dict(log_template, name="auto")
    sampling_logs = sorted({samples[name]["sampling_log"] for name, _ in summary_rows if not samples[name]["exclude"]})
    log_rows = [[sl, provision] + [log_template[k] if log_template[k] != "" else None for k in log_fields]
                for sl in sampling_logs]
    log_rows = [[True if v == "True" else False if v == "False" else v for v in r] for r in log_rows]

    summary_by_name = dict(summary_rows)
    check_rows = []
    taxa, asv = {}, {}

    for group, sheet in ASV_GROUPS.items():
        sheet_samples, labels, matrix, derived = read_asv_sheet(wb[sheet])

        not_in_summary = [s for s in sheet_samples if s not in summary_by_name]
        if not_in_summary:
            raise InputError(f"{sheet}: samples not in the Summary: {not_in_summary}")
        absent = [s for s in summary_by_name if s not in sheet_samples and not samples[s]["exclude"]]
        if absent:
            notes.append(f"{sheet}: {len(absent)} Summary sample(s) have no ASV column: {', '.join(absent)}")

        taxa[group] = labels

        # indicator column index in the Summary for this group
        group_cols = {key[2]: j for j, (_, key, _) in enumerate(columns) if key[0] == group}

        asv_rows = []
        for k, s in enumerate(sheet_samples):
            p = matrix[:, k]
            depth = rarefaction_depth(p)
            t = samples[s]
            computed = diversity(p, depth)
            check_rows.append([group, s, "column sum", 1.0, round(float(p.sum()), 6)])
            check_rows.append([group, s, "rarefaction depth", None, depth])
            for ind in RECOMPUTED:
                reported = summary_by_name[s][group_cols[ind]] if ind in group_cols else None
                check_rows.append([group, s, ind, reported, computed.get(ind)])
            if computed.get("chao1 classic") is not None:
                check_rows.append([group, s, "chao1 classic (info)", summary_by_name[s][group_cols["chao1 estimated richness"]], computed["chao1 classic"]])
            for label, vec in derived.items():
                ind = "simpson" if "simpson" in label else "shannon" if "shannon" in label else None
                if ind:
                    check_rows.append([group, s, f"{label} (sheet row)", float(vec[k]), computed[ind]])
            if t["exclude"]:
                continue
            for i in np.nonzero(p)[0]:
                asv_rows.append([log_name(t), t["sample"], f"{group}-{i + 1:06d}", labels[i],
                                 float(p[i]), int(round(p[i] * depth)) if depth else None])
        asv[group] = asv_rows

    wb.close()

    # mismatches: reported and computed both present and differing beyond rounding
    for r in check_rows:
        if r[2] == "rarefaction depth":
            ok = r[4] is not None
        elif "(info)" in r[2] or r[3] is None or r[4] is None:
            ok = None
        elif r[2] == "column sum":
            ok = abs(r[3] - r[4]) < 1e-4
        elif "chao1" in r[2]:
            ok = abs(r[3] - r[4]) <= max(TOLERANCE, 0.0005 * r[3])
        else:
            ok = abs(r[3] - r[4]) <= TOLERANCE
        r.append("" if ok is None else "ok" if ok else "MISMATCH")

    return {
        "log": (log_header, log_rows),
        "observation": (obs_header, obs_rows),
        "taxa": taxa,
        "asv": asv,
        "checks": check_rows,
        "excluded": excluded,
        "notes": notes,
    }


def write_outputs(result, out_dir, report_dir, source):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    written = []

    def xlsx(name, header, rows):
        path = os.path.join(out_dir, name)
        write_xlsx(path, header, rows)
        written.append((path, len(rows)))

    xlsx("AI4SH_observation_log_eDNA.xlsx", *result["log"])
    xlsx("AI4SH_observation_eDNA.xlsx", *result["observation"])
    for group in ASV_GROUPS:
        xlsx(f"taxa_{group}_singlecolumn.xlsx", ["Taxa"], [[t] for t in result["taxa"][group]])
        xlsx(f"AI4SH_asv_{group}_eDNA.xlsx",
             [LOG_COL, SAMPLE_COL, "asv", "taxa", "rel_abundance", "read_count"], result["asv"][group])

    check_path = os.path.join(report_dir, "edna_index_checks.csv")
    write_csv(check_path, ["group", "summary_name", "check", "reported", "computed", "status"], result["checks"])

    mismatches = [r for r in result["checks"] if r[5] == "MISMATCH"]
    depths = sorted({r[4] for r in result["checks"] if r[2] == "rarefaction depth" and r[4]})
    no_depth = [f"{r[0]} {r[1]}" for r in result["checks"] if r[2] == "rarefaction depth" and not r[4]]
    by_check = OrderedDict()
    for r in mismatches:
        by_check.setdefault((r[0], r[2]), []).append(r[1])

    lines = [f"# eDNA conversion report", "", f"Source: `{source}`", "", "## Output", ""]
    lines += [f"- `{os.path.basename(p)}`: {n} rows" for p, n in written]
    lines += ["", "## Excluded samples (eDNA_sample_name_translator.csv)", ""]
    lines += [f"- {name}: {reason}" for name, reason in result["excluded"]] or ["- none"]
    lines += ["", "## Notes", ""] + ([f"- {n}" for n in result["notes"]] or ["- none"])
    lines += ["", "## Recomputed indices", "",
              f"Rarefaction depth detected: {', '.join(map(str, depths)) or 'none'}"
              + (f"; no exact depth for: {', '.join(no_depth)}" if no_depth else ""), ""]
    if by_check:
        lines += ["| group | check | samples differing |", "|---|---|---|"]
        lines += [f"| {g} | {c} | {len(s)}: {', '.join(s[:12])}{' ...' if len(s) > 12 else ''} |" for (g, c), s in by_check.items()]
    else:
        lines += [f"All recomputed indices match the Summary and the sheet rows within {TOLERANCE}."]
    lines += ["", "Functional predictions (FAPROTAX / FUNGuild) cannot be recomputed from the delivered data.",
              "", f"Per sample detail: `{os.path.basename(check_path)}`"]
    report_path = os.path.join(report_dir, "edna_conversion_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return written, report_path, len(mismatches)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", nargs="?", default=DEFAULT_SOURCE, help="master workbook (default: %(default)s)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="folder for the xspatula excel files (default: %(default)s)")
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR, help="folder for the check report (default: %(default)s)")
    parser.add_argument("--translator-dir", default=TRANSLATOR_DIR, help="folder with the translators (default: %(default)s)")
    args = parser.parse_args(argv)

    source = os.path.abspath(args.source)
    if not os.path.exists(source):
        print(f"❌ ERROR - source workbook not found: {source}")
        return 1

    try:
        result = build(source, args.translator_dir)
    except InputError as e:
        print(f"❌ ERROR - {e}\n   nothing written")
        return 1

    written, report_path, n_mismatch = write_outputs(result, os.path.abspath(args.out_dir),
                                                     os.path.abspath(args.report_dir), source)
    for path, n in written:
        print(f"✅ {path} ({n} rows)")
    for name, reason in result["excluded"]:
        print(f"⚠️  excluded {name}: {reason}")
    for note in result["notes"]:
        print(f"⚠️  {note}")
    print(("⚠️  %s index value(s) differ from the recomputed value" % n_mismatch) if n_mismatch
          else "✅ all recomputed indices match")
    print(f"   report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
