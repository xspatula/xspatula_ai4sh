"""Derived nominal classifications, computed from `@`-indicator values at import time.

This is the extension point for the generic "derived nominal classification" design
(see .claude/CLAUDE.md, "Automatic adding of soil texture"): any indicator registered
as a `provision_indicator` with quantity "class" and *not* supplied directly in the
source data is a candidate for one of the functions in `CLASSIFIER_FUNCTIONS`. The
import hook (see `import_data.py::_Manage_specifics`) only stores a classifier's result
for indicator names that are actually registered for the observation's provision - a
classifier itself never checks that; it just answers "given these `@`-values, what can
be derived" and returns nothing for whatever it can't.

Soil texture (IUSS and USDA) is the first case. `iuss_by_tg`/`usda_by_tg` below are
hand-built decision trees (not a box-range lookup) checked against two reference
sources:
    IUSS: /Users/thomasgumbricht/GitHub_xspatula/LUCAS_TO_JSON/IUSS_class_boundaries as vertices.txt
          (exact polygon vertices in ternary clay/silt/sand space - the authoritative
          IUSS reference; iuss_by_tg matches all 5151 integer sand+silt+clay=100 grid
          points exactly against these polygons)
    USDA: /Users/thomasgumbricht/GitHub_xspatula/LUCAS_TO_JSON/soil_texture_classification_2.csv
          (box ranges only, no vertex file available; usda_by_tg was checked against
          this and matches all but a handful of edge cases in the sand>=70% corner,
          flagged in a comment on that branch below - would need the true diagonal
          boundary, like the sand/loamy sand fix in iuss_by_tg, to close those)
Class names match observation_utility.nominal_classification exactly (see
<project>/import_data/utility/observation/excel/nominal_classes.xlsx) - IUSS's high-clay
tier is "clay" / "silty clay" / "light clay" (formerly "heavy clay" / "silty clay" /
"clay"), and IUSS uses "silt loam" (like USDA) for what used to be "silty loam".

IUSS_MAJOR_GROUP / USDA_MAJOR_GROUP are plain literals, not derived from a rule table -
every class name they cover has been verified (grid-tested) to be exactly the set of
class names iuss_by_tg/usda_by_tg can actually produce, so lookups never KeyError.
"""

SUM_TOLERANCE = (98, 102)

IUSS_MAJOR_GROUP = {
    "clay": "heavy clayey soils",
    "silty clay": "heavy clayey soils",
    "light clay": "heavy clayey soils",
    "sandy clay": "light clayey soils",
    "silty clay loam": "light clayey soils",
    "clay loam": "light clayey soils",
    "sandy clay loam": "light clayey soils",
    "silt loam": "loamy soils",
    "loam": "loamy soils",
    "sandy loam": "loamy soils",
    "loamy sand": "sandy soils",
    "sand": "sandy soils",
}

USDA_MAJOR_GROUP = {
    "silty clay": "heavy clayey soils",
    "clay": "heavy clayey soils",
    "sandy clay": "light clayey soils",
    "silty clay loam": "light clayey soils",
    "clay loam": "light clayey soils",
    "sandy clay loam": "light clayey soils",
    "silt loam": "loamy soils",
    "silt": "loamy soils",
    "loam": "loamy soils",
    "sandy loam": "loamy soils",
    "loamy sand": "sandy soils",
    "sand": "sandy soils",
}


def _at(at_columns_D, key):
    """Return @key's value from at_columns_D as a float, or None if absent/unparseable."""
    value = at_columns_D.get("@" + key)
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_component(at_columns_D, direct_key, part_a_key, part_b_key):
    """Direct @-value if present, else the sum of the two part keys if *both* present, else None."""
    direct = _at(at_columns_D, direct_key)
    if direct is not None:
        return direct
    part_a = _at(at_columns_D, part_a_key)
    part_b = _at(at_columns_D, part_b_key)
    if part_a is not None and part_b is not None:
        return part_a + part_b
    return None


def _resolve_component_iuss(at_columns_D, direct_key, part_a_key, part_b_key):
    """IUSS-preferred resolution: the sum of the two part keys if *both* present
    (fine+coarse wins over the direct value when an observation somehow has both -
    see module docstring), else the direct @-value, else None."""
    part_a = _at(at_columns_D, part_a_key)
    part_b = _at(at_columns_D, part_b_key)
    if part_a is not None and part_b is not None:
        return part_a + part_b
    return _at(at_columns_D, direct_key)


def iuss_by_tg(sand, silt, clay):
    if clay >= 45:
        return "clay"
    elif clay >= 25:
        if sand > 55:
            return "sandy clay"
        elif silt > 45:
            return "silty clay"
        else:
            return "light clay"
    elif clay >= 15:
        if silt > 45:
            return "silty clay loam"
        elif silt > 20:
            return "clay loam"
        else:
            return "sandy clay loam"
    elif 3 * clay + silt <= 15:
        return "sand"
    else:
        if sand > 85:
            return "loamy sand"
        elif sand > 65:
            return "sandy loam"
        elif silt < 45:
            return "loam"
        else:
            return "silt loam"

def usda_by_tg(sand, silt, clay):
    if clay >= 60:
        return "clay"
    elif clay >= 40:
        if silt > 40:
            return "silty clay"
        elif sand > 45:
            return "sandy clay"
        else:
            return "clay"
    elif clay >= 35 and sand > 45:
        return "sandy clay"
    elif clay >= 27:
        if sand < 20:
            return "silty clay loam"
        elif sand > 45:
            return "sandy clay loam"
        else:
            return "clay loam"
    elif clay > 20:
        if sand >= 45 and silt < 27:
            return "sandy clay loam"
        elif silt > 50:
            return "silt loam"
        else:
            return "loam"
    elif silt > 50:
        if clay < 12 and silt > 80:
            return "silt"
        else:
            return "silt loam"
    elif sand >= 70:
        # These three classes are not fully correct, that would demand an equation considering the exact proportions of sand, silt, and clay.
        if sand > 85 and clay < 8:
            return "sand"
        elif clay < 12:
            return "loamy sand"
        else:
            return "sandy loam"
    elif sand >= 52:
        return "sandy loam"
    elif silt < 50 and clay < 7:
        return "sandy loam"
    else:
        return "loam"

def classify_soil_texture_usda(at_columns_D):
    """USDA texture class (via usda_by_tg) + 4-group aggregation from @sand/@silt/@clay
    (or @fine sand+@coarse sand / @fine silt+@coarse silt when the direct value is
    absent). Returns {} if the three components can't be resolved, or don't sum to
    98-102."""

    sand = _resolve_component(at_columns_D, "sand", "fine sand", "coarse sand")
    silt = _resolve_component(at_columns_D, "silt", "fine silt", "coarse silt")
    clay = _at(at_columns_D, "clay")

    if sand is None or silt is None or clay is None:
        return {}

    total = sand + silt + clay

    if not (SUM_TOLERANCE[0] <= total <= SUM_TOLERANCE[1]):
        return {}

    class_name = usda_by_tg(sand, silt, clay)

    return {
        "soil texture usda": class_name,
        "soil texture usda major group": USDA_MAJOR_GROUP[class_name],
    }


def classify_soil_texture_iuss(at_columns_D):
    """IUSS texture class (via iuss_by_tg) + major-group aggregation. Prefers the
    precise AI4SoilHealth-style formula (sand = coarse_silt + fine_sand + coarse_sand,
    silt = fine_silt) when *both* fine/coarse sand AND fine/coarse silt are present;
    otherwise falls back to the LUCAS-style approximation (sand = SAND + SILT/2,
    silt = SILT/2), where SAND/SILT are each resolved IUSS-preferred - the fine+coarse
    sum wins over the direct @sand/@silt value whenever that specific pair is
    available, independently for sand and for silt (e.g. fine/coarse sand present but
    no fine/coarse silt still prefers the fine+coarse sand sum, falling back to direct
    @silt for silt). Clay is always the single @clay value. Returns {} if clay or the
    needed sand/silt inputs can't be resolved, or the resolved sand+silt+clay total
    falls outside 98-102."""

    clay = _at(at_columns_D, "clay")

    if clay is None:
        return {}

    fine_sand = _at(at_columns_D, "fine sand")
    coarse_sand = _at(at_columns_D, "coarse sand")
    fine_silt = _at(at_columns_D, "fine silt")
    coarse_silt = _at(at_columns_D, "coarse silt")

    if None not in (fine_sand, coarse_sand, fine_silt, coarse_silt):

        base_sand = fine_sand + coarse_sand
        base_silt = fine_silt + coarse_silt

        if not (SUM_TOLERANCE[0] <= base_sand + base_silt + clay <= SUM_TOLERANCE[1]):
            return {}

        sand = coarse_silt + fine_sand + coarse_sand
        silt = fine_silt

    else:

        base_sand = _resolve_component_iuss(at_columns_D, "sand", "fine sand", "coarse sand")
        base_silt = _resolve_component_iuss(at_columns_D, "silt", "fine silt", "coarse silt")

        if base_sand is None or base_silt is None:
            return {}

        if not (SUM_TOLERANCE[0] <= base_sand + base_silt + clay <= SUM_TOLERANCE[1]):
            return {}

        sand = base_sand + base_silt / 2.0
        silt = base_silt / 2.0

    class_name = iuss_by_tg(sand, silt, clay)

    return {
        "soil texture iuss": class_name,
        "soil texture iuss major group": IUSS_MAJOR_GROUP[class_name],
    }


# Extension point: append a function with the same (at_columns_D) -> {indicator_name:
# class_name} contract for future non-texture derived nominal classifications.
CLASSIFIER_FUNCTIONS = [classify_soil_texture_usda, classify_soil_texture_iuss]

# Every indicator name any CLASSIFIER_FUNCTIONS entry can possibly produce - lets the
# import hook cheaply skip calling classifiers when a provision has none of them
# registered, without having to call every classifier speculatively first.
CLASSIFIER_INDICATOR_NAMES = frozenset({
    "soil texture usda", "soil texture usda major group", "soil texture iuss", "soil texture iuss major group",
})
