"""Stable names and provenance for the renewable land-overlap estimate.

MODIS reports land-cover class fractions, while the Green Lory suitability
tables report a potentially different eligible fraction for wind and solar
inside each class.  Their pixel-level overlap is not available.  Version 1
therefore assumes that the smaller eligible footprint is perfectly nested in
the larger one and sums ``max(wind_factor, solar_factor)`` by class.  This is
the minimum possible union area (a lower bound), not an exact spatial union.

The method is encoded in a versioned numeric column.  Generic column names are
retained only as compatibility aliases and must not be used as provenance for
new scientific runs.
"""

CLASSWISE_NESTED_UNION_METHOD = "classwise_nested_overlap_lower_bound"
CLASSWISE_NESTED_UNION_VERSION = "v1"
CLASSWISE_NESTED_UNION_SOURCE = "explicit_classwise_nested_v1_lower_bound"

CLASSWISE_NESTED_UNION_AVAILABILITY_COLUMN = (
    "renewable_union_availability_classwise_nested_v1"
)
CLASSWISE_NESTED_UNION_AREA_COLUMN = (
    "renewable_union_area_km2_classwise_nested_v1"
)

RENEWABLE_UNION_METHOD_COLUMN = "renewable_union_method"
RENEWABLE_UNION_METHOD_VERSION_COLUMN = "renewable_union_method_version"

