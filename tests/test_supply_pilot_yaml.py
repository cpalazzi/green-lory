from pathlib import Path

import pytest

from model.run_global import _load_tech_config
from reconciliation.supply_pilot.run_supply_pilot import ROOT, TECH_YAML, tech_yaml_dependencies


def test_pilot_yaml_chain_resolves_inside_the_repository():
    chain = tech_yaml_dependencies(ROOT / TECH_YAML, ROOT)
    assert [path.name for path in chain] == [
        "tech_config_ammonia_plant_2050_way_eur_explicit_compressor_dea_tank.yaml",
        "tech_config_ammonia_plant_2050_way_eur.yaml",
    ]
    assert _load_tech_config(ROOT / TECH_YAML)


def test_recursive_yaml_dependencies(tmp_path):
    for name, text in (("a", "extends: b.yaml\n"),
                       ("b", "extends: c.yaml\n"), ("c", "techs: {}\n")):
        (tmp_path / f"{name}.yaml").write_text(text)
    assert [p.name for p in tech_yaml_dependencies(tmp_path / "a.yaml", tmp_path)] == [
        "a.yaml", "b.yaml", "c.yaml",
    ]


@pytest.mark.parametrize("parent,match", [
    ("../outside.yaml", "outside release root"),
    ("/absolute.yaml", "relative path"),
    ("a.yaml", "Circular"),
    ("[]", "relative path"),
])
def test_yaml_dependency_rejects_nonportable_or_circular_parent(tmp_path, parent, match):
    path = tmp_path / "a.yaml"
    path.write_text(f"extends: {parent}\n")
    with pytest.raises(ValueError, match=match):
        tech_yaml_dependencies(path, tmp_path)


def test_missing_parent_is_an_error(tmp_path):
    path = tmp_path / "a.yaml"
    path.write_text("extends: missing.yaml\n")
    with pytest.raises(FileNotFoundError):
        tech_yaml_dependencies(path, tmp_path)
