from pathlib import Path

import pytest

from model.run_global import _load_tech_config
from reconciliation.land.stage_pv_pilot import (
    ROOT, TECH_YAML, stage_release, tech_yaml_dependencies,
)


def test_actual_release_includes_inherited_configuration(tmp_path):
    output = tmp_path / "release"
    summary = stage_release(ROOT, output)
    assert summary["status"] == "passed"
    base = output / "inputs/tech_config_ammonia_plant_2050_way_eur.yaml"
    assert base.is_file()
    assert _load_tech_config(output / TECH_YAML) == _load_tech_config(ROOT / TECH_YAML)
    with pytest.raises(FileExistsError):
        stage_release(ROOT, output)


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


def test_missing_parent_fails_before_output_creation(tmp_path):
    source = tmp_path / "source"
    yaml_path = source / TECH_YAML
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text("extends: missing.yaml\n")
    output = tmp_path / "release"
    with pytest.raises(FileNotFoundError):
        stage_release(source, output)
    assert not output.exists()
