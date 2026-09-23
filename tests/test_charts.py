import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "docs" / "charts.py"


def _load_charts():
    spec = importlib.util.spec_from_file_location("stuntd_charts", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


charts = _load_charts()


def test_the_committed_charts_match_the_script():
    assert charts.main(["--check"]) == 0


def test_check_fails_on_a_chart_the_script_would_not_write(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(charts, "DOCS", tmp_path)
    assert charts.main([]) == 0
    assert charts.main(["--check"]) == 0
    (tmp_path / "snake.svg").write_text("<svg/>", encoding="utf-8")
    assert charts.main(["--check"]) == 1
    assert "snake.svg is out of date" in capsys.readouterr().err
