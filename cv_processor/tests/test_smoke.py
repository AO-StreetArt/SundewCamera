from cv_processor.cli import main


def test_main_smoke(capsys):
    assert main() == 0
    out = capsys.readouterr().out
    assert "cv-processor" in out
