import json

import pytest

from dpft.cli import EXIT_OK, main


def test_train_reports_epsilon(capsys):
    assert main(["train", "--n", "200", "--epochs", "5", "--noise", "1.0"]) == EXIT_OK
    assert "ε=" in capsys.readouterr().out


def test_train_json(capsys):
    main(["train", "--n", "200", "--epochs", "5", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert d["epsilon"] > 0 and "train_accuracy" in d


def test_budget_reports_noise(capsys):
    main(["budget", "2.0", "--q", "0.01", "--steps", "500"])
    assert "noise_multiplier" in capsys.readouterr().out


def test_audit_consistent_exit_ok(capsys):
    rc = main(["audit", "--n", "120", "--epochs", "5", "--noise", "1.0",
               "--trials", "40"])
    assert rc == EXIT_OK   # correct mechanism -> consistent -> exit 0


def test_version():
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
