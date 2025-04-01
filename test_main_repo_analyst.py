import pytest
import argparse
import tempfile
import os
import json
from main import parse_arguments, setup_directories, save_json, load_json


def test_parse_arguments_valid(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "main_repo_analyst.py",
        "--url", "https://github.com/test/test",
        "--role", "programmer"
    ])
    args = parse_arguments()
    assert args.url == "https://github.com/test/test"
    assert args.role == "programmer"


def test_setup_directories():
    with tempfile.TemporaryDirectory() as tmpdir:
        dirs = setup_directories(tmpdir, "sample_repo")
        assert os.path.exists(dirs["repo"])
        assert os.path.exists(dirs["data"])
        assert os.path.exists(dirs["embeddings"])
        assert os.path.exists(dirs["reports"])


def test_save_and_load_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        data = {"key": "value"}
        path = os.path.join(tmpdir, "sample.json")
        save_json(path, data)
        loaded = load_json(path)
        assert loaded == data


def test_parse_arguments_invalid(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main_repo_analyst.py", "--role", "programmer"])
    with pytest.raises(SystemExit):
        parse_arguments()
