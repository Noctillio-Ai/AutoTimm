"""Tests for the PyPI update notice."""

import json
import time

import pytest

from autotimm.core import update_check as uc


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Enable the check, isolate its cache, and capture notices."""
    for var in (
        "AUTOTIMM_DISABLE_UPDATE_CHECK",
        "AUTOTIMM_OFFLINE",
        "HF_HUB_OFFLINE",
        "CI",
        "RANK",
        "LOCAL_RANK",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(uc, "_notified", False)
    messages = []
    monkeypatch.setattr(uc.logger, "warning", messages.append)
    return messages


def _pypi(monkeypatch, latest):
    calls = []

    def fake_fetch(timeout=uc.REQUEST_TIMEOUT_SECONDS):
        calls.append(1)
        return latest

    monkeypatch.setattr(uc, "fetch_latest_version", fake_fetch)
    return calls


def test_notifies_when_newer_release(env, monkeypatch):
    _pypi(monkeypatch, "0.8.0")
    uc.check_for_updates("0.7.34", blocking=True)
    assert len(env) == 1
    assert "0.7.34 -> 0.8.0" in env[0]
    assert "pip install -U autotimm" in env[0]


@pytest.mark.parametrize("latest", ["0.7.34", "0.7.0", "0.9.0rc1", None])
def test_silent_when_up_to_date_prerelease_or_offline(env, monkeypatch, latest):
    _pypi(monkeypatch, latest)
    uc.check_for_updates("0.7.34", blocking=True)
    assert env == []


def test_notifies_only_once_per_process(env, monkeypatch):
    _pypi(monkeypatch, "0.8.0")
    uc.check_for_updates("0.7.34", blocking=True)
    uc.check_for_updates("0.7.34", blocking=True)
    assert len(env) == 1


def test_fresh_cache_skips_network(env, monkeypatch, tmp_path):
    calls = _pypi(monkeypatch, "0.8.0")
    uc.check_for_updates("0.7.34", blocking=True)
    monkeypatch.setattr(uc, "_notified", False)
    uc.check_for_updates("0.7.34", blocking=True)
    assert len(calls) == 1
    assert len(env) == 2


def test_stale_cache_refetches(env, monkeypatch, tmp_path):
    cache = tmp_path / "autotimm" / "update_check.json"
    cache.parent.mkdir(parents=True)
    stale = time.time() - uc.CACHE_TTL_SECONDS - 1
    cache.write_text(json.dumps({"checked_at": stale, "latest": "0.7.0"}))
    calls = _pypi(monkeypatch, "0.8.0")
    uc.check_for_updates("0.7.34", blocking=True)
    assert len(calls) == 1
    assert len(env) == 1


@pytest.mark.parametrize(
    "var", ["AUTOTIMM_DISABLE_UPDATE_CHECK", "AUTOTIMM_OFFLINE", "HF_HUB_OFFLINE"]
)
def test_opt_out_env_vars(env, monkeypatch, var):
    monkeypatch.setenv(var, "1")
    calls = _pypi(monkeypatch, "0.8.0")
    uc.check_for_updates("0.7.34", blocking=True)
    assert calls == [] and env == []


def test_skipped_in_ci_and_nonzero_rank(env, monkeypatch):
    calls = _pypi(monkeypatch, "0.8.0")
    monkeypatch.setenv("CI", "true")
    uc.check_for_updates("0.7.34", blocking=True)
    monkeypatch.delenv("CI")
    monkeypatch.setenv("LOCAL_RANK", "1")
    uc.check_for_updates("0.7.34", blocking=True)
    assert calls == [] and env == []


@pytest.mark.parametrize("version", ["unknown", "0.7.35.dev3", "0.7.34+g1234"])
def test_skipped_for_dev_and_unknown_versions(env, monkeypatch, version):
    calls = _pypi(monkeypatch, "0.8.0")
    uc.check_for_updates(version, blocking=True)
    assert calls == [] and env == []


def test_never_raises(env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(uc, "_read_cache", boom)
    uc.check_for_updates("0.7.34", blocking=True)  # must not raise
