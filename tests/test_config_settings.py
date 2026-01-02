import yaml

from engine.config import get_settings


def _empty_env_file(tmp_path):
    env_file = tmp_path / "empty.env"
    env_file.write_text("", encoding="utf-8")
    return str(env_file)


def test_settings_environment_aliases(monkeypatch, tmp_path):
    env_file = _empty_env_file(tmp_path)

    monkeypatch.setenv("ELEANOR_ENVIRONMENT", "staging")
    settings = get_settings(env_file=env_file, reload=True)
    assert settings.environment == "staging"

    monkeypatch.delenv("ELEANOR_ENVIRONMENT", raising=False)
    monkeypatch.setenv("ELEANOR_ENV", "production")
    settings = get_settings(env_file=env_file, reload=True)
    assert settings.environment == "production"


def test_settings_yaml_mapping(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "storage": {"evidence_path": "custom.jsonl"},
                "performance": {"max_concurrency": 12},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ELEANOR_CONFIG", str(config_path))

    env_file = _empty_env_file(tmp_path)
    settings = get_settings(env_file=env_file, reload=True)

    assert settings.evidence.jsonl_path == "custom.jsonl"
    assert settings.performance.max_concurrency == 12
