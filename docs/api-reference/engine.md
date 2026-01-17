# Engine API

## EleanorEngineV8

Primary runtime engine defined in `engine/engine.py`.

Key entrypoints (async):

- `run(text, context=None, detail_level=None, trace_id=None)`
- `run_stream(text, context=None, detail_level=None, trace_id=None)`

`detail_level` controls output verbosity:

- 1: output only
- 2: output + critics + precedent/uncertainty
- 3: full forensic payload

## EngineConfig

Configuration data model defined in `engine/runtime/config.py` and loaded via
`EngineConfig` in `engine/engine.py`. Environment-backed settings live in
`engine/config/settings.py` and can be converted into `EngineConfig`.

## EngineResult

Structured output model in `engine/runtime/models.py`.

See also:

- `engine/runtime/run.py`
- `engine/runtime/streaming.py`
