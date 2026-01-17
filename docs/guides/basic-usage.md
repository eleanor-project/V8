# Basic Usage

## Run an evaluation

```python
result = await engine.run(
    "Evaluate this response for constitutional compliance",
    context={"domain": "general"},
    detail_level=2,
)
```

## Interpret results

- `result.output_text`: final output
- `result.critic_findings`: per-critic findings
- `result.aggregated`: aggregated decision payload

## Streaming

```python
async for event in engine.run_stream("streamed input", detail_level=2):
    print(event["event"], event)
```
