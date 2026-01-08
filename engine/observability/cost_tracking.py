"""
ELEANOR V8 — LLM Cost Tracking
--------------------------------

Track LLM API costs and token usage for cost optimization.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Histogram = None


# Cost Metrics
if PROMETHEUS_AVAILABLE:
    LLM_COST_TOTAL = Counter(
        "eleanor_llm_cost_total",
        "Total LLM API cost in USD",
        ["model", "provider", "operation"],
    )
    
    LLM_TOKENS = Counter(
        "eleanor_llm_tokens_total",
        "Total tokens used",
        ["model", "provider", "type"],  # type: input, output, total
    )
    
    LLM_REQUESTS = Counter(
        "eleanor_llm_requests_total",
        "Total LLM API requests",
        ["model", "provider", "status"],  # status: success, error, timeout
    )
    
    LLM_LATENCY = Histogram(
        "eleanor_llm_latency_seconds",
        "LLM API latency in seconds",
        ["model", "provider"],
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    )
else:
    LLM_COST_TOTAL = None
    LLM_TOKENS = None
    LLM_REQUESTS = None
    LLM_LATENCY = None


# Pricing per 1M tokens (approximate, update as needed)
PRICING = {
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "grok-beta": {"input": 0.1, "output": 0.1},
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    provider: Optional[str] = None,
) -> float:
    """
    Calculate cost for LLM API call.
    
    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        provider: Provider name (openai, anthropic, etc.)
    
    Returns:
        Cost in USD
    """
    model_key = model.lower()
    pricing = PRICING.get(model_key)
    
    if not pricing:
        # Default pricing if model not found
        logger.warning(f"Unknown model pricing for {model}, using defaults")
        pricing = {"input": 1.0, "output": 2.0}
    
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + output_cost


def record_llm_call(
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    latency_seconds: float,
    status: str = "success",
    operation: str = "inference",
) -> None:
    """
    Record LLM API call metrics.
    
    Args:
        model: Model name
        provider: Provider name
        input_tokens: Input tokens used
        output_tokens: Output tokens used
        latency_seconds: Request latency
        status: Request status (success, error, timeout)
        operation: Operation type (inference, embedding, etc.)
    """
    if not PROMETHEUS_AVAILABLE:
        return
    
    # Record cost
    cost = calculate_cost(model, input_tokens, output_tokens, provider)
    if LLM_COST_TOTAL:
        LLM_COST_TOTAL.labels(
            model=model,
            provider=provider,
            operation=operation,
        ).inc(cost)
    
    # Record tokens
    if LLM_TOKENS:
        LLM_TOKENS.labels(
            model=model,
            provider=provider,
            type="input",
        ).inc(input_tokens)
        
        LLM_TOKENS.labels(
            model=model,
            provider=provider,
            type="output",
        ).inc(output_tokens)
        
        LLM_TOKENS.labels(
            model=model,
            provider=provider,
            type="total",
        ).inc(input_tokens + output_tokens)
    
    # Record request
    if LLM_REQUESTS:
        LLM_REQUESTS.labels(
            model=model,
            provider=provider,
            status=status,
        ).inc()
    
    # Record latency
    if LLM_LATENCY:
        LLM_LATENCY.labels(
            model=model,
            provider=provider,
        ).observe(latency_seconds)


def extract_token_usage(response: Dict[str, Any]) -> tuple[int, int]:
    """
    Extract token usage from LLM response.
    
    Args:
        response: LLM API response
    
    Returns:
        Tuple of (input_tokens, output_tokens)
    """
    # Try common response formats
    usage = response.get("usage") or response.get("token_usage") or {}
    
    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        return int(input_tokens), int(output_tokens)
    
    # Fallback: estimate from text length (rough approximation)
    text = response.get("response_text") or response.get("text") or ""
    estimated_tokens = len(text.split()) * 1.3  # Rough estimate
    return int(estimated_tokens), 0


__all__ = [
    "calculate_cost",
    "record_llm_call",
    "extract_token_usage",
    "PRICING",
]
