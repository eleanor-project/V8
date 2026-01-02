"""
Tests for Retry Utility with Exponential Backoff
------------------------------------------------

Tests engine/utils/retry.py
"""

import pytest
from engine.utils.retry import (
    RetryConfig,
    RetryExhausted,
    calculate_delay,
    should_retry,
    retry_with_backoff,
    retry_async
)


# ============================================================
# RetryConfig Tests
# ============================================================

def test_retry_config_defaults():
    """Test RetryConfig default values."""
    config = RetryConfig()

    assert config.max_retries == 3
    assert config.base_delay == 1.0
    assert config.max_delay == 30.0
    assert config.exponential_base == 2.0
    assert config.jitter is True
    assert config.jitter_factor == 0.1
    assert Exception in config.retryable_exceptions
    assert KeyboardInterrupt in config.non_retryable_exceptions
    assert SystemExit in config.non_retryable_exceptions


def test_retry_config_custom():
    """Test RetryConfig with custom values."""
    config = RetryConfig(
        max_retries=5,
        base_delay=0.5,
        max_delay=60.0,
        exponential_base=3.0,
        jitter=False,
        jitter_factor=0.2
    )

    assert config.max_retries == 5
    assert config.base_delay == 0.5
    assert config.max_delay == 60.0
    assert config.exponential_base == 3.0
    assert config.jitter is False
    assert config.jitter_factor == 0.2


# ============================================================
# RetryExhausted Tests
# ============================================================

def test_retry_exhausted_exception():
    """Test RetryExhausted exception creation."""
    original_error = ValueError("Original error")
    exc = RetryExhausted(
        message="Test failed",
        attempts=3,
        last_exception=original_error,
        total_delay=5.5
    )

    assert exc.attempts == 3
    assert exc.last_exception == original_error
    assert exc.total_delay == 5.5
    assert "Test failed" in str(exc)
    assert "Attempts: 3" in str(exc)
    assert "5.50s" in str(exc)


# ============================================================
# calculate_delay Tests
# ============================================================

def test_calculate_delay_exponential():
    """Test exponential backoff calculation."""
    config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

    assert calculate_delay(0, config) == 1.0  # 1.0 * 2^0
    assert calculate_delay(1, config) == 2.0  # 1.0 * 2^1
    assert calculate_delay(2, config) == 4.0  # 1.0 * 2^2
    assert calculate_delay(3, config) == 8.0  # 1.0 * 2^3


def test_calculate_delay_max_cap():
    """Test that delay is capped at max_delay."""
    config = RetryConfig(base_delay=1.0, max_delay=5.0, jitter=False)

    # 1.0 * 2^10 = 1024, should be capped at 5.0
    assert calculate_delay(10, config) == 5.0


def test_calculate_delay_with_jitter():
    """Test that jitter adds randomness to delay."""
    config = RetryConfig(base_delay=10.0, jitter=True, jitter_factor=0.1)

    delays = [calculate_delay(1, config) for _ in range(10)]

    # Base delay for attempt 1 is 20.0, jitter should vary it
    # Jitter range is 20.0 * 0.1 = 2.0, so delay should be in [18.0, 22.0]
    assert all(18.0 <= d <= 22.0 for d in delays)
    # Check that we actually got some variation
    assert len(set(delays)) > 1


def test_calculate_delay_non_negative():
    """Test that delay is never negative."""
    config = RetryConfig(base_delay=0.0, jitter=True)

    delay = calculate_delay(0, config)
    assert delay >= 0


# ============================================================
# should_retry Tests
# ============================================================

def test_should_retry_retryable_exception():
    """Test that retryable exceptions return True."""
    config = RetryConfig(retryable_exceptions=(ValueError, KeyError))

    assert should_retry(ValueError("test"), config) is True
    assert should_retry(KeyError("test"), config) is True


def test_should_retry_non_retryable_exception():
    """Test that non-retryable exceptions return False."""
    config = RetryConfig(non_retryable_exceptions=(KeyboardInterrupt, SystemExit))

    assert should_retry(KeyboardInterrupt(), config) is False
    assert should_retry(SystemExit(), config) is False


def test_should_retry_unknown_exception():
    """Test behavior for exceptions not in either list."""
    config = RetryConfig(
        retryable_exceptions=(ValueError,),
        non_retryable_exceptions=(KeyboardInterrupt,)
    )

    # RuntimeError is neither retryable nor non-retryable
    assert should_retry(RuntimeError("test"), config) is False


# ============================================================
# retry_with_backoff Async Tests
# ============================================================

@pytest.mark.asyncio
async def test_retry_async_success_first_attempt():
    """Test async function succeeds on first attempt."""
    call_count = 0

    @retry_with_backoff(max_retries=3, base_delay=0.01)
    async def test_func():
        nonlocal call_count
        call_count += 1
        return "success"

    result = await test_func()

    assert result == "success"
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_async_success_after_retries():
    """Test async function succeeds after retries."""
    call_count = 0

    @retry_with_backoff(max_retries=3, base_delay=0.01)
    async def test_func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Temporary error")
        return "success"

    result = await test_func()

    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_async_exhausted():
    """Test async function exhausts retries."""
    call_count = 0

    @retry_with_backoff(max_retries=2, base_delay=0.01)
    async def test_func():
        nonlocal call_count
        call_count += 1
        raise ValueError("Persistent error")

    with pytest.raises(RetryExhausted) as exc_info:
        await test_func()

    assert call_count == 3  # Initial + 2 retries
    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.last_exception, ValueError)


@pytest.mark.asyncio
async def test_retry_async_non_retryable_exception():
    """Test async function with non-retryable exception."""
    call_count = 0

    @retry_with_backoff(max_retries=3, base_delay=0.01)
    async def test_func():
        nonlocal call_count
        call_count += 1
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        await test_func()

    # Should fail immediately without retries
    assert call_count == 1


# ============================================================
# retry_with_backoff Sync Tests
# ============================================================

def test_retry_sync_success_first_attempt():
    """Test sync function succeeds on first attempt."""
    call_count = 0

    @retry_with_backoff(max_retries=3, base_delay=0.01)
    def test_func():
        nonlocal call_count
        call_count += 1
        return "success"

    result = test_func()

    assert result == "success"
    assert call_count == 1


def test_retry_sync_success_after_retries():
    """Test sync function succeeds after retries."""
    call_count = 0

    @retry_with_backoff(max_retries=3, base_delay=0.01)
    def test_func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Temporary error")
        return "success"

    result = test_func()

    assert result == "success"
    assert call_count == 3


def test_retry_sync_exhausted():
    """Test sync function exhausts retries."""
    call_count = 0

    @retry_with_backoff(max_retries=2, base_delay=0.01)
    def test_func():
        nonlocal call_count
        call_count += 1
        raise ValueError("Persistent error")

    with pytest.raises(RetryExhausted) as exc_info:
        test_func()

    assert call_count == 3  # Initial + 2 retries
    assert exc_info.value.attempts == 3


# ============================================================
# retry_with_backoff Configuration Tests
# ============================================================

@pytest.mark.asyncio
async def test_retry_with_config_object():
    """Test retry decorator with RetryConfig object."""
    call_count = 0
    config = RetryConfig(max_retries=2, base_delay=0.01)

    @retry_with_backoff(config=config)
    async def test_func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Error")
        return "success"

    result = await test_func()
    assert result == "success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_with_individual_params():
    """Test retry decorator with individual parameters."""
    call_count = 0

    @retry_with_backoff(max_retries=1, base_delay=0.01, max_delay=10.0)
    async def test_func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Error")
        return "success"

    result = await test_func()
    assert result == "success"


@pytest.mark.asyncio
async def test_retry_on_retry_callback():
    """Test on_retry callback is called."""
    callback_calls = []

    def on_retry_callback(attempt, exception, delay):
        callback_calls.append((attempt, type(exception).__name__, delay))

    call_count = 0

    @retry_with_backoff(max_retries=2, base_delay=0.01, on_retry=on_retry_callback)
    async def test_func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Error")
        return "success"

    await test_func()

    # Should have 2 callbacks (for attempt 1 and 2)
    assert len(callback_calls) == 2
    assert callback_calls[0][0] == 1  # First retry
    assert callback_calls[1][0] == 2  # Second retry
    assert all(exc_type == "ValueError" for _, exc_type, _ in callback_calls)


@pytest.mark.asyncio
async def test_retry_callback_exception_ignored():
    """Test that exceptions in callback don't affect retry."""
    def failing_callback(attempt, exception, delay):
        raise RuntimeError("Callback error")

    call_count = 0

    @retry_with_backoff(max_retries=2, base_delay=0.01, on_retry=failing_callback)
    async def test_func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Error")
        return "success"

    # Should still succeed despite callback failing
    result = await test_func()
    assert result == "success"


# ============================================================
# retry_async Function Tests
# ============================================================

@pytest.mark.asyncio
async def test_retry_async_function_success():
    """Test retry_async function with success."""
    call_count = 0

    async def test_func(x, y):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Error")
        return x + y

    config = RetryConfig(max_retries=3, base_delay=0.01)
    result = await retry_async(test_func, 2, 3, config=config)

    assert result == 5
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_async_function_with_kwargs():
    """Test retry_async function with kwargs."""
    async def test_func(x, y=10):
        return x * y

    config = RetryConfig(max_retries=1, base_delay=0.01)
    result = await retry_async(test_func, 3, config=config, y=4)

    assert result == 12


@pytest.mark.asyncio
async def test_retry_async_function_exhausted():
    """Test retry_async function exhausts retries."""
    async def always_fails():
        raise ValueError("Always fails")

    config = RetryConfig(max_retries=2, base_delay=0.01)

    with pytest.raises(RetryExhausted):
        await retry_async(always_fails, config=config)


@pytest.mark.asyncio
async def test_retry_async_with_callback():
    """Test retry_async with on_retry callback."""
    callback_calls = []

    def callback(attempt, exception, delay):
        callback_calls.append(attempt)

    call_count = 0

    async def test_func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Error")
        return "success"

    config = RetryConfig(max_retries=2, base_delay=0.01)
    await retry_async(test_func, config=config, on_retry=callback)

    assert len(callback_calls) == 1
    assert callback_calls[0] == 1


# ============================================================
# Edge Cases
# ============================================================

@pytest.mark.asyncio
async def test_retry_with_zero_retries():
    """Test retry with max_retries=0."""
    call_count = 0

    @retry_with_backoff(max_retries=0, base_delay=0.01)
    async def test_func():
        nonlocal call_count
        call_count += 1
        raise ValueError("Error")

    with pytest.raises(RetryExhausted):
        await test_func()

    # Should only try once
    assert call_count == 1


def test_retry_preserves_function_metadata():
    """Test that decorator preserves function metadata."""
    @retry_with_backoff(max_retries=1)
    def my_function():
        """My docstring"""
        pass

    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "My docstring"
