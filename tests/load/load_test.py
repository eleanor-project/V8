"""
ELEANOR V8 — Load Testing Suite
--------------------------------

Comprehensive load testing for production readiness.
"""

import asyncio
import time
import statistics
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None


@dataclass
class LoadTestResult:
    """Result of a load test run."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_seconds: float
    requests_per_second: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_max: float
    error_rate: float
    errors: List[Dict[str, str]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class LoadTester:
    """
    Load testing tool for ELEANOR V8 API.
    
    Tests:
    - Concurrent request handling
    - Rate limiting behavior
    - Error handling under load
    - Resource exhaustion
    - Performance degradation
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        auth_token: Optional[str] = None,
    ):
        """
        Initialize load tester.
        
        Args:
            base_url: Base URL of the API
            auth_token: Optional authentication token
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required for load testing. Install with: pip install httpx")
        
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.client = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=30.0,
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()
    
    async def run_load_test(
        self,
        endpoint: str = "/deliberate",
        concurrent_users: int = 10,
        requests_per_user: int = 10,
        request_payload: Optional[Dict] = None,
        ramp_up_seconds: int = 5,
    ) -> LoadTestResult:
        """
        Run load test.
        
        Args:
            endpoint: API endpoint to test
            concurrent_users: Number of concurrent users
            requests_per_user: Requests per user
            request_payload: Request payload (defaults to sample)
            ramp_up_seconds: Ramp-up time in seconds
        
        Returns:
            LoadTestResult with statistics
        """
        if not self.client:
            raise RuntimeError("LoadTester not initialized. Use async context manager.")
        
        if request_payload is None:
            request_payload = {
                "input": "Test input for load testing",
                "context": {},
            }
        
        start_time = time.time()
        latencies: List[float] = []
        errors: List[Dict[str, str]] = []
        successful = 0
        failed = 0
        
        async def make_request(user_id: int, request_num: int) -> None:
            """Make a single request."""
            nonlocal successful, failed, latencies, errors
            request_start = time.time()
            try:
                response = await self.client.post(endpoint, json=request_payload)
                latency = time.time() - request_start
                latencies.append(latency)
                
                if response.status_code < 400:
                    successful += 1
                else:
                    failed += 1
                    errors.append({
                        "user_id": str(user_id),
                        "request_num": str(request_num),
                        "status_code": str(response.status_code),
                        "error": response.text[:200],
                    })
            except Exception as exc:
                latency = time.time() - request_start
                latencies.append(latency)
                failed += 1
                errors.append({
                    "user_id": str(user_id),
                    "request_num": str(request_num),
                    "error": str(exc),
                })
        
        # Ramp up users gradually
        tasks = []
        for user_id in range(concurrent_users):
            # Stagger user start times
            await asyncio.sleep(ramp_up_seconds / concurrent_users)
            
            for req_num in range(requests_per_user):
                task = asyncio.create_task(make_request(user_id, req_num))
                tasks.append(task)
        
        # Wait for all requests
        await asyncio.gather(*tasks, return_exceptions=True)
        
        duration = time.time() - start_time
        total_requests = concurrent_users * requests_per_user
        
        # Calculate statistics
        if latencies:
            latencies_sorted = sorted(latencies)
            latency_p50 = statistics.median(latencies_sorted)
            latency_p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
            latency_p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
            latency_max = max(latencies)
        else:
            latency_p50 = latency_p95 = latency_p99 = latency_max = 0.0
        
        return LoadTestResult(
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            duration_seconds=duration,
            requests_per_second=total_requests / duration if duration > 0 else 0.0,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            latency_max=latency_max,
            error_rate=failed / total_requests if total_requests > 0 else 0.0,
            errors=errors[:100],  # Limit to first 100 errors
        )
    
    async def run_stress_test(
        self,
        endpoint: str = "/deliberate",
        max_concurrent: int = 100,
        duration_seconds: int = 60,
    ) -> LoadTestResult:
        """
        Run stress test to find breaking point.
        
        Args:
            endpoint: API endpoint to test
            max_concurrent: Maximum concurrent requests
            duration_seconds: Test duration
        
        Returns:
            LoadTestResult with statistics
        """
        if not self.client:
            raise RuntimeError("LoadTester not initialized. Use async context manager.")
        
        start_time = time.time()
        latencies: List[float] = []
        errors: List[Dict[str, str]] = []
        successful = 0
        failed = 0
        request_count = 0
        
        async def continuous_requests():
            """Continuously make requests."""
            nonlocal request_count, successful, failed, latencies, errors
            request_payload = {
                "input": "Stress test input",
                "context": {},
            }
            
            while time.time() - start_time < duration_seconds:
                request_start = time.time()
                try:
                    response = await self.client.post(endpoint, json=request_payload)
                    latency = time.time() - request_start
                    latencies.append(latency)
                    request_count += 1
                    
                    if response.status_code < 400:
                        successful += 1
                    else:
                        failed += 1
                        errors.append({
                            "status_code": str(response.status_code),
                            "error": response.text[:200],
                        })
                except Exception as exc:
                    latency = time.time() - request_start
                    latencies.append(latency)
                    request_count += 1
                    failed += 1
                    errors.append({"error": str(exc)})
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
        
        # Start concurrent request generators
        tasks = [asyncio.create_task(continuous_requests()) for _ in range(max_concurrent)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        duration = time.time() - start_time
        
        # Calculate statistics
        if latencies:
            latencies_sorted = sorted(latencies)
            latency_p50 = statistics.median(latencies_sorted)
            latency_p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
            latency_p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
            latency_max = max(latencies)
        else:
            latency_p50 = latency_p95 = latency_p99 = latency_max = 0.0
        
        return LoadTestResult(
            total_requests=request_count,
            successful_requests=successful,
            failed_requests=failed,
            duration_seconds=duration,
            requests_per_second=request_count / duration if duration > 0 else 0.0,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            latency_max=latency_max,
            error_rate=failed / request_count if request_count > 0 else 0.0,
            errors=errors[:100],
        )


async def main():
    """Run load tests."""
    import sys
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    auth_token = sys.argv[2] if len(sys.argv) > 2 else None
    
    async with LoadTester(base_url=base_url, auth_token=auth_token) as tester:
        print("Running load test...")
        result = await tester.run_load_test(
            concurrent_users=10,
            requests_per_user=10,
        )
        
        print(f"\nLoad Test Results:")
        print(f"  Total Requests: {result.total_requests}")
        print(f"  Successful: {result.successful_requests}")
        print(f"  Failed: {result.failed_requests}")
        print(f"  Error Rate: {result.error_rate:.2%}")
        print(f"  Duration: {result.duration_seconds:.2f}s")
        print(f"  Requests/sec: {result.requests_per_second:.2f}")
        print(f"  Latency P50: {result.latency_p50*1000:.2f}ms")
        print(f"  Latency P95: {result.latency_p95*1000:.2f}ms")
        print(f"  Latency P99: {result.latency_p99*1000:.2f}ms")
        print(f"  Latency Max: {result.latency_max*1000:.2f}ms")
        
        if result.errors:
            print(f"\n  Errors ({len(result.errors)}):")
            for error in result.errors[:5]:
                print(f"    - {error}")


if __name__ == "__main__":
    asyncio.run(main())
