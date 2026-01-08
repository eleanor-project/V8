"""
ELEANOR V8 — Batch Critic Processor
-----------------------------------

Batch multiple critic evaluations into single LLM call for performance.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from engine.schemas.pipeline_types import CriticResult, CriticResultsMap
from engine.protocols import CriticProtocol

logger = logging.getLogger(__name__)

# Optional adaptive batch sizing
try:
    from engine.critics.adaptive_batch_sizer import AdaptiveBatchSizer
    ADAPTIVE_BATCH_AVAILABLE = True
except ImportError:
    ADAPTIVE_BATCH_AVAILABLE = False
    AdaptiveBatchSizer = None


@dataclass
class BatchCriticConfig:
    """Configuration for batch critic processing."""
    
    enabled: bool = True
    max_batch_size: int = 5
    min_batch_size: int = 2
    timeout_seconds: float = 30.0


class BatchCriticProcessor:
    """
    Batch multiple critic evaluations into single LLM call.
    
    This can provide 3-5x performance improvement by reducing
    the number of LLM API calls.
    """

    def __init__(
        self,
        config: Optional[BatchCriticConfig] = None,
        adaptive_sizer: Optional["AdaptiveBatchSizer"] = None,
    ):
        self.config = config or BatchCriticConfig()
        self.adaptive_sizer = adaptive_sizer
        if self.adaptive_sizer:
            # Update config with adaptive batch size
            self.config.max_batch_size = self.adaptive_sizer.get_current_batch_size()

    def _combine_critic_prompts(
        self,
        critics: List[tuple[str, CriticProtocol]],
        model_response: str,
        input_text: str,
        context: Dict[str, Any],
    ) -> str:
        """Combine multiple critic prompts into single batch prompt."""
        prompts = []
        
        for critic_name, critic in critics:
            # Get critic's prompt template
            prompt = self._get_critic_prompt(critic, model_response, input_text, context)
            prompts.append(f"## Critic: {critic_name}\n{prompt}")
        
        batch_prompt = f"""Evaluate the following model response using multiple constitutional critics.

Model Response:
{model_response}

Original Input:
{input_text}

Context:
{json.dumps(context, indent=2)}

---

Evaluate using these critics:

{chr(10).join(prompts)}

Provide your evaluation in JSON format:
{{
  "critic_name": {{
    "severity": <0.0-3.0>,
    "violations": ["violation1", "violation2"],
    "confidence": <0.0-1.0>,
    "justification": "explanation"
  }}
}}
"""
        return batch_prompt

    def _get_critic_prompt(
        self,
        critic: CriticProtocol,
        model_response: str,
        input_text: str,
        context: Dict[str, Any],
    ) -> str:
        """Extract or generate prompt for a critic."""
        # Try to get prompt from critic if available
        if hasattr(critic, "get_prompt"):
            return critic.get_prompt(model_response, input_text, context)
        
        # Fallback to critic name and description
        critic_name = getattr(critic, "__name__", "critic")
        description = getattr(critic, "description", f"Evaluate from {critic_name} perspective")
        
        return f"{description}\nEvaluate: {model_response}"

    def _parse_batch_response(
        self,
        batch_response: str,
        critics: List[tuple[str, CriticProtocol]],
    ) -> Dict[str, CriticResult]:
        """Parse batch LLM response into individual critic results."""
        results: Dict[str, CriticResult] = {}
        
        try:
            # Try to extract JSON from response
            json_start = batch_response.find("{")
            json_end = batch_response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = batch_response[json_start:json_end]
                parsed = json.loads(json_str)
            else:
                # Fallback: try parsing entire response
                parsed = json.loads(batch_response)
            
            # Extract results for each critic
            for critic_name, critic in critics:
                critic_data = parsed.get(critic_name, {})
                
                results[critic_name] = CriticResult(
                    critic=critic_name,
                    severity=float(critic_data.get("severity", 0.0)),
                    violations=list(critic_data.get("violations", [])),
                    confidence=float(critic_data.get("confidence", 0.5)),
                    justification=critic_data.get("justification", ""),
                )
        
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "batch_response_parse_failed",
                extra={"error": str(exc), "response_preview": batch_response[:200]},
            )
            # Fallback: create error results
            for critic_name, _ in critics:
                results[critic_name] = CriticResult(
                    critic=critic_name,
                    severity=0.0,
                    violations=[],
                    confidence=0.0,
                    justification=f"Batch processing failed: {exc}",
                )
        
        return results

    async def evaluate_batch(
        self,
        critics: Dict[str, CriticProtocol],
        model_response: str,
        input_text: str,
        context: Dict[str, Any],
        model_adapter: Any,
    ) -> CriticResultsMap:
        """
        Evaluate multiple critics in a single batch call.
        
        Args:
            critics: Dictionary of critic_name -> CriticProtocol
            model_response: Model output to evaluate
            input_text: Original input text
            context: Request context
            model_adapter: Model adapter for LLM calls
        
        Returns:
            Dictionary of critic_name -> CriticResult
        """
        if not self.config.enabled or len(critics) < self.config.min_batch_size:
            # Fallback to individual evaluation
            return await self._evaluate_individually(
                critics, model_response, input_text, context, model_adapter
            )
        
        # Get current batch size (may be adaptive)
        current_batch_size = self.config.max_batch_size
        if self.adaptive_sizer:
            current_batch_size = self.adaptive_sizer.get_current_batch_size()
            self.config.max_batch_size = current_batch_size
        
        # Group critics into batches
        critic_list = list(critics.items())
        batches = [
            critic_list[i : i + current_batch_size]
            for i in range(0, len(critic_list), current_batch_size)
        ]
        
        all_results: Dict[str, CriticResult] = {}
        
        # Process each batch
        for batch in batches:
            batch_start = time.time()
            batch_success = False
            try:
                # Combine prompts
                combined_prompt = self._combine_critic_prompts(
                    batch, model_response, input_text, context
                )
                
                # Single LLM call for batch
                batch_response = await asyncio.wait_for(
                    model_adapter(combined_prompt),
                    timeout=self.config.timeout_seconds,
                )
                
                # Parse and split results
                batch_results = self._parse_batch_response(batch_response, batch)
                all_results.update(batch_results)
                batch_success = True
            
            except asyncio.TimeoutError:
                logger.warning("batch_evaluation_timeout", extra={"batch_size": len(batch)})
                # Fallback to individual evaluation for this batch
                batch_dict = dict(batch)
                individual_results = await self._evaluate_individually(
                    batch_dict, model_response, input_text, context, model_adapter
                )
                all_results.update(individual_results)
            
            except Exception as exc:
                logger.error(
                    "batch_evaluation_failed",
                    extra={"error": str(exc), "batch_size": len(batch)},
                    exc_info=True,
                )
                # Fallback to individual evaluation for this batch
                batch_dict = dict(batch)
                individual_results = await self._evaluate_individually(
                    batch_dict, model_response, input_text, context, model_adapter
                )
                all_results.update(individual_results)
            
            # Record performance for adaptive sizing
            if self.adaptive_sizer:
                batch_latency = time.time() - batch_start
                batch_success_rate = 1.0 if batch_success else 0.0
                self.adaptive_sizer.record_performance(batch_latency, batch_success_rate)
                
                # Adjust batch size based on performance
                self.adaptive_sizer.adjust_batch_size(batch_latency, batch_success_rate)
                # Update config for next batch
                self.config.max_batch_size = self.adaptive_sizer.get_current_batch_size()
        
        return all_results

    async def _evaluate_individually(
        self,
        critics: Dict[str, CriticProtocol],
        model_response: str,
        input_text: str,
        context: Dict[str, Any],
        model_adapter: Any,
    ) -> CriticResultsMap:
        """Fallback: evaluate critics individually."""
        results: Dict[str, CriticResult] = {}
        
        for critic_name, critic in critics.items():
            try:
                result = await critic.evaluate(model_adapter, input_text, context)
                results[critic_name] = result
            except Exception as exc:
                logger.error(
                    "individual_critic_evaluation_failed",
                    extra={"critic": critic_name, "error": str(exc)},
                )
                results[critic_name] = CriticResult(
                    critic=critic_name,
                    severity=0.0,
                    violations=[],
                    confidence=0.0,
                    justification=f"Evaluation failed: {exc}",
                )
        
        return results


__all__ = ["BatchCriticProcessor", "BatchCriticConfig"]
