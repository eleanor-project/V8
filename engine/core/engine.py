"""
ELEANOR V8 — Unified Runtime Engine
------------------------------------

This is the main execution engine. It performs:

1. Input interpretation
2. Routing to LLM via RouterV8
3. Critic evaluation via OrchestratorV8
4. Precedent retrieval from vector store
5. Precedent alignment and drift detection
6. Uncertainty calculation
7. Aggregation (constitutional decision-making)
8. Governance enforcement via OPA
9. Evidence package assembly + recording
10. Final structured AI governance output
"""

from typing import Dict, Any
import uuid
import time

from engine.aggregator.aggregator import AggregatorV8
from engine.precedent.alignment import PrecedentAlignmentEngineV8
from engine.uncertainty.uncertainty import UncertaintyEngineV8


class EleanorEngineV8:

    def __init__(
        self,
        constitutional_config: Dict[str, Any],
        router,
        orchestrator,
        precedent_retriever,
        evidence_recorder,
        opa_governance_callback
    ):
        self.constitution = constitutional_config
        self.router = router
        self.orchestrator = orchestrator
        self.precedent_retriever = precedent_retriever
        self.evidence_recorder = evidence_recorder
        self.opa_callback = opa_governance_callback

        self.aggregator = AggregatorV8()
        self.alignment_engine = PrecedentAlignmentEngineV8()
        self.uncertainty_engine = UncertaintyEngineV8()

    # =====================================================================
    #  MAIN EXECUTION METHOD
    # =====================================================================
    async def deliberate(self, user_text: str) -> Dict[str, Any]:
        """
        This is the full deliberation pipeline.
        """

        start_time = time.time()
        trace_id = str(uuid.uuid4())

        # ----------------------------------------------------------
        # Step 1: Route input to LLM
        # ----------------------------------------------------------
        llm_output, model_used = await self.router.route(user_text)

        # ----------------------------------------------------------
        # Step 2: Critic evaluation
        # ----------------------------------------------------------
        critics_output = await self.orchestrator.evaluate(user_text, llm_output)

        # ----------------------------------------------------------
        # Step 3: Precedent Retrieval
        # ----------------------------------------------------------
        precedents = self.precedent_retriever.retrieve(user_text)

        query_embedding = precedents.get("query_embedding", [])
        precedent_cases = precedents.get("cases", [])

        # ----------------------------------------------------------
        # Step 4: Precedent Alignment + Drift Detection
        # ----------------------------------------------------------
        alignment = self.alignment_engine.analyze(
            critics=critics_output,
            precedent_cases=precedent_cases,
            query_embedding=query_embedding
        )

        # ----------------------------------------------------------
        # Step 5: Uncertainty Engine
        # ----------------------------------------------------------
        uncertainty = self.uncertainty_engine.compute(
            critics=critics_output,
            model_used=model_used,
            precedent_alignment=alignment
        )

        # ----------------------------------------------------------
        # Step 6: Aggregation (constitutional reasoning)
        # ----------------------------------------------------------
        agg_result = self.aggregator.aggregate(
            critics=critics_output,
            precedent=alignment,
            uncertainty=uncertainty
        )

        # ----------------------------------------------------------
        # Step 7: Governance (OPA)
        # ----------------------------------------------------------
        opa_result = self.opa_callback({
            "critics": critics_output,
            "aggregator": agg_result,
            "precedent": alignment,
            "uncertainty": uncertainty,
            "model_used": model_used,
            "user_input": user_text,
        })

        # ----------------------------------------------------------
        # Step 8: Final Decision Assembly
        # ----------------------------------------------------------
        final_decision = {
            "trace_id": trace_id,
            "timestamp": time.time(),
            "model_used": model_used,
            "router_output": llm_output,
            "critics": critics_output,
            "precedent_alignment": alignment,
            "uncertainty": uncertainty,
            "aggregator_output": agg_result,
            "opa_governance": opa_result,
            "final_decision": self._resolve_output(agg_result, opa_result)
        }

        # ----------------------------------------------------------
        # Step 9: Evidence Recording
        # ----------------------------------------------------------
        self.evidence_recorder.record(
            trace_id=trace_id,
            input_text=user_text,
            model_used=model_used,
            critics=critics_output,
            precedent_alignment=alignment,
            uncertainty=uncertainty,
            aggregator_output=agg_result,
            opa_result=opa_result,
            final_decision=final_decision,
            duration=time.time() - start_time
        )

        # ----------------------------------------------------------
        # Done
        # ----------------------------------------------------------
        return final_decision

    # =====================================================================
    #  Decision Reconciliation (Aggregator + OPA)
    # =====================================================================
    def _resolve_output(self, agg, opa):
        """
        Reconcile aggregator decision with OPA result.

        Rules:
          • If OPA denies → deny
          • If aggregator denies → deny
          • If OPA escalates → escalate
          • Else aggregator's decision stands
        """

        opa_allow = opa.get("allow", True)
        opa_escalate = opa.get("escalate", False)
        agg_decision = agg["decision"]

        if not opa_allow:
            return "deny"

        if opa_escalate:
            return "escalate"

        return agg_decision
