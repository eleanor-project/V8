package eleanor.governance

# Constitutional execution gate enforced by OPA.
# This policy mirrors the runtime invariants:
# - Only ExecutableDecision objects may execute
# - Tier 2 requires human acknowledgment with the canonical statement
# - Tier 3 requires human determination with the canonical statement
# - Human action must reference all triggering escalation signals

default allow = false

# Entry point: allow execution only when all conditions pass.
allow {
    input.action == "execute"
    valid_executable_decision
    escalation_requirements_satisfied
}

valid_executable_decision {
    input.decision.executable == true
    input.decision.execution_reason != ""
    input.decision.audit_record_id != ""
}

escalation_requirements_satisfied {
    no_escalation
} else {
    tier2_acknowledged
} else {
    tier3_determined
}

no_escalation {
    not input.decision.aggregation_result.execution_gate.gated
}

tier2_acknowledged {
    input.decision.aggregation_result.execution_gate.escalation_tier == "TIER_2"
    input.decision.human_action.action_type == "HUMAN_ACK"
    valid_tier2_statement
    linked_to_all_escalations
}

valid_tier2_statement {
    input.decision.human_action.statement == "I acknowledge the identified constitutional risks and accept responsibility for proceeding."
}

tier3_determined {
    input.decision.aggregation_result.execution_gate.escalation_tier == "TIER_3"
    input.decision.human_action.action_type == "HUMAN_DETERMINATION"
    valid_tier3_statement
    linked_to_all_escalations
}

valid_tier3_statement {
    input.decision.human_action.statement == "I affirmatively determine the appropriate course of action in light of the identified constitutional risks."
}

linked_to_all_escalations {
    escalations := input.decision.aggregation_result.escalation_summary.triggering_signals
    actions := input.decision.human_action.linked_escalations

    count(escalations) > 0
    count(actions) >= count(escalations)
    all_escalations_linked(escalations, actions)
}

all_escalations_linked(escalations, actions) {
    not missing_escalation(escalations, actions)
}

missing_escalation(escalations, actions) {
    some e
    escalations[e]
    not escalation_linked(escalations[e], actions)
}

escalation_linked(escalation, actions) {
    some a
    actions[a].critic_id == escalation.critic_id
    actions[a].clause_id == escalation.clause_id
}
