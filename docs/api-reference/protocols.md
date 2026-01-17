# Protocol Interfaces

Protocols define the contracts that runtime components must implement. They live in
`engine/protocols.py`.

Key protocols include:

- `RouterProtocol`
- `CriticProtocol`
- `DetectorEngineProtocol`
- `EvidenceRecorderProtocol`
- `PrecedentEngineProtocol`
- `PrecedentRetrieverProtocol`
- `UncertaintyEngineProtocol`
- `AggregatorProtocol`
- `ReviewTriggerEvaluatorProtocol`

Use these when building custom integrations or mocks.
