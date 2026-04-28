from dataclasses import dataclass


@dataclass
class PipelineConfig:
    top_k: int = 10
    use_propagated_terms: bool = True
    ic_threshold: float = 1.5
    use_canonical_profiles: bool = True
