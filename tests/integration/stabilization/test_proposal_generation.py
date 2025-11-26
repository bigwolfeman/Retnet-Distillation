import json
import time
from pathlib import Path

import torch

from src.core.api.proposals import ProposalService, solution_proposal_to_dict
from src.models.titans.data_model import Problem, SolutionProposal
from src.models.titans.decoding_profiles import (
    DecodingProfile,
    DecodingProfileCatalog,
    DecodingProfileMode,
    DecodingProfileParameters,
    dump_catalog,
)


class StubTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True, max_length=None):
        return [ord(ch) % 256 for ch in text]

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return "".join(chr(token_id) for token_id in token_ids)


class StubEngine:
    def __init__(self):
        self.last_parameters = None
        self.last_profile = None
        self.call_count = 0

    def solve(self, problem: Problem, blackboard=None, *, profile=None, parameters=None):
        self.last_parameters = parameters
        self.last_profile = profile
        self.call_count += 1

        content_tokens = torch.tensor([1, 2, 3], dtype=torch.long)
        proposal = SolutionProposal(
            proposal_id=f"stub-{self.call_count}",
            problem_id=problem.problem_id,
            engine_id="stub-engine",
            version=0,
            checksum=0,
            content=content_tokens,
            content_text="stub-output",
            raw_confidence=0.7,
            calibrated_confidence=0.65,
            logits=None,
            timestamp=time.time(),
            latency=0.01,
            cost=float(len(content_tokens)),
            was_constrained=False,
            verified=False,
            reasoning_trace=["stub"],
        )
        return proposal


def _write_catalog(path: Path) -> DecodingProfileCatalog:
    deterministic = DecodingProfile(
        profile_id="11111111-1111-1111-1111-111111111111",
        name="deterministic",
        mode=DecodingProfileMode.DETERMINISTIC,
        parameters=DecodingProfileParameters(
            temperature=0.0,
            top_p=0.0,
            top_k=1,
            max_new_tokens=20,
        ),
        latency_budget_ms=200,
        quality_notes="Deterministic",
        enabled=True,
    )
    exploratory = DecodingProfile(
        profile_id="22222222-2222-2222-2222-222222222222",
        name="exploratory",
        mode=DecodingProfileMode.EXPLORATORY,
        parameters=DecodingProfileParameters(
            temperature=0.8,
            top_p=0.9,
            top_k=20,
            max_new_tokens=40,
        ),
        latency_budget_ms=400,
        quality_notes="Exploratory",
        enabled=True,
    )
    catalog = DecodingProfileCatalog([deterministic, exploratory])
    dump_catalog(catalog, path)
    return catalog


def test_proposal_service_generates_with_profile(tmp_path):
    catalog_path = tmp_path / "profiles.json"
    catalog = _write_catalog(catalog_path)

    engine = StubEngine()
    tokenizer = StubTokenizer()
    service = ProposalService(engine=engine, tokenizer=tokenizer, profiles_path=catalog_path)

    proposal, profile = service.generate_from_prompt(
        problem_id="prob-123",
        prompt="test prompt",
        profile_id="22222222-2222-2222-2222-222222222222",
        max_new_tokens=55,
    )

    assert engine.call_count == 1
    assert engine.last_profile.profile_id == profile.profile_id
    assert engine.last_parameters.max_new_tokens == 55
    assert proposal.content_text == "stub-output"

    serialized = solution_proposal_to_dict(proposal)
    assert serialized["proposalId"] == "stub-1"
    assert serialized["latencyMs"] == 10
    assert serialized["contentText"] == "stub-output"

    enabled_profiles = list(service.list_profiles())
    assert len(enabled_profiles) == 2
