"""Unit tests for Blackboard implementation.

Tests per tasks.md T016:
- Test reserve_version() monotonicity
- Test concurrent write_proposal() from multiple engines
- Test commit_answer() CAS prevents stale writes
- Test commit_answer() rejects unverified proposals
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from src.models.titans.blackboard import Blackboard
from src.models.titans.data_model import (
    ContextSlot,
    SlotType,
    SolutionProposal,
    VerificationResult,
    VerifierType,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def blackboard():
    """Create a fresh blackboard for each test."""
    return Blackboard()


@pytest.fixture
def sample_proposal():
    """Create a sample solution proposal."""
    return SolutionProposal(
        proposal_id="prop_001",
        problem_id="prob_001",
        engine_id="math_engine",
        version=0,  # Will be set by blackboard
        checksum=0,  # Will be computed
        content=torch.tensor([1, 2, 3]),
        content_text="42",
        raw_confidence=0.95,
        calibrated_confidence=0.92,
    )


# ============================================================================
# Test reserve_version() Monotonicity
# ============================================================================

def test_reserve_version_monotonicity(blackboard):
    """Test that reserve_version() returns monotonically increasing numbers."""
    versions = [blackboard.reserve_version() for _ in range(100)]

    # Check all versions are unique and increasing
    assert len(set(versions)) == 100, "Versions should be unique"
    assert versions == sorted(versions), "Versions should be monotonically increasing"
    assert versions == list(range(1, 101)), "Versions should be consecutive integers"


def test_reserve_version_thread_safety(blackboard):
    """Test reserve_version() is thread-safe under concurrent access."""
    num_threads = 10
    versions_per_thread = 100
    all_versions = []

    def reserve_many():
        """Reserve many versions in a thread."""
        versions = [blackboard.reserve_version() for _ in range(versions_per_thread)]
        return versions

    # Run concurrent reservations
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(reserve_many) for _ in range(num_threads)]
        for future in futures:
            all_versions.extend(future.result())

    # Verify all versions are unique (no race conditions)
    assert len(all_versions) == num_threads * versions_per_thread
    assert len(set(all_versions)) == num_threads * versions_per_thread, \
        "All versions should be unique (no race conditions)"
    assert all_versions == sorted(all_versions), \
        "Versions should be monotonically increasing across threads"


# ============================================================================
# Test write_proposal() Concurrent Access
# ============================================================================

def test_write_proposal_sets_version_and_checksum(blackboard, sample_proposal):
    """Test that write_proposal() sets version and checksum."""
    blackboard.write_proposal("engine_1", sample_proposal)

    # Verify version was set
    assert sample_proposal.version > 0, "Version should be set"

    # Verify checksum was computed
    assert sample_proposal.checksum > 0, "Checksum should be computed"

    # Verify proposal is in blackboard
    proposals = blackboard.read_proposals()
    assert "engine_1" in proposals
    assert proposals["engine_1"].proposal_id == "prop_001"


def test_write_proposal_concurrent_engines(blackboard):
    """Test concurrent write_proposal() from multiple engines."""
    num_engines = 10

    def write_from_engine(engine_id):
        """Write a proposal from a specific engine."""
        proposal = SolutionProposal(
            proposal_id=f"prop_{engine_id}",
            problem_id="prob_001",
            engine_id=f"engine_{engine_id}",
            version=0,
            checksum=0,
            content=torch.tensor([engine_id]),
            content_text=str(engine_id),
            raw_confidence=0.9,
            calibrated_confidence=0.85,
        )
        blackboard.write_proposal(f"engine_{engine_id}", proposal)
        return f"engine_{engine_id}"

    # Write proposals concurrently
    with ThreadPoolExecutor(max_workers=num_engines) as executor:
        futures = [executor.submit(write_from_engine, i) for i in range(num_engines)]
        engine_ids = [f.result() for f in futures]

    # Verify all proposals were written
    proposals = blackboard.read_proposals()
    assert len(proposals) == num_engines, "All engines should have proposals"

    for engine_id in engine_ids:
        assert engine_id in proposals, f"{engine_id} should have proposal"
        assert proposals[engine_id].engine_id == engine_id


def test_write_proposal_overwrites_previous(blackboard, sample_proposal):
    """Test that write_proposal() overwrites previous proposal from same engine."""
    # Write first proposal
    blackboard.write_proposal("engine_1", sample_proposal)
    first_version = sample_proposal.version

    # Create second proposal
    second_proposal = SolutionProposal(
        proposal_id="prop_002",
        problem_id="prob_001",
        engine_id="engine_1",
        version=0,
        checksum=0,
        content=torch.tensor([4, 5, 6]),
        content_text="100",
        raw_confidence=0.98,
        calibrated_confidence=0.96,
    )

    # Write second proposal
    blackboard.write_proposal("engine_1", second_proposal)

    # Verify only second proposal is stored
    proposals = blackboard.read_proposals()
    assert len(proposals) == 1
    assert proposals["engine_1"].proposal_id == "prop_002"
    assert proposals["engine_1"].version > first_version


# ============================================================================
# Test commit_answer() CAS and Version Checking
# ============================================================================

def test_commit_answer_requires_verified(blackboard, sample_proposal):
    """Test that commit_answer() rejects unverified proposals."""
    blackboard.write_proposal("engine_1", sample_proposal)

    # Try to commit unverified proposal
    success = blackboard.commit_answer(sample_proposal)

    assert not success, "Should reject unverified proposal"
    assert blackboard.get_answer() is None, "No answer should be committed"

    # Check event log
    events = blackboard.get_events(event_type='commit_rejected')
    assert len(events) > 0
    assert events[-1]['reason'] == 'unverified'


def test_commit_answer_succeeds_when_verified(blackboard, sample_proposal):
    """Test that commit_answer() succeeds for verified proposals."""
    blackboard.write_proposal("engine_1", sample_proposal)

    # Mark as verified
    sample_proposal.verified = True

    # Commit should succeed
    success = blackboard.commit_answer(sample_proposal)

    assert success, "Should accept verified proposal"
    assert blackboard.get_answer() is not None
    assert blackboard.get_answer().proposal_id == "prop_001"
    assert blackboard.answer_version == sample_proposal.version
    assert blackboard.answer_checksum == sample_proposal.checksum


def test_commit_answer_cas_prevents_stale_writes(blackboard):
    """Test that CAS prevents stale proposals from overwriting newer ones."""
    # Create two proposals with different versions
    proposal_v1 = SolutionProposal(
        proposal_id="prop_v1",
        problem_id="prob_001",
        engine_id="engine_1",
        version=1,
        checksum=1000,
        content=torch.tensor([1]),
        content_text="old",
        raw_confidence=0.9,
        calibrated_confidence=0.85,
        verified=True,
    )

    proposal_v2 = SolutionProposal(
        proposal_id="prop_v2",
        problem_id="prob_001",
        engine_id="engine_2",
        version=2,
        checksum=2000,
        content=torch.tensor([2]),
        content_text="new",
        raw_confidence=0.95,
        calibrated_confidence=0.90,
        verified=True,
    )

    # Commit newer proposal first
    blackboard.commit_answer(proposal_v2)
    assert blackboard.answer_version == 2

    # Try to commit older proposal - should be rejected
    success = blackboard.commit_answer(proposal_v1)
    assert not success, "Should reject stale proposal"
    assert blackboard.answer_version == 2, "Should keep newer version"
    assert blackboard.get_answer().proposal_id == "prop_v2"

    # Check event log
    events = blackboard.get_events(event_type='commit_rejected')
    assert any(e['reason'] == 'stale_version' for e in events)


def test_commit_answer_race_condition(blackboard):
    """Test CAS handles race conditions between concurrent commits."""
    num_threads = 10
    proposals = []

    # Create proposals with sequential versions
    for i in range(num_threads):
        proposal = SolutionProposal(
            proposal_id=f"prop_{i}",
            problem_id="prob_001",
            engine_id=f"engine_{i}",
            version=i + 1,
            checksum=i * 1000,
            content=torch.tensor([i]),
            content_text=str(i),
            raw_confidence=0.9,
            calibrated_confidence=0.85,
            verified=True,
        )
        proposals.append(proposal)

    # Try to commit all concurrently
    results = []

    def try_commit(proposal):
        return blackboard.commit_answer(proposal)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(try_commit, p) for p in proposals]
        results = [f.result() for f in futures]

    # Only one commit should succeed (or possibly more if versions are ordered)
    # But final state should be highest version
    assert blackboard.get_answer() is not None
    final_version = blackboard.answer_version

    # Final version should be one of the attempted versions
    attempted_versions = [p.version for p in proposals]
    assert final_version in attempted_versions

    # All proposals with version <= final should either succeed or be rejected as stale
    # At least one should have succeeded
    assert any(results), "At least one commit should succeed"


# ============================================================================
# Test Context Slots (MAC Coupling)
# ============================================================================

def test_write_and_read_slots(blackboard):
    """Test basic slot write and read operations."""
    # Write some slots
    blackboard.write_slot(
        slot_id="goal_1",
        value=torch.randn(128),
        slot_type=SlotType.GOAL,
        utility=1.0,
    )

    blackboard.write_slot(
        slot_id="evidence_1",
        value=torch.randn(128),
        slot_type=SlotType.EVIDENCE,
        utility=0.8,
    )

    # Read all slots
    all_slots = blackboard.read_slots()
    assert len(all_slots) == 2
    assert "goal_1" in all_slots
    assert "evidence_1" in all_slots

    # Read filtered by type
    goal_slots = blackboard.read_slots(slot_types=[SlotType.GOAL])
    assert len(goal_slots) == 1
    assert "goal_1" in goal_slots


def test_read_slots_budget_control(blackboard):
    """Test that read_slots() respects max_slots budget."""
    # Write many slots with different utilities
    for i in range(10):
        blackboard.write_slot(
            slot_id=f"slot_{i}",
            value=torch.randn(128),
            slot_type=SlotType.EVIDENCE,
            utility=float(i),  # Increasing utility
        )

    # Read with budget limit
    limited_slots = blackboard.read_slots(max_slots=5)
    assert len(limited_slots) == 5

    # Should get top 5 by utility (5, 6, 7, 8, 9)
    slot_ids = set(limited_slots.keys())
    expected_ids = {f"slot_{i}" for i in range(5, 10)}
    assert slot_ids == expected_ids, "Should get highest utility slots"


def test_slot_versioning(blackboard):
    """Test that slot updates increment version."""
    # Write initial slot
    blackboard.write_slot(
        slot_id="goal_1",
        value=torch.randn(128),
        slot_type=SlotType.GOAL,
    )

    slot_v1 = blackboard.read_slot("goal_1")
    assert slot_v1.version == 1

    # Update slot
    blackboard.write_slot(
        slot_id="goal_1",
        value=torch.randn(128),
        slot_type=SlotType.GOAL,
    )

    slot_v2 = blackboard.read_slot("goal_1")
    assert slot_v2.version == 2


# ============================================================================
# Test Reset and Snapshot
# ============================================================================

def test_reset_clears_all_state(blackboard, sample_proposal):
    """Test that reset() clears all blackboard state."""
    # Add some state
    blackboard.write_proposal("engine_1", sample_proposal)
    blackboard.write_slot("slot_1", torch.randn(128), SlotType.GOAL)
    sample_proposal.verified = True
    blackboard.commit_answer(sample_proposal)

    # Reset
    blackboard.reset()

    # Verify everything is cleared
    assert len(blackboard.read_proposals()) == 0
    assert len(blackboard.read_slots()) == 0
    assert blackboard.get_answer() is None
    assert blackboard.answer_version == 0
    assert len(blackboard.event_log) == 0
    assert blackboard._version_counter == 0


def test_snapshot_captures_state(blackboard, sample_proposal):
    """Test that snapshot() captures current state."""
    # Add some state
    blackboard.write_proposal("engine_1", sample_proposal)
    blackboard.write_slot("slot_1", torch.randn(128), SlotType.GOAL)

    # Take snapshot
    snapshot = blackboard.snapshot()

    # Verify snapshot contents
    assert 'slots' in snapshot
    assert 'proposals' in snapshot
    assert 'answer' in snapshot
    assert 'version' in snapshot

    assert len(snapshot['proposals']) == 1
    assert len(snapshot['slots']) == 1
    assert snapshot['answer'] is None
    assert snapshot['version'] > 0


# ============================================================================
# Test Event Logging
# ============================================================================

def test_event_logging(blackboard, sample_proposal):
    """Test that operations are logged correctly."""
    # Perform some operations
    blackboard.write_proposal("engine_1", sample_proposal)
    sample_proposal.verified = True
    blackboard.commit_answer(sample_proposal)

    # Check events
    events = blackboard.get_events()
    assert len(events) >= 2

    # Check proposal written event
    proposal_events = blackboard.get_events(event_type='proposal_written')
    assert len(proposal_events) == 1
    assert proposal_events[0]['engine_id'] == 'engine_1'

    # Check answer committed event
    commit_events = blackboard.get_events(event_type='answer_committed')
    assert len(commit_events) == 1
    assert commit_events[0]['proposal_id'] == 'prop_001'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
