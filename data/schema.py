"""
DataRecord schema and validation per data-model.md specifications.

Defines the atomic Q/A example format before packing into sequences.
"""

import re
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import jsonschema


@dataclass
class DataRecord:
    """
    Atomic Q/A example before packing into sequences.

    Attributes:
        id: Unique UUID v4 identifier
        band: Curriculum band ID (A0-A12, MBPP_LITE)
        question: Wrapped question text (e.g., "⟨N⟩3 7⟨/N⟩+⟨N⟩5 8⟨/N⟩")
        answer: Normal answer text for tools/verifier (e.g., "95")
        answer_split: Split answer for training (e.g., "9 5") - NEW!
        hints: DEPRECATED - derive from split view instead
        dual: Optional dual-view dict (e.g., {"rev": "59"})
        tool_supervision: Optional tool call supervision (e.g., {"call": "<TOOL:calc>..."})
        verifier: Verification result (must have {"ok": True})
    """

    id: str  # UUID as string
    band: str
    question: str  # Wrapped numbers: "⟨N⟩3 7⟨/N⟩+⟨N⟩5 8⟨/N⟩"
    answer: str  # Normal format: "95"
    answer_split: Optional[str] = None  # Split format: "9 5" (NEW!)
    hints: Optional[Dict[str, str]] = None  # DEPRECATED
    dual: Optional[Dict[str, str]] = None
    tool_supervision: Optional[Dict[str, str]] = None
    verifier: Optional[Dict[str, bool]] = None

    def __post_init__(self):
        """Validate fields after initialization."""
        # Validate UUID format
        try:
            uuid.UUID(self.id, version=4)
        except ValueError:
            raise ValueError(f"Invalid UUID v4: {self.id}")

        # Validate band format
        band_pattern = r"^(FORMAT|A[0-9]|A1[0-2]|MBPP_LITE)$"
        if not re.match(band_pattern, self.band):
            raise ValueError(f"Invalid band: {self.band}. Must match {band_pattern}")

        # Validate non-empty question and answer
        if not self.question or not self.answer:
            raise ValueError("Question and answer must be non-empty strings")

        # Validate verifier
        if self.verifier is None:
            self.verifier = {"ok": True}
        elif not self.verifier.get("ok", False):
            raise ValueError(f"Verifier must have 'ok': True. Got: {self.verifier}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataRecord":
        """
        Create DataRecord from dictionary.

        Args:
            data: Dictionary with record fields

        Returns:
            DataRecord instance
        """
        return cls(**data)

    def validate_against_schema(self, schema: Dict[str, Any]) -> None:
        """
        Validate this record against a JSON schema.

        Args:
            schema: JSON schema dictionary

        Raises:
            jsonschema.ValidationError: If validation fails
        """
        jsonschema.validate(self.to_dict(), schema)


def load_schema(schema_path: str) -> Dict[str, Any]:
    """
    Load JSON schema from file.

    Args:
        schema_path: Path to JSON schema file

    Returns:
        Loaded schema dictionary
    """
    import json
    from pathlib import Path

    with open(Path(schema_path), "r") as f:
        return json.load(f)


def validate_record(record: DataRecord, schema_path: Optional[str] = None) -> bool:
    """
    Validate a DataRecord against its JSON schema.

    Args:
        record: DataRecord to validate
        schema_path: Optional path to schema file (defaults to contracts/data_record_schema.json)

    Returns:
        True if validation passes

    Raises:
        jsonschema.ValidationError: If validation fails
    """
    if schema_path is None:
        # Default schema location
        from pathlib import Path

        schema_path = (
            Path(__file__).parent.parent
            / "specs"
            / "002-new-curriculum-specs"
            / "contracts"
            / "data_record_schema.json"
        )

    schema = load_schema(str(schema_path))
    record.validate_against_schema(schema)
    return True


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    from utils.number_wrapping import wrap_numbers, split_num

    # Create a sample record with new format
    question_text = "37 + 58"
    answer_text = "95"

    # Wrap numbers
    wrapped_question = wrap_numbers(question_text)

    # Create split answer
    answer_split_text = split_num(answer_text)

    record = DataRecord(
        id=str(uuid.uuid4()),
        band="A2",
        question=wrapped_question,  # Wrapped
        answer=answer_text,  # Normal
        answer_split=answer_split_text,  # Split (NEW!)
        hints=None,  # Deprecated
        dual={"rev": "59"},
        tool_supervision=None,
        verifier={"ok": True},
    )

    print("DataRecord created (NEW FORMAT):")
    print(f"  ID: {record.id}")
    print(f"  Band: {record.band}")
    print(f"  Question (wrapped): {record.question}")
    print(f"  Answer (normal): {record.answer}")
    print(f"  Answer (split): {record.answer_split}")
    print(f"  Hints: {record.hints} (deprecated)")
    print(f"  Verifier: {record.verifier}")

    # Test validation
    try:
        # This should fail (invalid band)
        bad_record = DataRecord(
            id=str(uuid.uuid4()),
            band="INVALID",
            question="test",
            answer="test",
        )
    except ValueError as e:
        print(f"\n✓ Validation caught invalid band: {e}")

    # Convert to/from dict
    record_dict = record.to_dict()
    restored = DataRecord.from_dict(record_dict)
    print(f"\n✓ Round-trip serialization successful: {restored.id == record.id}")
