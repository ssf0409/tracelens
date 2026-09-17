"""Tests asserting documentation alignment with the statistical contract (issue #132)."""

from pathlib import Path


def test_no_documentation_promises_pass_at_k_fallback() -> None:
    """Docs must not teach that pass@k falls back to empirical rates for n < k."""
    docs_dir = Path("docs")
    assert docs_dir.exists() and docs_dir.is_dir()

    forbidden_phrases = [
        "fall back to an empirical rate",
        "falls back to an empirical rate",
        "fallback to an empirical rate",
        "fallback empirical rate",
    ]

    for md_file in docs_dir.glob("**/*.md"):
        content = md_file.read_text(encoding="utf-8")
        for phrase in forbidden_phrases:
            assert phrase not in content, (
                f"{md_file} contains forbidden fallback phrase: {phrase!r}"
            )


def test_user_guide_and_comparing_versions_use_paired_comparison_guidance() -> None:
    """User guide and comparing-versions docs must recommend paired tools with caveats."""
    user_guide = Path("docs/user-guide.md").read_text(encoding="utf-8")
    assert "compare_runs" in user_guide
    assert "tracelens compare" in user_guide

    comparing_versions = Path("docs/comparing-versions.md").read_text(encoding="utf-8")
    assert "independent" in comparing_versions
