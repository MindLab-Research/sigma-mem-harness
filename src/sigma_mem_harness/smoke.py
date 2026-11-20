"""Run the compact/read gate without a model checkpoint."""

from .harness import SigmaConversationHarness
from .toy import TraceBackend


def main() -> None:
    backend = TraceBackend()
    harness = SigmaConversationHarness(backend, context_window_tokens=7, compact_target_tokens=4)
    harness.run_turn([1, 2, 3], max_new_tokens=1)
    result = harness.run_turn([4, 5, 6], max_new_tokens=1)
    print({
        "compacted": result.compacted,
        "read_enabled": result.read_enabled,
        "compact_count": harness.state.compact_count,
        "written_tokens": harness.sigma_written_tokens,
        "readable_tokens": harness.sigma_readable_tokens,
        "calls": [call.__dict__ for call in backend.calls],
    })


if __name__ == "__main__":
    main()
