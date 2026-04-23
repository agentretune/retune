"""Client-side OptimizationReport with apply/show/copy_snippets."""
from __future__ import annotations

from typing import Any, Callable

from retune.optimizer.models import OptimizationReport as _BaseReport
from retune.optimizer.models import Suggestion


class OptimizationReport(_BaseReport):
    """Extends the shared model with client-side convenience methods."""
    markdown: str = ""

    @classmethod
    def from_cloud_dict(cls, raw: dict[str, Any]) -> "OptimizationReport":
        def _tag(items: list, tier: int) -> list[Suggestion]:
            tagged = []
            for it in items:
                if "tier" not in it:
                    it = {**it, "tier": tier}
                tagged.append(Suggestion.model_validate(it))
            return tagged

        return cls(
            run_id=raw["run_id"],
            understanding=raw.get("understanding", ""),
            summary=raw.get("summary", {}),
            tier1=_tag(raw.get("tier1", []), 1),
            tier2=_tag(raw.get("tier2", []), 2),
            tier3=_tag(raw.get("tier3", []), 3),
            pareto_data=raw.get("pareto_data", []),
            markdown=raw.get("markdown", ""),
        )

    def show(self) -> None:
        """Print the markdown report to stdout."""
        print(self.markdown)

    def apply(
        self,
        tier: int = 1,
        apply_fn: Callable[[Suggestion], None] | None = None,
    ) -> list[Suggestion]:
        """Apply tier-N suggestions."""
        if tier == 1:
            items = self.tier1
        elif tier == 2:
            items = self.tier2
        elif tier == 3:
            items = self.tier3
        else:
            raise ValueError("tier must be 1, 2, or 3")

        if apply_fn is None:
            return items
        for s in items:
            apply_fn(s)
        return items

    def copy_snippets(self, to: str = "stdout") -> str:
        """Return concatenated Tier-2 code snippets (or print them)."""
        blocks = []
        for s in self.tier2:
            if s.code_snippet:
                blocks.append(f"# {s.title}\n{s.code_snippet}")
        text = "\n\n".join(blocks)
        if to == "stdout":
            print(text)
        return text
