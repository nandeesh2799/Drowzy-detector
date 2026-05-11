from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ExplainabilityOutput:
    summary: str
    breakdown: Dict[str, float]
    top_reasons: List[str]


class ExplainabilityEngine:
    def explain(
        self,
        fatigue_state: str,
        reasons: Dict[str, float],
        blink_rate: float,
        ear: float,
        mar: float,
        distraction_reason: str,
    ) -> ExplainabilityOutput:
        total = max(1e-6, sum(reasons.values()))
        breakdown = {k: (v / total) * 100.0 for k, v in reasons.items() if v > 0}
        sorted_items = sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)
        top = [f"{name.replace('_', ' ').title()} {pct:.0f}%" for name, pct in sorted_items[:3]]

        if fatigue_state == "Critical":
            prefix = "CRITICAL"
        elif fatigue_state == "Warning":
            prefix = "WARNING"
        else:
            prefix = "ALERT"

        evidence = f"EAR {ear:.2f}, MAR {mar:.2f}, Blink {blink_rate:.1f}/min"
        if distraction_reason != "Focused":
            evidence += f", {distraction_reason}"
        summary = f"{prefix}: " + (" + ".join(top) if top else "low-risk pattern") + f" ({evidence})"
        return ExplainabilityOutput(summary=summary, breakdown=breakdown, top_reasons=top)
