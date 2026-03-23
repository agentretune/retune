"""Credit assigner tool — determines which step caused success/failure.

Inspired by Microsoft Agent Lightning's hierarchical credit assignment:
for each step in an execution trace, compute a "responsibility score"
indicating how much that step contributed to the overall outcome.
"""

from __future__ import annotations

from typing import Any

from retune.tools.base import RetuneTool


class CreditAssignerTool(RetuneTool):
    """Assigns credit/blame to individual steps in an execution trace.

    Uses heuristics inspired by Agent Lightning:
    - Steps closer to the final output get higher blame for bad answers
    - Retrieval steps that returned no docs are highly blamed
    - Tool calls that weren't used in the response are flagged as wasteful
    - LLM calls with high token usage but poor quality get blamed

    This tool provides the data; the LLM subagent reasons about causality.
    """

    name: str = "credit_assigner"
    description: str = (
        "Assign credit/blame scores to each step in an execution trace. "
        "Input: steps (list of step dicts), response (str), eval_results (list). "
        "Output: per-step credit scores with blame analysis."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "steps": {"type": "array", "description": "List of step dicts from trace"},
            "response": {"type": "string", "description": "Final response text"},
            "eval_results": {"type": "array", "description": "Evaluation results"},
        },
        "required": ["steps"],
    }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        steps = kwargs.get("steps", [])
        response = str(kwargs.get("response", "")).lower()
        eval_results = kwargs.get("eval_results", [])

        if not steps:
            return {"credits": [], "summary": "No steps to analyze"}

        # Overall quality signal
        overall_score = 1.0
        if eval_results:
            scores = [r.get("score", 0.5) for r in eval_results if isinstance(r, dict)]
            if scores:
                overall_score = sum(scores) / len(scores)

        is_failure = overall_score < 0.6

        credits = []
        total_steps = len(steps)

        for i, step in enumerate(steps):
            step_type = step.get("step_type", "unknown")
            name = step.get("name", "unknown")
            output_data = step.get("output_data", {})


            # Base blame: later steps get more blame (they're closer to output)
            position_weight = (i + 1) / total_steps

            blame = 0.0
            reasons = []

            if step_type == "retrieval":
                num_docs = output_data.get("num_docs", 0)
                if num_docs == 0:
                    blame = 0.9
                    reasons.append("Retrieval returned ZERO documents — primary failure point")
                elif num_docs == 1:
                    blame = 0.5
                    reasons.append("Only 1 document retrieved — insufficient context")
                else:
                    # Check if retrieved docs were used
                    from retune.utils.text_similarity import text_overlap_score

                    docs = output_data.get("documents", [])
                    doc_overlap = 0.0
                    for doc in docs:
                        content = str(doc.get("content", "")).lower()
                        if content:
                            doc_overlap = max(doc_overlap, text_overlap_score(content, response))

                    if doc_overlap < 0.05:
                        blame = 0.6
                        reasons.append("Retrieved docs not reflected in response — ignored by LLM")
                    else:
                        blame = 0.1
                        reasons.append(
                            f"Retrieval successful — "
                            f"{doc_overlap:.0%} content overlap"
                        )

            elif step_type == "tool_call":
                tool_output = str(output_data.get("output", "")).lower()
                tool_overlap = text_overlap_score(tool_output, response) if tool_output else 0.0

                if tool_overlap < 0.05 and tool_output:
                    blame = 0.7
                    reasons.append(f"Tool '{name}' output was IGNORED — wasteful call")
                elif not tool_output:
                    blame = 0.5
                    reasons.append(f"Tool '{name}' returned empty output")
                else:
                    blame = 0.1
                    reasons.append(
                        f"Tool '{name}' output used in response "
                        f"({tool_overlap:.0%} overlap)"
                    )

            elif step_type == "llm_call":
                # LLM is always somewhat responsible for the final output
                token_usage = step.get("token_usage", {})
                total_tokens = token_usage.get("total_tokens", 0) if token_usage else 0

                if is_failure:
                    blame = 0.6 * position_weight
                    reasons.append(
                        "LLM generated the final response — shares blame for quality issues"
                    )
                    if total_tokens > 5000:
                        blame += 0.1
                        reasons.append(f"High token usage ({total_tokens}) suggests inefficiency")
                else:
                    blame = 0.05
                    reasons.append("LLM produced a good response")

            elif step_type == "reasoning":
                # Reasoning steps are generally positive
                blame = 0.05
                reasons.append("Reasoning step present — this is good")

            else:
                blame = 0.2 * position_weight
                reasons.append(f"Unknown step type: {step_type}")

            credits.append({
                "step_index": i,
                "step_type": step_type,
                "name": name,
                "blame_score": round(blame, 2),
                "is_bottleneck": blame >= 0.5,
                "reasons": reasons,
            })

        # === Causal credit assignment pass ===
        # For each step, compute unique information contribution
        step_outputs = []
        for step in steps:
            od = step.get("output_data", {})
            output = str(od.get("output", "") or od.get("response", "") or od.get("documents", ""))
            step_outputs.append(output.lower())

        if response and any(step_outputs):
            from retune.utils.text_similarity import unique_information_score

            for i, credit in enumerate(credits):
                if len(step_outputs[i]) < 20:
                    credit["causal_contribution"] = 0.0
                    continue

                other_outputs = [o for j, o in enumerate(step_outputs) if j != i]
                causal = unique_information_score(
                    step_outputs[i], response, other_outputs
                )
                credit["causal_contribution"] = round(causal, 3)

                # Blend heuristic blame with causal analysis
                # High causal contribution = useful step = lower blame
                heuristic_blame = credit["blame_score"]
                causal_adjustment = max(0, 1.0 - causal * 3)
                credit["blame_score"] = round(
                    0.4 * heuristic_blame + 0.6 * causal_adjustment, 2
                )
                credit["is_bottleneck"] = credit["blame_score"] >= 0.5

                if causal > 0.05:
                    credit["reasons"].append(
                        f"Causal: {causal:.0%} unique info contributed to response"
                    )

        # Find top bottlenecks
        bottlenecks = [c for c in credits if c["is_bottleneck"]]
        bottlenecks.sort(key=lambda x: x["blame_score"], reverse=True)

        # Find top causal contributors
        contributors = sorted(
            [c for c in credits if c.get("causal_contribution", 0) > 0.05],
            key=lambda x: x.get("causal_contribution", 0),
            reverse=True,
        )

        # Summary
        if bottlenecks:
            top = bottlenecks[0]
            summary = (
                f"Primary bottleneck: [{top['step_type']}] {top['name']} "
                f"(blame={top['blame_score']:.2f}). {top['reasons'][0]}"
            )
        elif contributors:
            top_c = contributors[0]
            summary = (
                f"Execution healthy. Top contributor: [{top_c['step_type']}] "
                f"{top_c['name']} ({top_c['causal_contribution']:.0%} unique info)."
            )
        else:
            summary = "No significant bottlenecks found — execution looks healthy."

        return {
            "credits": credits,
            "bottlenecks": bottlenecks,
            "contributors": contributors,
            "summary": summary,
            "overall_score": round(overall_score, 2),
            "is_failure": is_failure,
        }
