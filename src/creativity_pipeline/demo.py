"""Deterministic fixtures for plumbing checks. These are NOT model generations."""
import json
from .interfaces import GenerationRequest, GenerationResult


class DemoBackend:
    def describe(self):
        return {"backend": "demo", "fixture": True, "purpose": "Offline integration test; no claims of writing quality or detector performance"}

    def count_input_tokens(self, request: GenerationRequest):
        return max(1, (sum(len(x["content"]) for x in request.messages) + 3) // 4)

    def generate(self, request: GenerationRequest):
        task = request.metadata["task"]
        history = task["mode"] == "history"
        if request.role == "plan":
            index = int(request.metadata["plan_id"][1:]) - 1
            approaches = (["Compare the two districts", "Begin with the conflicting reports", "Separate burden from collection", "Organize around what remains unknown"] if history else
                          ["A quiet discovery through a repair", "A conversation with a neighbor", "An object changes the narrator's memory", "An unfinished practical task"])
            move = {"id": "M1", "description": approaches[index % len(approaches)], "claim_status": "inference" if history else "fiction",
                    "source_ids": [task["sources"][0]["id"]] if history else [], "support_quotes": [], "depends_on": []}
            value = {"id": request.metadata["plan_id"], "approach": approaches[index % len(approaches)],
                     "intent": "FIXTURE: exercise a structured plan and the checks", "moves": [move],
                     "presentation": ["M1"], "open_options": []}
        elif request.role == "semantic_critic":
            from .critic import ENGLISH_CRITERIA, HISTORY_CRITERIA
            prose = request.metadata["draft"]["text"]
            span = prose.split("\n\n")[0]
            criteria = HISTORY_CRITERIA if history else ENGLISH_CRITERIA
            value = {
                "mode": task["mode"], "decision": "accept", "coverage": "complete",
                "coverage_note": "FIXTURE: report-shape exercise only; no semantic or quality assessment occurred.",
                "rubric": [{"criterion": criterion, "judgment": "adequate", "draft_quote": span,
                            "explanation": "FIXTURE: this quoted passage exercises the rubric contract only."}
                           for criterion in criteria],
                "source_assessments": [], "evidence_findings": [], "issues": [],
                "revision_priorities": [], "preserve": ["FIXTURE: no literary judgment was performed."],
                "missing_evidence": [],
            }
            if history:
                value["source_assessments"] = [
                    {"source_id": source["id"], "useful_for": "FIXTURE: the statements in this supplied passage.",
                     "limitations": "FIXTURE: truth and independence are not established.",
                     "context_basis": "unknown", "context_quotes": [], "related_source_ids": [],
                     "independence": "unknown"} for source in task["sources"]
                ]
                source = task["sources"][0]
                value["evidence_findings"] = [{
                    "draft_quote": span, "claim_kind": "source_assertion", "support_level": "supported_as_attributed",
                    "evidence": [{"source_id": source["id"], "quote": source["text"].split(".")[0].strip(), "relation": "supports"}],
                    "explanation": "FIXTURE: links an attribution to supplied text without establishing its truth.",
                }]
        elif request.role.startswith("review"):
            findings = request.metadata.get("defects") or []
            value = {"decision": "revise" if any(i.get("severity") == "material" for i in findings) else "accept",
                     "issues": [], "strengths": ["Fixture reviewer; no independent literary or factual assessment"]}
        elif request.role == "select":
            value = {"plan_id": request.metadata["candidates"][0]["plan"]["id"], "reason": "Fixture selects first mechanically eligible plan"}
        elif request.role in {"draft", "repair"}:
            if history:
                source = task["sources"][0]
                quote = source["text"].split(".")[0].strip()
                sentence = f'The supplied account states: "{quote}" [{source["id"]}].'
                text = sentence + "\n\nThis assertion is evidence of what the account reports. It does not, by itself, establish the cause of the events. The source's perspective and corroboration would need assessment before drawing a broader conclusion.\n\nThis deliberately brief fixture demonstrates source linkage and an explicit limit on inference. It is not a trained model's historical analysis."
                value = {"text": text, "claims": [{"id": "C1", "text": sentence, "source_ids": [source["id"]], "support_quotes": [{"source_id": source["id"], "quote": quote}], "claim_status": "attested"}], "state_updates": {"world": [], "reader": []}}
            else:
                text = "Mara found the loose hinge after she had packed the photographs.\n\nShe could have left it for the next tenant. Instead she sat on the kitchen floor with the screwdriver, holding the cupboard door against her shoulder. A small wedge of folded paper fell from behind the plate. There was no message on it; it had only kept the door level.\n\nFor years she had remembered her father as someone who finished nothing. The paper did not disprove that. But somebody had noticed the door. Somebody had wanted it to close.\n\nShe tightened the screw, then loosened it a quarter turn. It worked better that way."
                value = {"text": text, "claims": [], "state_updates": {"world": [{"id": "mara_hinge", "text": "Mara found a loose hinge after packing photographs.", "evidence": "Mara found the loose hinge after she had packed the photographs."}], "reader": []}}
        else:
            raise ValueError(f"Unsupported fixture role {request.role}")
        text = json.dumps(value, ensure_ascii=False)
        output_tokens = max(1, (len(text) + 3)//4)
        return GenerationResult(text=text, input_tokens=self.count_input_tokens(request), output_tokens=output_tokens,
                                finish_reason="stop" if output_tokens <= request.max_new_tokens else "length",
                                accounting="fixture_estimate", metadata={"fixture": True})

