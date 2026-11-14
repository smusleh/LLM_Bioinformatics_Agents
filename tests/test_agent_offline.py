"""Offline test of the Agent loop + PI delegation using a scripted fake Anthropic client."""
import os
import sys
from types import SimpleNamespace as NS

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["LLM_PROVIDER"] = "anthropic"  # exercise the Anthropic path with the fake client
from omics_agents import agent as A  # noqa: E402
from omics_agents.team import make_pi, make_team  # noqa: E402
from omics_agents import tools  # noqa: E402


def tu(id_, name, inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def text(t):
    return NS(type="text", text=t)


def think(t):
    return NS(type="thinking", thinking=t)


def resp(content, stop):
    return NS(content=content, stop_reason=stop, usage=NS(input_tokens=100, output_tokens=20))


class FakeMessages:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def create(self, **kw):
        self.calls.append(kw)
        return self.script.pop(0)


def install(script):
    fm = FakeMessages(script)
    A._anthropic_client = NS(beta=NS(messages=fm))
    return fm


# --- 1. single agent: parallel tools, one bad call, then final answer
fm = install([
    resp([think("Plan: overview then DE."), tu("t1", "list_modalities", {}),
          tu("t2", "differential_features", {"modality": "mrna", "group_a": "Basal", "group_b": "Nope"})],
         "tool_use"),
    resp([tu("t3", "classify_subtype", {"modalities": ["protein"]})], "tool_use"),
    resp([text("Final answer: protein CV acc 0.9")], "end_turn"),
])
ag = A.Agent("T", "sys", toolbox=tools.Toolbox())
out = ag.run("q")
assert out == "Final answer: protein CV acc 0.9", out
assert [e["error"] for e in ag.tool_log] == [False, True, False]
last_user = fm.calls[1]["messages"][2]  # [task, assistant tool_use, user tool_results, ...]
assert last_user["role"] == "user" and len(last_user["content"]) == 2  # both results in one turn
assert last_user["content"][1]["is_error"] is True
kw = fm.calls[0]
assert kw["fallbacks"] == "default" and kw["output_config"] == {"effort": "low"}
assert "tool_choice" not in kw
print("single agent OK; cost", round(ag.cost(), 5))

# --- 2. PI delegating to a specialist (nested agent loops) + modality restriction
team = make_team(verbose=True)
fm = install([
    resp([tu("p1", "delegate", {"specialist": "proteomics", "task": "DE Basal vs LumA"})], "tool_use"),
    # proteomics specialist turn 1: tries a forbidden layer and a valid call
    resp([tu("s1", "pca_summary", {"modality": "mrna"}),
          tu("s2", "differential_features", {"modality": "protein", "group_a": "Basal", "group_b": "LumA", "top_n": 3}),
          tu("s3", "search_knowledge", {"query": "ER-alpha"})], "tool_use"),
    resp([text("ER-alpha lower in Basal [er_alpha_pr]")], "end_turn"),
    resp([text("# Report\nER-alpha ...")], "end_turn"),
])
pi = make_pi(team)
rep = pi.run("go")
assert rep.startswith("# Report"), rep
plog = team["proteomics"].tool_log
assert plog[0]["error"] and "only use modalities" in plog[0]["result"], plog[0]
assert not plog[1]["error"] and "ER-alpha" in plog[1]["result"]
assert "er_alpha_pr" in plog[2]["result"]
print("PI delegation OK")

# --- 3. refusal + max_steps
install([resp([text("")], "refusal")])
assert "declined" in A.Agent("R", "s").run("x")
install([resp([tu(f"x{i}", "list_modalities", {})], "tool_use") for i in range(3)])
assert "max_steps" in A.Agent("M", "s", toolbox=tools.Toolbox(), max_steps=3).run("x")
print("refusal / max_steps OK")
