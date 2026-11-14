"""Multi-agent team: a PI coordinator + omics-specialist agents (centralized architecture).

Mirrors the paper's "Multiple agents -> Centralized architecture" (e.g. The Virtual Lab) and
modality-specialised agents connected through one interface (PromptBio, DrBioRight 2.0).
"""
from __future__ import annotations

from . import rag
from .agent import Agent
from .tools import Toolbox

SPECIALISTS = {
    "transcriptomics": ("mrna", "mRNA (RNA-seq gene expression)"),
    "mirna": ("mirna", "miRNA-seq (microRNA expression)"),
    "proteomics": ("protein", "RPPA protein / phospho-protein abundance"),
}

_SPECIALIST_TOOLS = ["describe_modality", "differential_features", "pca_summary",
                     "classify_subtype", "make_plot"]

_RAG = {"search_knowledge": (rag.search_knowledge, rag.SEARCH_SCHEMA)}


def make_specialist(key: str, **agent_kwargs) -> Agent:
    modality, desc = SPECIALISTS[key]
    system = (
        f"You are the {key} specialist in a breast-cancer multi-omics team. You can ONLY access "
        f"the '{modality}' layer ({desc}) of the TCGA-BRCA training data (PAM50 subtypes Basal, "
        "Her2, LumA). Use your tools to answer the PI's request; interpret top features using "
        "search_knowledge and cite sources as [source]. If the knowledge base has nothing on a "
        "feature, say 'no KB evidence' rather than guessing. Report concisely (<200 words) with "
        "the key numbers (t, FDR, accuracy) from your tools."
    )
    tb = Toolbox(_SPECIALIST_TOOLS, allowed_modalities=[modality], extra=_RAG)
    return Agent(name=key.capitalize(), system=system, toolbox=tb, **agent_kwargs)


def make_integrator(**agent_kwargs) -> Agent:
    system = (
        "You are the integration specialist. You combine omics layers to predict PAM50 subtype. "
        "Compare single layers, early fusion and late (uncertainty-weighted) fusion with "
        "classify_subtype, then evaluate on the held-out test split with predict_on_test. "
        "Explicitly note any layer missing from the test split and how it was handled. "
        "Report concisely (<200 words) with numbers."
    )
    tb = Toolbox(["list_modalities", "classify_subtype", "predict_on_test"])
    return Agent(name="Integrator", system=system, toolbox=tb, **agent_kwargs)


def make_team(**agent_kwargs) -> dict[str, Agent]:
    team = {k: make_specialist(k, **agent_kwargs) for k in SPECIALISTS}
    team["integration"] = make_integrator(**agent_kwargs)
    return team


PI_SYSTEM = (
    "You are the Principal Investigator (PI) coordinating a multi-omics team analysing TCGA "
    "breast cancer (PAM50 subtypes Basal, Her2, LumA). You cannot see data yourself: delegate "
    "focused sub-tasks to specialists with the `delegate` tool ({names}). Plan first, delegate "
    "(you may delegate several tasks in parallel), then synthesise a final markdown report with: "
    "1) key discriminative features per layer and their biology (keep the specialists' [source] "
    "citations), 2) cross-omics consistency (e.g. the same pathway seen in mRNA and protein), "
    "3) which layer(s) best predict subtype and the fusion result, 4) limitations. "
    "Only state numbers that specialists reported."
)


def make_pi(team: dict[str, Agent], **agent_kwargs) -> Agent:
    def delegate(specialist: str, task: str) -> str:
        return team[specialist].run(task)

    schema = {
        "description": "Send a focused task to one specialist agent and get its report back.",
        "input_schema": {
            "type": "object",
            "properties": {
                "specialist": {"type": "string", "enum": list(team)},
                "task": {"type": "string"},
            },
            "required": ["specialist", "task"],
            "additionalProperties": False,
        },
    }
    tb = Toolbox([], extra={"delegate": (delegate, schema)})
    return Agent(name="PI", system=PI_SYSTEM.format(names=", ".join(team)), toolbox=tb,
                 **agent_kwargs)
