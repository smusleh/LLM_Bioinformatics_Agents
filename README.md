# Multi-omics LLM Agents: a small demo of Yang *et al.* (2025)

Small, readable Python notebooks that show the ideas in

> Yang T, Xiao Y, Bao Z, Hao J, Peng J. **The rise and potential opportunities of large language model agents in bioinformatics and biomedicine.** *Briefings in Bioinformatics* 26(6), bbaf601 (2025). doi:10.1093/bib/bbaf601

applied to a public **multi-omics** dataset: TCGA breast cancer mRNA + miRNA + protein, from mixOmics `breast.TCGA`. The agents use Claude through the Anthropic Python SDK.

## What each notebook shows

| Notebook | Paper concepts | What happens |
|---|---|---|
| `00_data_tour` | Action module (the tools) | No LLM. Explore the data and call the analysis tools by hand. |
| `01_single_agent_tools` | Core architecture (Fig. 1), tool use (Fig. 2), single agent (Fig. 3) | An LLM with no tools versus a tool-using agent that plans and calls tools. |
| `02_rag_memory_selfverify` | RAG (Fig. 2), memory, GeneAgent-style self-verification, hallucination | Answers cite a knowledge base, a verifier agent checks the claims, and findings carry over between sessions. |
| `03_multi_agent_pi` | Multi-agent centralized architecture (Virtual Lab, PromptBio, DrBioRight), omics heterogeneity | A PI agent delegates to transcriptomics, miRNA, proteomics and integration specialists. |
| `04_human_in_loop_critic` | Human–agent collaboration, critic, Challenges section | Human approval gates, a critic audits the report against the tool logs, and the PI writes a revised final report. |

## Run in Google Colab (no install, free LLM)
Click a badge, then **run the first code cell** (it clones this repo, installs packages and sets up the LLM):

| Notebook | |
|---|---|
| 00 Data tour | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smusleh/LLM_Bioinformatics_Agents/blob/main/notebooks/00_data_tour.ipynb) |
| 01 Single agent + tools | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smusleh/LLM_Bioinformatics_Agents/blob/main/notebooks/01_single_agent_tools.ipynb) |
| 02 RAG, memory, self-verification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smusleh/LLM_Bioinformatics_Agents/blob/main/notebooks/02_rag_memory_selfverify.ipynb) |
| 03 Multi-agent PI team | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smusleh/LLM_Bioinformatics_Agents/blob/main/notebooks/03_multi_agent_pi.ipynb) |
| 04 Human-in-the-loop + critic | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smusleh/LLM_Bioinformatics_Agents/blob/main/notebooks/04_human_in_loop_critic.ipynb) |

Pick the LLM with `BACKEND` in that first cell:
- `"ollama"` (default): free, no account. Choose **Runtime → Change runtime type → T4 GPU** first. The first run in a session takes ~3–5 min to install Ollama and download `qwen2.5:7b`. A 7B model sometimes skips steps (e.g. not saving findings in notebook 02).
- `"gemini"`: free API key from [Google AI Studio](https://aistudio.google.com/apikey), saved as a Colab secret named `GEMINI_API_KEY` (key icon in the left sidebar, with notebook access on). Faster and better at tool use. Free-tier prompts may be used by Google to improve its products; only aggregate statistics are sent.
- `"anthropic"`: Claude; Colab secret `ANTHROPIC_API_KEY` and API credits.

Notebook 04 reads the outputs of 03, so run 03 first in the same Colab session.

## Layout
```
omics_agents/
  agent.py      ~100-line agent loop (perceive → plan → act), trace printing, human_gate()
  tools.py      deterministic omics tools: DE (Welch + BH), PCA, CV classifier, early/late fusion
  rag.py        TF-IDF retrieval over knowledge/*.md (curated notes with literature sources)
  memory.py     long-term JSON memory (save_finding / recall_findings)
  team.py       PI coordinator + modality-restricted specialist agents
  data.py       downloads breast.TCGA once → data/*.csv
  config.py     backend (LLM_PROVIDER), MODEL, EFFORT, paths
  colab.py      one-call Colab setup (Ollama / Gemini / Anthropic)
notebooks/      00–04
tests/          offline test of the agent loop (fake client, no API key needed)
```

## Setup (WSL Ubuntu)
```bash
git clone https://github.com/smusleh/LLM_Bioinformatics_Agents.git
cd LLM_Bioinformatics_Agents
conda create -n omics-agents -c conda-forge python=3.11 pip
conda activate omics-agents
pip install -r requirements.txt jupyterlab
python -m ipykernel install --user --name omics-agents --display-name "Python (omics-agents)"
cp .env.example .env    # then choose a backend in .env (see the comments there)
```
In VS Code (connected to WSL), open a notebook and choose **Select Kernel → Python Environments… → omics-agents**. Run the notebooks in order: 04 reads the outputs of 03.

Headless run (approval gates are approved automatically):
```bash
conda activate omics-agents
cd notebooks && AUTO_APPROVE=1 jupyter nbconvert --to notebook --execute --inplace 01_single_agent_tools.ipynb
```

## Model and cost
`config.py` uses `MODEL = "claude-opus-5-5"` with `EFFORT = "low"`. Change it to `claude-sonnet-5-5` or `claude-haiku-5-5` for a cheaper run. Each notebook prints an approximate cost. Requests turn on server-side refusal fallbacks (`fallbacks="default"`); remove those two lines in `agent.py` if you don't want them.

## Notes and limitations
- The knowledge base is intentionally small. Some top hits (ZNF552, MEX3A, miR-505, miR-590) are left out on purpose so you can see the agent report "no KB evidence" instead of making something up.
- The LLM only receives aggregate statistics, never patient-level rows.
- This is a teaching demo, not a validated analysis pipeline.
