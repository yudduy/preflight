# preflight

Profile DPO preference datasets before training. Catches data quality issues that silently degrade RLHF performance.

## What it checks

| Check | What it detects |
|-------|----------------|
| **Length bias** | Whether "chosen" responses are systematically longer (a known reward-hacking signal) |
| **Embedding similarity** | Low-contrast pairs where chosen and rejected are nearly identical |
| **Easy pairs** | Trivially distinguishable pairs that add little learning signal |
| **Prompt coverage** | Clusters prompts to find underrepresented topic areas |
| **Duplicates** | Exact and near-duplicate prompts via embedding cosine similarity |
| **LLM judge** | Optional GPT/open-model scoring to find mislabeled or trivial pairs |

## Install

```bash
pip install .

# With LLM judge support:
pip install ".[judge]"
```

## Usage

```bash
preflight audit dataset.jsonl
```

This produces a `preflight_report.json` with per-check results and actionable recommendations, plus a printed summary.

### Options

```
--output PATH          Output JSON path (default: preflight_report.json)
--n-clusters N         Number of prompt clusters for coverage analysis (default: 8)
--embedding-model NAME Sentence-transformers model (default: all-MiniLM-L6-v2)
--judge MODEL          Enable LLM judge scoring (e.g. gpt-4o-mini)
--judge-base-url URL   Custom API base URL for judge
--judge-api-key KEY    API key for judge (or set OPENAI_API_KEY)
```

### Supported formats

- **HuggingFace** (`prompt`/`chosen`/`rejected` keys, string or message-list values)
- **Together AI** (`input.messages`/`preferred_output`/`non_preferred_output`)
- **Anthropic HH** (`chosen`/`rejected` with `\n\nHuman:`/`\n\nAssistant:` turns)

Format is auto-detected from the first line.

## Example

```bash
preflight audit examples/sample_hf.jsonl --output report.json
```

## Validation experiment

See [`experiments/dpo_validation.ipynb`](experiments/dpo_validation.ipynb) for a reproducible experiment showing that filtering preflight-flagged pairs from UltraFeedback improves DPO reward accuracy on a held-out eval set (Qwen2.5-0.5B, LoRA, Colab T4).

## License

MIT
