# Multilingual Health Question Answering in Low-Resource African Languages

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noel-odero/Zindi-Health_QA/blob/main/Zindi_health_qa.ipynb)

**Zindi Competition:** [Multilingual Health QA in Low-Resource African Languages](https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages-challenge)  
**Student:** Modestine Nformi · African Leadership University · June 2026  
**Course:** Machine Learning Techniques I — Final Project (40%)

---

## Project Overview

This project implements a **Retrieval-Augmented pipeline** for answering health questions in five African languages:

| Language | Subset Code | Script | Country |
|---|---|---|---|
| Amharic | `Amh_Eth` | Ge'ez (Ethiopic) | Ethiopia |
| Luganda | `Lug_Uga` | Latin | Uganda |
| Akan | `Aka_Gha` | Latin | Ghana |
| Swahili | `Swa_Ken` | Latin | Kenya |
| English | `Eng_Eth/Uga/Ken/Gha` | Latin | Multiple |

Rather than generating answers from scratch, the pipeline retrieves the most relevant answer from the training corpus using a `SemanticRoutingIndex`  a custom retrieval engine that combines **sparse TF-IDF** and **dense multilingual sentence embeddings** per language subset.

---

## Approach

### SemanticRoutingIndex
The core retrieval engine supports three strategies per language subset:

```
score = α × TF-IDF_similarity + (1-α) × semantic_similarity
```

- **TF-IDF only** (`α=1.0`) best for Latin-script languages with standardised medical vocabulary
- **Semantic only** (`α=0.0`)  essential for Amharic (Ge'ez script has no character overlap with Latin-script training data)
- **Hybrid** (`0 < α < 1`)  combines complementary sparse and dense signals

### Semantic Model
`paraphrase-multilingual-MiniLM-L12-v2` (118M parameters, 50+ languages) produces 384-dimensional embeddings in a shared cross-lingual space  enabling semantic retrieval for Amharic despite the Ge'ez script barrier that completely defeats TF-IDF.

---

## Experiments

Ten experiments systematically vary one component at a time:

| ID | Experiment | Variable Tested |
|---|---|---|
| E01 | TF-IDF Word N-grams | Feature type: word unigrams+bigrams |
| E02 | TF-IDF Char N-grams | Feature type: char trigrams-5grams |
| E03 | Pure Semantic Embeddings | Strategy: semantic only |
| E04 | Hybrid 50/50 | Strategy: hybrid (α=0.5) |
| E05 | Per-Language Weight Tuning | Per-subset α grid search |
| E06 | Exact-Match Lookup | + exact_match_lookup=True |
| E07 | Deduplicated Corpus | + dedup_questions=True |
| E08 | Cross-Subset Fallback | + Amharic/Luganda → English fallback |
| E09 | Similarity Threshold | + confidence threshold=0.3 |
| E10 | Final Combined Model | All above + train+val corpus |

---

## Repository Structure

```
.
├── Zindi_health_qa_MARKED.ipynb   ← Main notebook (run this)
├── README.md                      ← This file
├── NformiM_FinalProject.pdf       ← Academic report
│
├── zindi_health_qa_data/          ← Competition data (download from Zindi)
│   ├── Train.csv
│   ├── Val.csv
│   ├── Test.csv
│   └── SampleSubmission.csv
│
└── outputs/                       ← Generated automatically on first run
    ├── submission_E01.csv
    ├── submission_E02.csv
    ├── ...
    ├── experiment_summary.csv     ← ROUGE scores for all experiments
    ├── train_embeddings.npy       ← Cached sentence embeddings
    └── *.png                      ← Charts saved for report
```

---

## How to Reproduce

### Option 1 — Google Colab (Recommended)

1. Click the **Open in Colab** badge at the top of this README
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Upload competition CSV files to `MyDrive/zindi-health-qa/`
4. Run all cells (`Runtime → Run all`)

> First run encodes embeddings (~2 min on GPU) and saves to Drive.  
> Subsequent runs load from cache and skip completed experiments automatically.

### Option 2 — Kaggle Notebooks

1. Go to [kaggle.com](https://kaggle.com) → **Code** → **New Notebook**
2. Settings → Accelerator → **GPU P100**
3. Add your competition data as a Kaggle Dataset input
4. Upload `Zindi_health_qa_MARKED.ipynb` and run

### Option 3 — Local / VS Code

```bash
pip install sentence-transformers scikit-learn rouge-score transformers torch matplotlib pandas numpy tqdm
```

Place competition CSV files in `./zindi_health_qa_data/` next to the notebook and run all cells. GPU optional — CPU works for all retrieval experiments.

---

## Setup

### Dependencies

```bash
pip install sentence-transformers scikit-learn rouge-score transformers \
            torch matplotlib pandas numpy tqdm
```

All packages are installed automatically by Cell 2 of the notebook.

### Data

Download competition files from Zindi and place them at:

```
zindi_health_qa_data/
├── Train.csv
├── Val.csv
├── Test.csv
└── SampleSubmission.csv
```

---

## Evaluation

The competition uses a composite score:

```
Zindi Score = (ROUGE-1 F1 × 0.37) + (ROUGE-L F1 × 0.37) + (LLM-as-a-Judge × 0.26)
```

Local ROUGE is computed using a **whitespace tokeniser** (language-agnostic — avoids English Porter stemming being applied to Amharic or Akan). Local ROUGE consistently underestimates the Zindi composite score because retrieved human-written answers are clinically accurate and the LLM judge rewards accuracy independently of word overlap.

---

## Key Findings

- **Character n-grams outperform word n-grams** for agglutinative languages (Swahili, Luganda) where morphological variation breaks exact word matching
- **Semantic embeddings are essential for Amharic**  Ge'ez script has zero character overlap with Latin-script training data, making TF-IDF retrieval random for that subset
- **Hybrid retrieval outperforms pure methods** TF-IDF and semantic embeddings provide complementary signals that combine effectively
- **Per-language weight tuning** reveals that optimal α differs substantially across subsets (Amharic → α≈0; English → higher α)
- **Retrieval engineering compounds** exact-match, deduplication, cross-subset fallback, and confidence gating each add incremental improvement

---

## Ethical Considerations

This project addresses health communication in low-resource African language communities. Key considerations:

- **Misinformation risk:** Retrieved answers are delivered with high confidence regardless of factual accuracy. Expert corpus curation and medical disclaimers are required for any real-world deployment
- **Linguistic equity:** Amharic and Akan receive lower-quality predictions due to script incompatibility and limited training data — deploying a system with this known disparity amplifies existing health information inequality
- **Data privacy:** The training corpus contains sensitive health questions; expansion to new data sources requires appropriate consent mechanisms

---

## Citation

If referencing this work:

```bibtex
@misc{nformi2026healthqa,
  author = {Nformi, Modestine},
  title  = {Multilingual Health QA: Retrieval-Augmented Approach with Semantic Routing},
  year   = {2026},
  note   = {Zindi Competition Project, African Leadership University}
}
```

---

## References

- Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. *ACL Workshop on Text Summarization*.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT. *EMNLP 2019*.
- Reimers, N., & Gurevych, I. (2020). Making monolingual sentence embeddings multilingual. *EMNLP 2020*.
- Wolf, T. et al. (2020). Transformers: State-of-the-art NLP. *EMNLP 2020*.
- Zindi Africa. (2026). *Multilingual Health QA Challenge*. https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages-challenge

---

*AI tool disclosure: Claude (Anthropic) was used as an AI assistant for debugging and code structure suggestions. All experimental design, implementation, and analysis reflect the student's own understanding and reasoning.*