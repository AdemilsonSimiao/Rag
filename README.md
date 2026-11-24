*This project has been created as part of the 42 curriculum by adedias-.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about the
[vLLM](https://github.com/vllm-project/vllm) 0.10.1 repository (its source code
and its documentation).

Given a question, the system:

1. finds the most relevant pieces of the repository (**retrieval**),
2. gives them to a small language model, `Qwen/Qwen3-0.6B`, so that it can write an
   answer based on them (**generation**),
3. and reports where the answer came from, as file paths with character ranges.

The goal of the project is to build the whole pipeline ourselves: chunking,
indexing, ranking, prompting and evaluation. Retrieval quality is measured with
**recall@k**, and the main targets are at least **80 % recall@5 on documentation
questions** and **50 % on code questions**.

## Instructions

### Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)
- The vLLM repository placed in `data/raw/vllm-0.10.1/` (not included in this
  repository)
- Enough RAM and an internet connection (the first time) to download and run
  `Qwen/Qwen3-0.6B` (about 1.2 GB), only needed for the `answer` commands

### Installation

```bash
make install        # runs `uv sync`
```

### Expected data layout

```text
data/
├── raw/vllm-0.10.1/...                 # the corpus to index
├── datasets/
│   ├── UnansweredQuestions/*.json      # questions only
│   └── AnsweredQuestions/*.json        # questions, answers and expected sources
├── processed/                          # created by `index`
└── output/                             # created by the search/answer commands
```

The `data/` folder is ignored by git. All paths are command-line arguments;
the ones shown are only defaults.

### Commands

Every command is run from the repository root as
`uv run python -m src <command> [--options]`.

| Command | What it does |
|---|---|
| `index` | Reads `data/raw/`, chunks the files and saves the BM25 index in `data/processed/`. |
| `search` | Prints the top-k sources for one query. |
| `search_dataset` | Searches every question of a dataset and writes a `StudentSearchResults` JSON file. |
| `answer` | Searches and answers one query with the language model. |
| `answer_dataset` | Answers every question of a search-results file and writes a `StudentSearchResultsAndAnswer` JSON file. |
| `evaluate` | Computes recall@1, @3, @5 and @10 against a ground-truth dataset (for our own tests). |

`make run` and `make debug` start the program (and the Python debugger) without a
command; they print the list of commands.

### Development

```bash
make lint           # flake8 + mypy
uv run pytest       # unit tests
make clean          # removes caches
```

## System architecture

The pipeline has two phases: **indexing** (done once) and **querying**.

```text
 INDEXING (command: index)
 data/raw/ ──► read files ──► chunk (by file type) ──► BM25 index ──► data/processed/index.pkl
               .py .md .txt    parser.py                retriever.py

 QUERYING
 question ──► tokenize ──► BM25 scores ──► top-k chunks ──► search results (JSON)
                               ▲                                  │
                         index.pkl                                ▼
                                          read the text of each source ──► prompt ──► Qwen3-0.6B ──► answer (JSON)
                                                                                generator.py
```

| Module | Responsibility |
|---|---|
| `src/models.py` | Pydantic models for every structure exchanged between stages (`MinimalSource`, `Chunk`, `StudentSearchResults`, ...). |
| `src/parser.py` | The two chunkers: `chunk_python` and `chunk_markdown`. |
| `src/indexer.py` | Finds the files, chunks them, builds and loads the index. |
| `src/retriever.py` | Tokenization and the `Bm25Index` class (build, search, save, load). |
| `src/question.py` | Runs one search and keeps only the fields the output format needs. |
| `src/passages.py` | Reads the text of the retrieved sources from the corpus. |
| `src/generator.py` | Prompt building, context budget and the Qwen model. |
| `src/evaluator.py` | Recall@k with the same overlap rule as the official tool. |
| `src/cli.py`, `src/__main__.py` | The command-line interface (Python Fire). |
| `src/error.py` | Prints an error message and exits cleanly. |

The index stores each chunk's file path, character range and text, so searching
never needs to open the corpus again.

## Chunking strategy

Only `.py`, `.md` and `.txt` files are indexed. In the public datasets, every
expected source of a code question is a `.py` file and every expected source of
a documentation question is a `.md` or `.txt` file.

Each file type has its own strategy. In both, a chunk never exceeds
`max_chunk_size` characters (2000 by default), and positions are counted in
**characters** of the original file.

**Python files: fixed windows with overlap.** The file is cut into windows of
`max_chunk_size` characters, and each window starts `max_chunk_size - overlap`
characters after the previous one. The overlap is 10 % of the window (200
characters by default), so a passage that falls on the boundary between two
windows is still complete in one of them.

We chose this after looking at the data: in the public code dataset, only 41 of
99 expected spans start at a `def`, `class` or decorator. The others start in
the middle of a function or at module level, so splitting by functions would
often miss the expected region.

**Markdown and text files: sections.**

1. A new section starts at every heading (`#` to `######` followed by a space).
   Headings inside fenced code blocks are ignored, so a shell comment such as
   `# install` is not mistaken for a heading. Text before the first heading is
   its own section.
2. A section longer than the limit is cut into pieces, preferably at a blank
   line (between paragraphs), then at a line break, and only as a last resort at
   the limit itself.
3. Short sections are kept as they are.
4. `.txt` files have no headings, so only the size rule applies.

In the public documentation dataset, 79 of 100 expected spans start exactly at a
heading, which is why a section is the natural unit. With this strategy, the 206
Markdown and text files give 1630 chunks, and an "oracle" that always picked the
best chunk reaches 100 % recall.

With the default settings, the 1969 indexed files give **12 879 chunks**.

## Retrieval method

The retriever is **BM25** (k1 = 1.5, b = 0.75), implemented from scratch with
NumPy, without a retrieval library.

**Tokenization** (the same for the index and for the question):

- everything is lowercased and split into words (letters, digits and `_`);
- an identifier is kept whole **and** split into its parts: `fused_batched_moe`
  gives `fused_batched_moe`, `fused`, `batched`, `moe`, and `LoRAAdapter` gives
  `loraadapter`, `lo`, `ra`, `adapter`. A question can quote an identifier
  exactly or only describe the idea, and both match;
- there is no stopword list: BM25's inverse document frequency already gives
  little weight to very common words;
- the **file path** is added to the text of each chunk before tokenizing, because
  the file name often holds the key word of the question (for example `lora.md`).

**Ranking.** At indexing time, the BM25 score of every (term, chunk) pair is
computed once and stored as an inverted index of NumPy arrays. A query only sums
the stored scores of its terms, and the top-k chunks are selected with
`lexsort`. The result is a list of chunks with their scores, best first.

## Performance analysis

Retrieval quality, measured with the official `moulinette` on the public datasets
(`k = 10`):

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Target (@5) |
|---|---|---|---|---|---|
| Documentation (100 questions) | 63.0 % | 78.0 % | **81.0 %** | 87.0 % | 80 % |
| Code (99 questions) | 63.6 % | 76.8 % | **80.8 %** | 89.9 % | 50 % |

The `evaluate` command gives the same numbers as the official tool.

Speed:

- indexing the whole corpus takes a few seconds (limit: 5 minutes);
- searching the 100 questions of a dataset takes under a second once the index
  is loaded.

Observations and limits:

- The margin on documentation is small (81 % against a target of 80 %). The
  reference datasets are public, but other questions could score differently.
- Generation was only run on a constrained development machine (a WSL
  environment with 3.8 GB of RAM, which was swapping), where it took about 150
  seconds per question on CPU. This number reflects that machine, not the
  system's design; it was not measured on a machine with more memory.

## Design decisions

- **BM25 instead of embeddings.** It is fast, needs no model to index, and
  works well when questions share words with the source (identifiers, endpoint
  names, file names).
- **A hand-written BM25 index.** It is short, easy to explain and test, and
  avoids an extra dependency. Scores are precomputed, which makes searching
  very fast.
- **Two chunkers, chosen from the data**, as explained above.
- **Hard limit of 2000 characters per chunk.** The official tool rejects the
  whole output if a single source is longer, so the chunkers enforce it.
- **Output with only the required fields.** Search results contain the file
  path and the two positions; the chunk text stays inside the index.
- **Context budget for the model.** At most 8000 characters of context are
  sent, best-ranked sources first, and the prompt tells the model to use only
  that context and to say so if the answer is not there. Generation is greedy
  (deterministic) and the model's "thinking" mode is turned off.
- **The model is not called when nothing was retrieved**, so it is never asked
  to invent an answer.
- **Heavy imports are lazy.** `torch` and `transformers` are imported only when
  a model is needed, so the search commands start instantly.
- **Robust command line.** Empty queries, `k <= 0`, missing files, malformed
  JSON and a model that cannot be loaded all print a clear message and exit with
  status 1, without a traceback. Every path is a command-line argument.
- **Typed, tested code.** Type hints everywhere, Pydantic models for every
  exchanged structure, and unit tests for the chunkers, retriever, indexer,
  generator and CLI (the model is replaced by a fake one in tests).

## Challenges faced

- **Expected spans do not follow the structure of the code.** Splitting Python
  by functions looked natural, but most reference spans start in the middle of
  one. We measured it on the dataset and used overlapping windows instead.
- **Matching the official overlap rule.** A first version of `evaluate` counted
  any overlap as a hit, which gave about one point more than the official tool.
  A source counts only if it is in the same file and the character ranges have
  an IoU of at least 0.05; with that rule our numbers match the official ones.
- **Positions must be exact.** The official tool compares file paths literally
  and uses character positions, so paths are written relative to the repository
  root (`data/raw/vllm-0.10.1/...`) and offsets are counted in characters.
- **Headings inside code blocks.** A naive "line starts with `#`" rule split
  Markdown files at shell comments; fenced blocks are now tracked and ignored.
- **Running the model on limited hardware.** On a machine with little memory the
  model was very slow because of swapping. Keeping the context small and the
  model loaded only once per command helps; more memory solves it.

## Example usage

```bash
# 1. Build the index (a few seconds)
uv run python -m src index

# 2. Search one query
uv run python -m src search "How do I load a LoRA adapter at runtime?" --k 3

# 3. Search a whole dataset
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results

# 4. Measure recall with our own command
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

# 5. Answer one question (downloads the model the first time)
uv run python -m src answer "How do I load a LoRA adapter at runtime?" --k 3

# 6. Answer a whole dataset
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer
```

Example output of step 5 (shortened):

```json
{
  "question": "How do I load a LoRA adapter at runtime?",
  "retrieved_sources": [
    {"file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
     "first_character_index": 6100, "last_character_index": 8082}
  ],
  "answer": "To load a LoRA adapter at runtime, send a POST request to the `/v1/load_lora_adapter` endpoint ..."
}
```

## Resources

- S. Robertson and H. Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond*, Foundations and Trends in Information Retrieval, 2009.
- P. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks*, 2020. <https://arxiv.org/abs/2005.11401>
- C. Manning, P. Raghavan and H. Schütze, *Introduction to Information
  Retrieval*, chapter 6 (term weighting and scoring).
  <https://nlp.stanford.edu/IR-book/>
- Qwen Team, *Qwen3 Technical Report*, 2025. <https://arxiv.org/abs/2505.09388>
- vLLM documentation: <https://docs.vllm.ai>
- Hugging Face Transformers documentation: <https://huggingface.co/docs/transformers>
- uv documentation: <https://docs.astral.sh/uv/>
- Python Fire: <https://github.com/google/python-fire>
- Pydantic documentation: <https://docs.pydantic.dev>

### How AI was used

- **Planning:** discussing the pipeline, drawing the flowchart, and analysing the
  public datasets (where expected spans start, their sizes) to choose the
  chunking strategies.
- **Code:** drafting the chunkers, the BM25 index, the indexer, the command-line
  interface, the generator and the evaluator, and writing the unit tests.
- **Review:** checking the code that was written by hand, finding errors (broken
  imports, an indentation error, an overlap rule that did not match the official
  one) and running the official `moulinette` to compare results.

All the code was read, run and understood before being kept, and the project
structure, the module split and several fixes were done by hand.