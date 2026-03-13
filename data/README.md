# Data Directory

This directory holds the CSV datasets used for training and evaluation.

## Expected Files

```
data/
├── train.csv
└── test.csv
```

## CSV Schema

### `train.csv`

| Column | Type | Description |
|---|---|---|
| `id` | int | Unique row identifier |
| `book_name` | str | Title of the source novel. Must match one of the keys in `BOOK_PATHS` in `train_up.py` |
| `char` | str | Character name whose backstory is being evaluated |
| `content` | str | Proposed backstory text |
| `label` | str | Ground truth: `consistent` or `contradict` |

### `test.csv`

| Column | Type | Description |
|---|---|---|
| `id` | int | Unique row identifier |
| `book_name` | str | Title of the source novel |
| `char` | str | Character name |
| `content` | str | Proposed backstory text |

## Supported Books

The system currently supports two novels. Book plain-text files should be placed in `Books/`:

| Book title (in CSV) | File path |
|---|---|
| `The Count of Monte Cristo` | `Books/The Count of Monte Cristo.txt` |
| `In Search of the Castaways` | `Books/In search of the castaways.txt` |

To add a new book, update the `BOOK_PATHS` dictionary in both `train_up.py` and `test_up.py`.

## Label Distribution (Training Set)

The system was trained on a binary classification task:
- **1 (consistent)** — the backstory is causally and logically compatible with the novel
- **0 (contradict / inconsistent)** — the backstory contains at least one causal or logical contradiction with the novel

A roughly balanced label distribution is recommended for best results.