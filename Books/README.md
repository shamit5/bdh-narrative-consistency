# Books Directory

This directory contains the plain-text versions of the novels used for training and evaluation.

## Required Files

```
Books/
├── The Count of Monte Cristo.txt
└── In search of the castaways.txt
```

## Obtaining the Books

Both novels are in the public domain and available from [Project Gutenberg](https://www.gutenberg.org/):

- **The Count of Monte Cristo** by Alexandre Dumas — https://www.gutenberg.org/ebooks/1184
- **In Search of the Castaways** by Jules Verne — https://www.gutenberg.org/ebooks/3134

Download the plain-text (UTF-8) versions and save them with the exact filenames shown above.

## Adding New Books

To extend the system to additional novels:

1. Place the `.txt` file in this directory
2. Add an entry to `BOOK_PATHS` in both `train_up.py` and `test_up.py`:

```python
BOOK_PATHS = {
    "The Count of Monte Cristo": "Books/The Count of Monte Cristo.txt",
    "In Search of the Castaways": "Books/In search of the castaways.txt",
    "Your New Book Title":        "Books/Your New Book Title.txt",   # add here
}
```

3. Ensure `book_name` values in your CSV data match the dictionary keys exactly (case-insensitive after `normalize_book_name()`).