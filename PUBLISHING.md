# Publishing Guide

Since `smart-reframe` is now a standalone library, here is how you can publish it to the world.

## 1. Build the Package
Generate the distribution files (`.tar.gz` and `.whl`).

```bash
cd smart_reframe_lib
pip install build twine
python3 -m build
```

## 2. Publish to PyPI
Upload the package so others can `pip install smart-reframe`.

```bash
python3 -m twine upload dist/*
```
*(You will need a PyPI account and API token)*

## 3. Push to GitHub
Initialize a fresh repository for the library.

```bash
git init
git add .
git commit -m "Initial release: Extraction from AutoReframe"
gh repo create smart-reframe --public --source=. --remote=origin --push
```

## 4. Dependencies
Ensure your `requirements.txt` is up to date (already generated).
