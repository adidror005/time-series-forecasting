# Put this project on GitHub

Unzip `electricity_forecasting_github.zip` first. Upload the extracted project
files rather than the ZIP, so GitHub can show the notebook, Markdown, and CSVs.

## Add it to an existing repository

1. Open your repository on GitHub.
2. Select **Add file → Upload files**.
3. Drag the extracted `electricity-forecasting` folder into the upload area.
4. Commit the upload. The project will live in its own subfolder.

This preserves the existing repository's root README and other series files.
If you already have a folder with this name, review the matching files before
committing updates. This archive contains no Git history or nested ZIP files.

## Use a dedicated repository

Create an empty repository. Upload the *contents* of `electricity-forecasting`
to the repository root. `README.md` and `electricity_models.ipynb` should then
appear directly on the repository's main page. You can also copy these files
into a local clone and commit with Git or GitHub Desktop. Include `.gitignore`
and `.gitattributes` when copying; macOS Finder may hide these two files.

The `.gitattributes` file preserves CSV bytes so the notebooks' pinned checksums
continue to match. `.gitignore` excludes local caches and optional machine timing
files, while retaining the included datasets, predictions, and fitted models.

## Read the CSV from GitHub

The notebook runs with its bundled data without edits. To fetch your hosted copy,
replace `YOUR_USERNAME` and `YOUR_REPO` in one of these examples and assign it to
`DATA_URL` in the configuration cell.

If the project is in the `electricity-forecasting` subfolder:

```python
DATA_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/electricity-forecasting/data/electricity_hourly_multifeature.csv"
```

If the project files are at the repository root:

```python
DATA_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/data/electricity_hourly_multifeature.csv"
```

Change `main` if your default branch has another name. Use the unchanged expanded
CSV for the expanded notebook. GitHub's **Raw** button on that CSV also gives you
its download URL. Retain the dataset attribution in `DATA_SOURCE.md`.

## Upload sizes

This bundle has 93 files. Its largest individual file is
15.91 MiB, below GitHub's browser upload limit of 25 MiB per file.
GitHub allows up to 100 files in one browser upload. All included files fit those
limits after extraction; the ZIP itself is larger than the per-file browser limit.

Reference: [GitHub: Adding a file to a repository](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository).
