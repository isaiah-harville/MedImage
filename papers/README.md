# Papers

Conference drafts (IEEE, `IEEEtran`) on published runs. Each folder is one paper:

| File | Holds |
|---|---|
| `main.tex`, `refs.bib` | the source |
| `make_figures.py` | draws `figures/*.pdf` from the run's committed files |
| `main.pdf` | the built paper, committed so the docs build needs no TeX |
| `paper.yaml` | `title`, `project`, `run`, and `pdf`: the published run the paper reports |

`opendiagnostics notebook` copies each PDF into the docs site and links it from the Papers
page, its run's page, and the dataset table. The build stops if a paper names a run that is
not published or a PDF that does not exist.

Build a paper from its folder:

```bash
uv run python papers/<name>/make_figures.py
cd papers/<name> && tectonic -X compile main.tex
```

Every number in a paper comes from its run's files under `models/<project>/runs/<run>/`.
Rebuild the PDF after any change to the source and commit both.
