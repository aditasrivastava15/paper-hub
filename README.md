# Paper readings

This project is the single place for research papers you study from the site you use. Each paper is its own project, and every one of those projects lives in [`papers/`](papers/).

The first project is [Neural Machine Translation by Jointly Learning to Align and Translate](papers/bahdanau-align-and-translate/README.md). That reading came from the Cursor project “reading on paper a week.” The summary, notes, formulas, and quiz from that chat are in its folder.

## Add the next paper

From the root of this project:

```bash
python3 tools/paper.py new "The paper title" --url "https://..." --website "the site"
```

That creates `papers/<slug>/` with its own notes, questions, and summary. It also refreshes [PAPERS.md](PAPERS.md) and [paper-readings.code-workspace](paper-readings.code-workspace).

Other commands:

```bash
python3 tools/paper.py list
python3 tools/paper.py check
python3 tools/paper.py index
```

`check` confirms that `papers/` still contains only paper projects. `index` rebuilds the catalog after you edit a title, link, or status in `paper.json`. Status is `unread`, `reading`, or `read`.

## Open the papers as separate projects

Open `paper-readings.code-workspace`. The sidebar lists this home project and each paper as its own folder. They stay inside `papers/` rather than scattering into other coursework.

Write your own notes in each paper folder. Leave the full paper on the source site.
