# Third-party sources and artifact boundaries

This project independently implements its pipeline and cites the work that
informed it. The terms of external data, models, packages and templates remain
separate from the [MIT license](LICENSE) for this project's original code and
documentation. The MIT grant does not apply to the external dataset, model
assets, dependency packages or the vendored NeurIPS style file.

## Dataset and published punctuation model

- Source: [Kurt & Çayır's release](https://github.com/uygarkurt/Turkish-Punctuation-Restoration),
  pinned CSV revision `7ede7a4ba9e51dad085a7f0d9b6c0a9c32622397`.
  [Exact URLs and hashes](docs/data-manifest.json) are public. The dataset license
  and exact derivation/split provenance remain unresolved. Corpus text and raw
  examples are not redistributed here.
- Published checkpoint: [uygarkurt/bert-restore-punctuation-turkish](https://huggingface.co/uygarkurt/bert-restore-punctuation-turkish),
  revision `dd8b9481ed3b3c05110d7e041ecbeb6a1c23b8b0`. Its model card declares MIT;
  this does not establish the dataset's license. No checkpoint weights are included.
- Base encoder: [BERTurk](https://github.com/stefan-it/turkish-bert),
  `dbmdz/bert-base-turkish-cased` revision `b6e1de16c983e0f2c70664591ea3f22810072608`.
  Dependencies and model assets retain their respective upstream terms.

## NeurIPS report template

`report/neurips_2026.sty` is distributed **unchanged** from the template linked
by the [official NeurIPS 2026 call](https://neurips.cc/Conferences/2026/CallForPapers):
[official ZIP](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip).
The style file credits Roman Garnett and previous style authors in its header.
The [official template listing on Overleaf](https://www.overleaf.com/latex/templates/formatting-instructions-for-neurips-2026/bjdwqfdkyftc)
attributes it to the NeurIPS 2026 Program Chairs and lists CC BY 4.0.
This project does not relicense that template or claim conference affiliation,
submission, acceptance or peer review. The report uses the `preprint` option.
The standalone proposal includes a byte-identical copy at
`report/proposal/neurips_2026.sty`, with the same attribution and external terms,
and also uses the `preprint` option.

- Official ZIP SHA-256: `82473931e3ef710fcd3f4a8cd4119b9de32e56825f90f9e5a6d55f2d01b817d9`.
- Unchanged style SHA-256: `c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a`.

## Papers and software

References in `report/main.tex` and the protocol pages identify the literature
adapted for this study. No paper text or full figures were copied into the
report. The scientific plots are generated from this project's aggregate
measurements. Python dependencies retain their upstream licenses; they are
listed in `pyproject.toml`, `environment.yml`, and the recorded Colab constraints.
