Replication Package
Group SPR-8 - Utrecht University

==================================================
1. Repository layout
==================================================
This package contains:

A) Cohen’s Kappa agreement calculations
   - Folder: Cohens Kappa Calculations/
     - Input datasets/
       - g04_raw.csv
       - g13_raw.csv
       - g16_raw.csv
       - g17_raw.csv
       - total_dataset.csv
     - Results/
       - g04_raw_kappa_results_YYYYMMDD_HHMM.csv
       - g13_raw_kappa_results_YYYYMMDD_HHMM.csv
       - g16_raw_kappa_results_YYYYMMDD_HHMM.csv
       - g17_raw_kappa_results_YYYYMMDD_HHMM.csv
       - total_dataset_kappa_results_YYYYMMDD_HHMM.csv
     - cohenskappa.py

B) Zip file of the AQUSA program
   - File: agusa-core.zip
   - When unzipped, main folder: agusa-core/
     - input/                                       # place source files for analysis
     - output/                                      # AQUSA writes results here
     - aquascore.py
     - models.py
     - test_analyzer.py
     - requirements.txt
     - README
     - LICENSE
     - corefiles/, .git/, .DS_Store

C) Link to Google Spreadsheet (https://docs.google.com/spreadsheets/d/1i0oNQ61ctDdM08c_NrNVXtTSC15ataeHpTSNcX-D-n0/edit?usp=sharing)
   - File: Link to Google Spreadsheet.rtf
   - Tabs available after opening the sheet:
     - Labeling Max
     - Labeling Guusje
     - Labeling Matthias
     - Final Dataset - g17 - cask
     - Final Dataset - g04 - recycling
     - Final Dataset - g13 - planningpoker
     - Final Dataset - g16 - mis
     - Group SPR-8 Combined Labeling
     - Results - Precision/Recall
     - Results - Cohen's Kappa
     - CombinedEvaluation


==================================================
2. Explanation of the Google Sheets
==================================================
Tabs and purpose:

- Legend
  Definitions of the QUS dimensions.

- Labeling Max
  Raw labels from rater Max.

- Labeling Guusje
  Raw labels from rater Guusje.

- Labeling Matthias
  Raw labels from rater Matthias.

- Final Dataset - g17 - cask
  Ground-truth labels for dataset g17.

- Final Dataset - g04 - recycling
  Ground-truth labels for dataset g04.

- Final Dataset - g13 - planningpoker
  Ground-truth labels for dataset g13.

- Final Dataset - g16 - mis
  Ground-truth labels for dataset g16.

- Group SPR-8 Combined Labeling
  Overview of per-item votes from all raters and the consensus decision( user stories are excluded when no consensus is met).

- Results - Precision/Recall
  Precision, recall, and F1 per model, per dataset and the combined datasets, computed against the ground truth.

- Results - Cohen's Kappa
  Cohen’s Kappa per model, per dataset and the combined datasets. computed against the ground truth.

- CombinedEvaluation
  Aggregated view that shows the counted results of each dataset.

==================================================
3. Reproduction steps (quick path)
==================================================

3.1 AQUSA evaluation

1) Unzip agusa-core.zip into the repo (folder name: agusa-core/).

2) Create a Python environment and install requirements:
   - cd agusa-core
   - python -m venv .venv
   - Activate the environment
   - pip install -r requirements.txt

3) Prepare input:
   - Place the dataset CSVs to analyze in agusa-core/input/.
   - Use the Final Dataset CSV exports from the Google Sheet, or the files under Cohens Kappa Calculations/Input datasets/.

4) Run AQUSA:
   - Use the command(s) documented in agusa-core/README.
   - Typical flow:
     - Run aquascore.py or test_analyzer.py on each CSV in input/.
     - Inspect generated files in output/.

5) Convert AQUSA outputs to the label schema:
   - One row per user story.
   - Columns: Atomic, Minimal, Independent, Uniform, Unique with values {true,false,unknown}.
   - Save each processed AQUSA result to /Cohens Kappa Calculations/Results/ or keep under agusa-core/output/ with clear filenames.


3.2 LLM evaluation

1) Use the prompt template from the paper appendix or your prompt file.

2) For each dataset:
   - Run the prompt in the designated Web interface of the LLM.
   
   - OpenAI — GPT models: https://platform.openai.com/docs/models
   - Anthropic — Claude 4.5 Sonnet: https://claude.ai/new
   - Google — Gemini Flash: https://ai.google.dev/gemini-api/docs/models/gemini
   - DeepSeek — V3.1: https://www.deepseek.com/
