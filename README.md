# ARISE (Amorphous Representation for Implicit Structural Embedding)
### Inference Code & Minimal Example

This repository provides the inference implementation and minimal example data for the model described in our manuscript:

> **"Machine-learning-enabled implicit structural embeddings for predicting composition-structure-property relationships in amorphous materials"**  
> (Submitted)

Because the original datasets (e.g., SciGlass) are subject to license restrictions, raw training data cannot be redistributed.
However, this repository provides **full inference reproducibility**, including:

- Complete model architecture (`ProposedModel`)
- JSON-based hyperparameter configuration
- Minimal toy property & test data showing correct input formatting
- Complete inference pipeline

---

## 🧩 Repository Structure
```
ARISE/
│
├── src/
│   ├── __init__.py
│   ├── model.py                # Model architecture (encoder, attention, pooling, decoder)
│   ├── dataset.py              # EmbeddingSequenceDataset
│   ├── utils.py                # inference() function
│   └── inference.py            # CLI inference script
│
├── configs/
│   └── viscosity_best_config.json   # Optimized hyperparameters (from TPE search)
│
├── data/
│   ├── properties_example.csv       # Minimal toy property table (open-source based)
│   └── test_data_example.csv        # Minimal toy test example (100% SiO₂ @ 1500°C)
│
├── models/
│   └── viscosity_model_best_trial.pth                 # Also can add model checkpoint manually
│
├── results/
│   └── (generated inference outputs)
│
└── README.md
```

---

## 🚀 Running the Inference Script

You can run inference using:

```bash
python -m src.inference \
  --properties_csv data/properties_example.csv \
  --test_csv data/test_data_example.csv \
  --checkpoint models/your_model_checkpoint.pth \
  --output_csv results/predictions.csv
  ```

---

## 📦 Included Example Data (Toy Dataset)

`properties_example.csv`
- Built from a small subset of **open-source Mendeleev** elemental data
- Demonstrates correct formatting of each property dimension
- Does not contain SciGlass data or derived values

`test_data_example.csv`

A symbolic example containing 100% SiO₂ at one temperature.

Shows the correct formatting:

```bash
[component mole fractions] + [temperature] + [dummy target]
```
- Provided solely to demonstrate expected input formatting
- Does not represent physical values

These toy datasets allow users to understand how to format their own data for inference.

---

## 🔒 Data Availability
The full datasets used during the study include:
- **SciGlass** viscosity and transport-property database
    - Licensed source: https://github.com/epam/SciGlass
    - Raw data cannot **be redistributed** under the SciGlass license

- **Mendeleev** open-source elemental property database
    - MIT licensed: https://github.com/lmmentel/mendeleev
    - Portions of these data were used to construct feature tables

The raw training datasets cannot be included in this repository.
However, **all code, configuration, trained weights, and toy datasets needed to run inference are fully provided**, and real datasets can be substituted by the user.