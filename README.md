# NADI Arabic Dialect Classification

A Transformer-based Arabic dialect identification system built using **MARBERT**, with region-specific fine-tuning across five major Arabic dialect groups.

## Overview

Arabic dialect identification is a challenging Natural Language Processing task due to the linguistic diversity between Arabic-speaking regions. This project addresses the problem using **MARBERT**, a Transformer language model specifically designed for Arabic.

The system fine-tunes dedicated MARBERT classifiers for five geographic dialect regions:

* Nile Valley
* Horn of Africa
* Gulf
* North Africa
* Levant

Rather than using a single classifier for all dialects, the project adopts a **region-specific classification approach**, allowing each model to specialize in the linguistic characteristics of its target region.

## Approach

The overall pipeline is:

```text
Arabic Tweets
     │
     ▼
Data Preprocessing
     │
     ▼
Region-Based Dataset Organization
     │
     ├── Nile Valley
     ├── Horn of Africa
     ├── Gulf
     ├── North Africa
     └── Levant
     │
     ▼
MARBERT Tokenization
     │
     ▼
Region-Specific Transformer Fine-Tuning
     │
     ▼
Dialect Classification
     │
     ▼
Accuracy & F1 Evaluation
```

## Model

### MARBERT

The project uses:

**UBC-NLP/MARBERT**

MARBERT is a Transformer-based language model developed specifically for Arabic language understanding. It is well suited for dialect-related NLP tasks because it was trained on large-scale Arabic text containing different linguistic varieties.

The model is **fine-tuned**, rather than used only as a fixed embedding extractor. During fine-tuning, the pretrained Transformer representations are adapted to the dialect classification task.

## Region-Specific Classification

Five separate classification models are fine-tuned, one for each region:

| Region         |   Accuracy |   F1-Score |
| -------------- | ---------: | ---------: |
| Nile Valley    |     99.71% |     99.71% |
| Horn of Africa |    100.00% |    100.00% |
| Gulf           |     95.91% |     95.88% |
| North Africa   |     97.78% |     97.78% |
| Levant         |     99.42% |     99.42% |
| **Mean**       | **98.57%** | **98.56%** |

The results demonstrate strong classification performance across all five regional groups, with the Gulf region presenting the lowest measured accuracy among the evaluated regions.

## NADI 2021 Context

The project is based on the **NADI 2021** Arabic dialect identification task, which focuses on identifying the geographic origin of Arabic tweets.

The original NADI task provides a challenging benchmark because Arabic dialects can share vocabulary and grammatical characteristics while still exhibiting regional differences.

This project explores a region-focused modeling strategy instead of relying exclusively on a single country-level classifier.

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* MARBERT
* NLP
* Deep Learning
* Transformer Architecture
* Arabic Dialect Classification
* Scikit-learn
* Pandas
* NumPy

## Key Concepts

### Transformer Fine-Tuning

The pretrained MARBERT model is adapted to the dialect classification task by adding a classification layer and training the model on labeled dialect data.

```text
Arabic Text
     │
     ▼
MARBERT Tokenizer
     │
     ▼
Transformer Encoder
     │
     ▼
Contextual Representation
     │
     ▼
Classification Layer
     │
     ▼
Dialect Prediction
```

### Why MARBERT?

Arabic contains substantial variation between Modern Standard Arabic and regional dialects. MARBERT provides contextual representations learned from Arabic text and can therefore be fine-tuned to capture linguistic patterns relevant to dialect identification.

### Why Region-Specific Models?

Instead of forcing one classifier to learn all regional differences simultaneously, the project separates the classification problem into regional groups.

This allows each model to focus on the dialect characteristics associated with its corresponding geographic region.

## Evaluation

The models are evaluated using:

### Accuracy

Measures the proportion of correctly classified samples:

```text
Accuracy = Correct Predictions / Total Predictions
```

### F1-Score

The F1-score combines precision and recall:

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Both metrics are reported to provide a more complete evaluation of classification performance.

## Project Structure

```text
NADI-Arabic-Dialect-Classification/
│
├── data/
│   └── NADI2021/
│
├── notebooks/
│   └── dialect_classification.ipynb
│
├── models/
│   ├── nile_valley/
│   ├── horn_of_africa/
│   ├── gulf/
│   ├── north_africa/
│   └── levant/
│
├── results/
│   ├── metrics/
│   └── confusion_matrices/
│
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/khawlaalshaikh/NADI-Arabic-Dialect-Classification.git
cd NADI-Arabic-Dialect-Classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Main Libraries

```text
torch
transformers
datasets
scikit-learn
pandas
numpy
matplotlib
seaborn
```

## Usage

Load the MARBERT tokenizer and model:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "UBC-NLP/MARBERT"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)
```

The input Arabic text is tokenized and passed through the fine-tuned MARBERT classifier to obtain the predicted dialect label.

## Results Summary

The region-specific MARBERT approach achieved a **98.57% mean accuracy** and **98.56% mean F1-score** across the five evaluated regions.

The strongest measured performance was obtained for the **Horn of Africa** model, while the **Gulf** model produced the lowest accuracy among the five regional models.

Overall, the experiment demonstrates the effectiveness of adapting a pretrained Arabic Transformer to regional dialect identification.

## Future Improvements

Potential extensions include:

* Joint hierarchical region → country classification
* Multilingual and cross-dialect evaluation
* Data augmentation for underrepresented dialects
* Additional Arabic dialect datasets
* Hyperparameter optimization
* Cross-region generalization experiments
* Comparison with AraBERT and other Arabic Transformer models
* Deployment as an interactive Arabic dialect identification application

## Author

**Khawla Al-Shaikh**

AI & Data Science | Machine Learning | NLP | Deep Learning
