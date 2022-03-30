# csv-to-embeddings-model
Trains a model on top of a sbert's pertained models with given trained pairs to be used with Python's Sentence Transformer

## Getting Started

To start, run:
```bash
pip install -r requirements.txt
```

## Creating Training Pairs

Make sure to create a `pairs.csv` file with the trained pairs you want to train the model with.
- Example can be found at `example_pairs.csv`

## Generating a Trained Model

After you have installed the requirements and created a `pairs.csv` file, run:

```bash
python model.py
```

A new, trained model will be stored under `trained_model/`.

## Using sbert Pretrained Models

You may use other pretrained models from here: [https://www.sbert.net/docs/pretrained_models.html](https://www.sbert.net/docs/pretrained_models.html)

The default model used is `multi-qa-MiniLM-L6-cos-v1`.
