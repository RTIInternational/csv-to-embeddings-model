import csv
import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Example Models can be found here: https://www.sbert.net/docs/pretrained_models.html
# Model Card: https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

def main():
	# Format csv file to prepare for model training
	# Example csv file: https://github.com/dsteedRTI/csv-to-embeddings-model/blob/main/example_pairs.csv
	with open("pairs.csv") as f:
	    pairs = [{k: str(v) for k, v in row.items()}
	        for row in csv.DictReader(f, skipinitialspace=True)]

	for pair in pairs:
		if isinstance(pair["label"], str):
			pair["label"] = float(pair["label"])

	with open("pairs.json", "w") as fp:
	    json.dump(pairs, fp)

    #Define your train examples.
	f = open("pairs.json")
	data = json.load(f)
	train_pairs = []
	for pair in data:
		train_pairs.append(InputExample(texts=[pair["text1"], pair["text2"]], label=pair["label"]))

	#Define your train dataset, the dataloader and the train loss
	train_dataloader = DataLoader(train_pairs, shuffle=True, batch_size=16)
	train_loss = losses.CosineSimilarityLoss(model)

	#Tune the model
	model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

	model.save("test_embeddings_model")

if __name__ == "__main__":
    main()
