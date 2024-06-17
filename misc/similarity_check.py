from transformers import BertTokenizer, BertModel
import torch

def main():
    # Load pre-trained model tokenizer (vocabulary) and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Define sentences
    texts = ["To advance the implementation of the human rights to safe drinking water and sanitation, private water operators federated in AquaFed commit to supporting governments of SWA 'priority countries' and others to establish new and/or engage/work through existing national, local and regional multi-stakeholder platforms.",
             "Support the African voice in the convening of regional and global forum on WASH, health and climate change and showcasing the best practices through initiatives taking place in Kenya, Uganda, Tanzania and Malawi."]

    # Encode text
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Use mean pooling to convert contextual embeddings into a single sentence vector
    embeddings = model_output.last_hidden_state.mean(dim=1)
    
    # Compute cosine similarity between vectors
    cos = torch.nn.CosineSimilarity(dim=1)
    similarity = cos(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    
    print("Similarity score:", similarity.item())

if __name__ == "__main__":
    main()
