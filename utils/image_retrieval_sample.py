import torch
import torch.nn.functional as F
import open_clip
import h5py
from PIL import Image

# --- 1. Load the fine-tuned model ---
# This only needs to be done once when the application starts.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, processor = open_clip.create_model_and_transforms(
    "hf-hub:imageomics/bioclip",
    pretrained=None
)
finetuned_weights = torch.load("C:/Mani/learn/Courses/BioCosmos/Butterfly_Project/Artifacts/best_img2img__v1_20250707.pt", map_location=device)
model.load_state_dict(finetuned_weights)
model.to(device)
model.eval()

# --- 2. Load the database embeddings and metadata ---
# This should also be loaded once at startup.
with h5py.File('C:/Mani/learn/Courses/BioCosmos/Butterfly_Project/Artifacts/2025-07-09_08-03-13_FineTunedBioCLIP_bioclip_butterfly_embeddings_v1.h5', 'r') as hf:
    db_embeddings = torch.from_numpy(hf['embeddings'][:]).to(device)
    db_species = [s.decode('utf-8') for s in hf['metadata']['species'][:]]
    db_image_paths = [p.decode('utf-8') for p in hf['metadata']['url'][:]]  
    db_mask_names = [m.decode('utf-8') for m in hf['metadata']['mask_name'][:]] 

print(f"Database embeddings shape: {db_embeddings.shape}")
print(f"Embedding dimension: {db_embeddings.shape[1]}")
print(f"Number of images in database: {db_embeddings.shape[0]}")


# --- 3. Perform a search for a new query image ---
def find_similar_images(query_image_path, top_k=10):
    with torch.no_grad():
        # Process the query image and get its embedding
        query_image = processor(Image.open(query_image_path)).unsqueeze(0).to(device)
        query_embedding = model.encode_image(query_image)
        query_embedding = F.normalize(query_embedding, dim=-1)

        # Compute cosine similarity against the entire database
        similarities = query_embedding @ db_embeddings.T

        # Get the top k results
        top_results = torch.topk(similarities, k=top_k, dim=-1)
        scores = top_results.values.squeeze(0).tolist()
        indices = top_results.indices.squeeze(0).tolist()

        # Look up the metadata for the results
        results_metadata = []
        for i, score in zip(indices, scores):
            results_metadata.append({
                "rank": len(results_metadata) + 1,
                "score": score,
                "species": db_species[i],
                "url": db_image_paths[i], 
                "mask_name": db_mask_names[i]
            })

        return results_metadata

# Example Usage
# results = find_similar_images("C:/Mani/learn/Courses/BioCosmos/Butterfly_Project/Test/3411259545_136f911881_b.jpg", top_k=15)
results = find_similar_images("C:/Mani/learn/Courses/BioCosmos/Butterfly_Project/Test/purple_butterfly.jpg", top_k=15)
print(results)
