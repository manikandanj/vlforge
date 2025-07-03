import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from utils.common import set_seed

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")


# ========== EVALUATION METRICS FUNCTIONS ==========

def precision_at_k(query_species: str, ranked_species: list, k: int, verbose: bool = False) -> float:
    """
    Calculate Precision@k using binary relevance (same species = relevant).
    
    Args:
        query_species: True species of the query image
        ranked_species: List of species in ranked order of similarity
        k: Number of top results to consider
        verbose: Whether to print detailed calculation steps
    
    Returns:
        Precision@k score (0.0 to 1.0)
    """
    if k <= 0 or len(ranked_species) == 0:
        return 0.0
    
    top_k_species = ranked_species[:k]
    relevant_count = sum(1 for species in top_k_species if species == query_species)
    precision = relevant_count / k
    
    if verbose:
        print(f"  Precision@{k}: {relevant_count}/{k} = {precision:.3f}")
        print(f"    Top-{k} species: {top_k_species}")
        print(f"    Relevant in top-{k}: {relevant_count}")
    
    return precision


def recall_at_k(query_species: str, ranked_species: list, all_species_in_db: list, k: int, verbose: bool = False) -> float:
    """
    Calculate Recall@k using binary relevance (same species = relevant).
    
    Args:
        query_species: True species of the query image
        ranked_species: List of species in ranked order of similarity
        all_species_in_db: All species present in the database
        k: Number of top results to consider
        verbose: Whether to print detailed calculation steps
    
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if k <= 0 or len(ranked_species) == 0:
        return 0.0
    
    # Total relevant items in database (all images of same species)
    total_relevant = sum(1 for species in all_species_in_db if species == query_species)
    
    if total_relevant == 0:
        return 0.0  # No relevant items in database
    
    top_k_species = ranked_species[:k]
    relevant_retrieved = sum(1 for species in top_k_species if species == query_species)
    recall = relevant_retrieved / total_relevant
    
    if verbose:
        print(f"  Recall@{k}: {relevant_retrieved}/{total_relevant} = {recall:.3f}")
    
    return recall


def mean_average_precision(query_species: str, ranked_species: list, verbose: bool = False) -> float:
    """
    Calculate mean Average Precision (mAP) using binary relevance.
    
    Args:
        query_species: True species of the query image
        ranked_species: List of species in ranked order of similarity
        verbose: Whether to print detailed calculation steps
    
    Returns:
        mAP score (0.0 to 1.0)
    """
    if len(ranked_species) == 0:
        return 0.0
    
    relevant_count = 0
    precision_sum = 0.0
    precision_values = []
    total_relevant = sum(1 for species in ranked_species if species == query_species)
    
    if total_relevant == 0:
        if verbose:
            print("  No relevant items found, mAP = 0.000")
        return 0.0  # No relevant items
    
    for i, species in enumerate(ranked_species):
        if species == query_species:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i
            precision_values.append(precision_at_i)
            if verbose:
                print(f"  Relevant item found at rank {i+1}: Precision = {relevant_count}/{i+1} = {precision_at_i:.3f}")
    
    map_score = precision_sum / total_relevant
    
    if verbose:
        print(f"  mAP = ({' + '.join([f'{p:.3f}' for p in precision_values])}) / {total_relevant} = {map_score:.3f}")
    
    return map_score


def get_relevance_score(query_species: str, result_species: str, 
                       query_taxonomy: dict, result_taxonomy: dict) -> int:
    """
    Calculate relevance score based on taxonomic hierarchy (from ndcg_evaluator):
    - Species match: 3 (Highly Relevant)
    - Genus match: 2 (Relevant) 
    - Subfamily match: 1 (Partially Relevant)
    - No match: 0 (Irrelevant)
    """
    if query_species == result_species:
        return 3
        
    query_genus = query_taxonomy.get('genus', '')
    result_genus = result_taxonomy.get('genus', '')
    query_subfamily = query_taxonomy.get('subfamily', '')
    result_subfamily = result_taxonomy.get('subfamily', '')
    
    # Check genus match
    if query_genus and result_genus and query_genus == result_genus:
        return 2
        
    # Check subfamily match
    if query_subfamily and result_subfamily and query_subfamily == result_subfamily:
        return 1
        
    return 0


def ndcg_at_k(query_species: str, ranked_results: list, query_taxonomy: dict, k: int, verbose: bool = False) -> float:
    """
    Calculate nDCG@k using hierarchical relevance scoring.
    
    Args:
        query_species: True species of the query image
        ranked_results: List of (species, similarity_score, taxonomy_dict) tuples in ranked order
        query_taxonomy: Taxonomy information for query species
        k: Number of top results to consider
        verbose: Whether to print detailed calculation steps
    
    Returns:
        nDCG@k score (0.0 to 1.0)
    """
    if k <= 0 or len(ranked_results) == 0:
        return 0.0
    
    # Get relevance scores for top-k results
    relevance_scores = []
    prediction_scores = []
    
    if verbose:
        print(f"nDCG@{k} CALCULATION:")
        print("(Hierarchical relevance: Species=3, Genus=2, Subfamily=1, None=0)")
    
    for i, (species, similarity_score, taxonomy) in enumerate(ranked_results[:k]):
        relevance = get_relevance_score(query_species, species, query_taxonomy, taxonomy)
        relevance_scores.append(relevance)
        prediction_scores.append(similarity_score)
        if verbose:
            print(f"  Rank {i+1}: {species} -> Relevance = {relevance}")
    
    # Calculate DCG
    dcg = 0.0
    for i, relevance in enumerate(relevance_scores):
        dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) is 0
    
    # Calculate ideal DCG (best possible ordering)
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = 0.0
    for i, relevance in enumerate(ideal_relevance):
        idcg += relevance / np.log2(i + 2)
    
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    if verbose:
        print(f"  DCG = {' + '.join([f'{rel}/log2({i+2})' for i, rel in enumerate(relevance_scores)])} = {dcg:.3f}")
        print(f"  IDCG = {' + '.join([f'{rel}/log2({i+2})' for i, rel in enumerate(ideal_relevance)])} = {idcg:.3f}")
        print(f"  nDCG@{k} = DCG/IDCG = {dcg:.3f}/{idcg:.3f} = {ndcg:.3f}")
    
    return ndcg


def calculate_all_metrics(query_info: dict, results_df: pd.DataFrame, database_species: list, k_values: list = [1, 5, 15], verbose: bool = False) -> dict:
    """
    Calculate all evaluation metrics for a single query.
    
    Args:
        query_info: Dictionary with query image information
        results_df: DataFrame with ranked results
        database_species: List of all species in the database
        k_values: List of k values to evaluate
        verbose: Whether to print detailed calculation steps
    
    Returns:
        Dictionary with all calculated metrics
    """
    query_species = query_info['species']
    query_taxonomy = {
        'genus': query_info['genus'],
        'subfamily': query_info['subfamily']
    }
    
    # Prepare data for metrics
    ranked_species = results_df['species'].tolist()
    ranked_results = []
    for _, row in results_df.iterrows():
        taxonomy = {
            'genus': row['genus'],
            'subfamily': row['subfamily']
        }
        ranked_results.append((row['species'], row['similarity_score'], taxonomy))
    
    metrics = {}
    
    if verbose:
        print("PRECISION@K CALCULATIONS:")
        print("(Relevant items in top-k / k)")
    
    # Calculate metrics for each k value
    for k in k_values:
        metrics[f'precision@{k}'] = precision_at_k(query_species, ranked_species, k, verbose)
        
    if verbose:
        print("\nRECALL@K CALCULATIONS:")
        print("(Relevant items in top-k / total relevant in database)")
        
    for k in k_values:
        metrics[f'recall@{k}'] = recall_at_k(query_species, ranked_species, database_species, k, verbose)
    
    if verbose:
        print("\nmAP CALCULATION:")
        print("(Average precision at each relevant item)")
    
    # Calculate mAP (using all results)
    metrics['mAP'] = mean_average_precision(query_species, ranked_species, verbose)
    
    if verbose:
        print()
        
    # Calculate nDCG for each k value
    for k in k_values:
        if verbose:
            print()
        metrics[f'ndcg@{k}'] = ndcg_at_k(query_species, ranked_results, query_taxonomy, k, verbose)
    
    return metrics


def create_demo_visualization(query_info, results_df, data_loader, metrics, save_path="demo_similarity_results.png"):
    """
    Create a captivating visualization showing query image and top similar images
    """
    # Set up the figure with a clean layout
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('white')
    
    # Create custom grid layout: smaller query image on left, 5x3 grid on right
    gs = GridSpec(4, 6, figure=fig, height_ratios=[0.8, 0.8, 0.8, 0.6], width_ratios=[1, 1, 1, 1, 1, 1], 
                  hspace=0.3, wspace=0.2)
    
    # Title
    fig.suptitle('Butterfly Image Similarity Search Results', 
                fontsize=22, fontweight='bold', y=0.95, color='#2c3e50')
    
    # Query image section (smaller, upper-left)
    ax_query = fig.add_subplot(gs[0:2, 0])
    query_image = data_loader.load_image(f"{query_info['species']}/{query_info['unique_id']}")
    if query_image:
        ax_query.imshow(query_image)
        ax_query.set_title(f"QUERY IMAGE\n{query_info['unique_id']}", 
                          fontsize=12, fontweight='bold', color='#e74c3c', pad=15)
        ax_query.text(0.5, -0.05, f"Species: {query_info['species']}\nGenus: {query_info['genus']}\nSubfamily: {query_info['subfamily']}", 
                     transform=ax_query.transAxes, ha='center', va='top',
                     fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='#ecf0f1', alpha=0.8))
    ax_query.axis('off')
    
    # Top similar images in 5x3 grid
    n_show = min(15, len(results_df))
    grid_rows = 3
    grid_cols = 5
    
    for i in range(n_show):
        row = i // grid_cols
        col = i % grid_cols
        
        ax = fig.add_subplot(gs[row, col + 1])
        
        result = results_df.iloc[i]
        similar_image = data_loader.load_image(f"{result['species']}/{result['unique_id']}")
        
        if similar_image:
            ax.imshow(similar_image)
            
            # Color-coded border based on taxonomic match
            if result['same_species']:
                border_color = '#27ae60'  # Green for species match
                match_text = "Species Match"
            elif result['same_genus']:
                border_color = '#f39c12'  # Orange for genus match
                match_text = "Genus Match"
            elif result['same_subfamily']:
                border_color = '#3498db'  # Blue for subfamily match
                match_text = "Subfamily Match"
            else:
                border_color = '#e74c3c'  # Red for no match
                match_text = "No Match"
            
            # Add colored border
            for spine in ax.spines.values():
                spine.set_color(border_color)
                spine.set_linewidth(3)
            
            # Title with rank and similarity - more compact
            ax.set_title(f"#{result['rank']} | Sim: {result['similarity_score']:.3f}", 
                        fontsize=10, fontweight='bold', color=border_color, pad=8)
            
            # Add match status as text overlay on image (top-left corner)
            ax.text(0.05, 0.95, match_text, transform=ax.transAxes, 
                   fontsize=8, fontweight='bold', color='white',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=border_color, alpha=0.8),
                   verticalalignment='top', horizontalalignment='left')
        
        ax.axis('off')
    
    # Add summary statistics box (bottom area)
    summary_text = f"""SEARCH SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Query: {query_info['species']}
Database Size: {len(results_df)} images searched

PRECISION@K:           RECALL@K:             nDCG@K:
@1: {metrics['precision@1']:.3f}             @1: {metrics['recall@1']:.3f}            @1: {metrics['ndcg@1']:.3f}
@5: {metrics['precision@5']:.3f}             @5: {metrics['recall@5']:.3f}            @5: {metrics['ndcg@5']:.3f}
@15: {metrics['precision@15']:.3f}            @15: {metrics['recall@15']:.3f}           @15: {metrics['ndcg@15']:.3f}

mAP: {metrics['mAP']:.3f}

TAXONOMY MATCHES:
Species: {sum(results_df['same_species'])} / {len(results_df)}
Genus: {sum(results_df['same_genus'])} / {len(results_df)}
Subfamily: {sum(results_df['same_subfamily'])} / {len(results_df)}
    """
    
    fig.text(0.02, 0.02, summary_text, fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#ecf0f1', alpha=0.9),
             verticalalignment='bottom')
    
    plt.subplots_adjust(top=0.90, bottom=0.30, left=0.02, right=0.98)
    
    # Save the visualization
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Demo visualization saved to: {save_path}")
    
    plt.show()
    
    return fig


@hydra.main(config_path="conf", config_name="experiment", version_base="1.1")
def main(cfg: DictConfig) -> None:

    set_seed(cfg.experiment.seed)
    print(f"Starting {cfg.experiment.name}")
    print(f"Mode: {cfg.mode}")
    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = instantiate(cfg.model)
    print(f"Model info: {model.get_model_info()}")

    data_loader = instantiate(cfg.data)
    print(f"Dataset info: {data_loader.get_dataset_info()}")
    
    # Collect all data for dataframe creation
    all_embeddings = []
    all_metadata = []
    all_unique_ids = []
    
    print("Collecting embeddings and metadata...")
    
    # Process batches and collect data
    batch_count = 0
    for images, labels, unique_ids in data_loader.get_batch_data(n_samples=100):  # Adjust n_samples as needed
        print(f"Processing batch {batch_count + 1} with {len(images)} images")
        
        # Get embeddings for this batch
        embeddings = model.get_image_embeddings(images)
        print(f"Embedding shape for batch: {embeddings.shape}")
        
        # Get metadata for this batch
        metadata = data_loader.get_metadata_for_ids(unique_ids)
        
        # Store data (convert tensor to numpy and move to CPU if needed)
        if torch.is_tensor(embeddings):
            embeddings_np = embeddings.cpu().numpy()
        else:
            embeddings_np = embeddings
        all_embeddings.append(embeddings_np)
        all_metadata.extend(metadata)
        all_unique_ids.extend(unique_ids)
        
        batch_count += 1
    
    # Combine all embeddings
    if all_embeddings:
        combined_embeddings = np.vstack(all_embeddings)
        print(f"Final combined embeddings shape: {combined_embeddings.shape}")
        
        # Create dataframe
        print("Creating comprehensive dataframe...")
        
        # Start with metadata
        df = pd.DataFrame(all_metadata)
        
        # Add unique_ids as a separate column for clarity
        df['unique_id'] = all_unique_ids
        
        # Add embedding dimensions as separate columns
        embedding_dim = combined_embeddings.shape[1]
        for i in range(embedding_dim):
            df[f'embedding_dim_{i}'] = combined_embeddings[:, i]
        
        print(f"Dataframe created with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        
        print(f"\nDataframe info:")
        print(df.info())
        
        print(f"\nSample metadata fields:")
        metadata_cols = [col for col in df.columns if not col.startswith('embedding_dim_')]
        print(df[metadata_cols].head())
        
        # Save dataframe for later use
        output_path = "embeddings_metadata_df.csv"
        df.to_csv(output_path, index=False)
        print(f"\nDataframe saved to: {output_path}")
        
        # Store the dataframe globally for interactive use
        globals()['embeddings_df'] = df
        
        # ========== QUERY-DATABASE SPLIT AND SIMILARITY SEARCH ==========
        print("\n" + "="*60)
        print("PERFORMING QUERY-DATABASE SPLIT AND SIMILARITY SEARCH")
        print("="*60)
        
        # Split into query and database sets (80% database, 20% query)
        np.random.seed(cfg.experiment.seed)  # For reproducible splits
        n_total = len(df)
        query_size = max(1, n_total // 5)  # 20% for query, at least 1
        
        # Random split
        indices = np.random.permutation(n_total)
        query_indices = indices[:query_size]
        database_indices = indices[query_size:]
        
        query_df = df.iloc[query_indices].copy()
        database_df = df.iloc[database_indices].copy()
        
        print(f"Query set size: {len(query_df)}")
        print(f"Database set size: {len(database_df)}")
        
        # Extract embeddings for similarity computation
        embedding_cols = [col for col in df.columns if col.startswith('embedding_dim_')]
        query_embeddings = query_df[embedding_cols].values
        database_embeddings = database_df[embedding_cols].values
        
        # Take the first query image
        query_idx = 0
        query_image_info = query_df.iloc[query_idx]
        query_embedding = query_embeddings[query_idx:query_idx+1]  # Keep 2D shape
        
        print(f"\nQuery Image:")
        print(f"  Unique ID: {query_image_info['unique_id']}")
        print(f"  Species: {query_image_info['species']}")
        print(f"  Genus: {query_image_info['genus']}")
        print(f"  Subfamily: {query_image_info['subfamily']}")
        
        # Calculate cosine similarity between query and all database images
        similarities = cosine_similarity(query_embedding, database_embeddings)[0]
        
        # Get top 15 most similar images
        top_k = 15
        top_indices = np.argsort(similarities)[::-1][:top_k]  # Sort descending, take top k
        
        print(f"\nTop {top_k} Most Similar Images:")
        print("-" * 80)
        
        results_data = []
        for rank, db_idx in enumerate(top_indices, 1):
            similar_image = database_df.iloc[db_idx]
            similarity_score = similarities[db_idx]
            
            # Check if it's the same species (for evaluation)
            same_species = similar_image['species'] == query_image_info['species']
            same_genus = similar_image['genus'] == query_image_info['genus']
            same_subfamily = similar_image['subfamily'] == query_image_info['subfamily']
            
            print(f"Rank {rank}: {similar_image['unique_id']}")
            print(f"  Similarity: {similarity_score:.4f}")
            print(f"  Species: {similar_image['species']} {'✓' if same_species else '✗'}")
            print(f"  Genus: {similar_image['genus']} {'✓' if same_genus else '✗'}")
            print(f"  Subfamily: {similar_image['subfamily']} {'✓' if same_subfamily else '✗'}")
            print()
            
            # Store for analysis
            results_data.append({
                'rank': rank,
                'unique_id': similar_image['unique_id'],
                'similarity_score': similarity_score,
                'species': similar_image['species'],
                'genus': similar_image['genus'],
                'subfamily': similar_image['subfamily'],
                'same_species': same_species,
                'same_genus': same_genus,
                'same_subfamily': same_subfamily
            })
        
        # Create results dataframe
        results_df = pd.DataFrame(results_data)
        

        
        # Save results
        results_output_path = "similarity_search_results.csv"
        results_df.to_csv(results_output_path, index=False)
        print(f"Results saved to: {results_output_path}")
        
        # Store globally for further analysis
        globals()['query_df'] = query_df
        globals()['database_df'] = database_df
        globals()['results_df'] = results_df
        
        # ========== CREATE STUNNING DEMO VISUALIZATION ==========
        print("\n" + "="*60)
        print("CREATING DEMO VISUALIZATION")
        print("="*60)
        
        try:
            # Calculate evaluation metrics
            print("Calculating evaluation metrics...")
            database_species = database_df['species'].tolist()
            
            # Add detailed logging to show how metrics are calculated
            query_species = query_image_info['species']
            ranked_species = results_df['species'].tolist()
            
            print(f"\nDETAILED METRIC CALCULATIONS:")
            print("=" * 80)
            print(f"Query Species: {query_species}")
            print(f"Total Database Images: {len(database_species)}")
            print(f"Database species of same type: {database_species.count(query_species)}")
            print(f"Retrieved Results: {len(ranked_species)}")
            print()
            
            print("RANKED RESULTS:")
            for i, (_, row) in enumerate(results_df.iterrows()):
                match_type = "SPECIES" if row['species'] == query_species else (
                    "GENUS" if row['genus'] == query_image_info['genus'] else (
                        "SUBFAMILY" if row['subfamily'] == query_image_info['subfamily'] else "NO MATCH"
                    )
                )
                print(f"  Rank {i+1}: {row['species']} (Sim: {row['similarity_score']:.3f}) [{match_type}]")
            print()
            
            # Calculate all metrics with verbose logging
            metrics = calculate_all_metrics(query_image_info.to_dict(), results_df, database_species, verbose=True)
            
            # Print final metrics summary
            print("\nFINAL EVALUATION METRICS:")
            print("-" * 50)
            print(f"Precision@1: {metrics['precision@1']:.3f}")
            print(f"Precision@5: {metrics['precision@5']:.3f}")
            print(f"Precision@15: {metrics['precision@15']:.3f}")
            print()
            print(f"Recall@1: {metrics['recall@1']:.3f}")
            print(f"Recall@5: {metrics['recall@5']:.3f}")
            print(f"Recall@15: {metrics['recall@15']:.3f}")
            print()
            print(f"nDCG@1: {metrics['ndcg@1']:.3f}")
            print(f"nDCG@5: {metrics['ndcg@5']:.3f}")
            print(f"nDCG@15: {metrics['ndcg@15']:.3f}")
            print()
            print(f"mAP: {metrics['mAP']:.3f}")
            
            # Create the demo visualization
            create_demo_visualization(query_image_info, results_df, data_loader, metrics)
            print("Demo visualization completed successfully!")
        except Exception as e:
            print(f"Error creating visualization: {e}")
            print("Continuing without visualization...")
        
    else:
        print("No data collected!")

    print(f"Completed {cfg.experiment.name}")


if __name__ == "__main__":
    main() 