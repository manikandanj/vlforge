import h5py
import pandas as pd


def browse_h5(filename):
    """Quick browser function"""
    with h5py.File(filename, 'r') as f:
        embeddings = f['embeddings'][:]
            
        # Load metadata
        metadata = {}
        for key in f['metadata'].keys():
            values = f['metadata'][key][:]
            metadata[key] = values
        
        # Convert to DataFrame
        metadata_df = pd.DataFrame(metadata)
        
        # Load attributes
        model_name = f['attributes'].attrs['model_name']
        timestamp = f['attributes'].attrs['creation_timestamp']

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(f"Embeddings: {embeddings.shape}")
    print(f"Metadata: ")
    print(metadata_df.head())
    print(f"Model: {model_name}")
    print(f"Created: {timestamp}")


browse_h5('C:/Mani/learn/Courses/BioCosmos/Butterfly_Project/Code/vlforge/storage/2025-07-07_08-03-55_BioCLIP_bioclip_butterfly_embeddings.h5')