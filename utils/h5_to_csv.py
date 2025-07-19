"""
Simple H5 to CSV Converter for PostgreSQL Import

Uses only source data, no synthesis. Keeps taxonomy values fixed as specified.
"""

import h5py
import csv
import math
import uuid

def convert_h5_to_csv_simple(h5_file_path, output_path, max_records=None, batch_size=1000):
    """
    Simple converter that uses only source data without synthesis
    If max_records is None, processes all records
    batch_size: Number of records to process in each batch (default: 1000)
    """
    print(f"Loading data from {h5_file_path}")
    
    with h5py.File(h5_file_path, 'r') as hf:
        # Load embeddings
        embeddings = hf['embeddings'][:]
        
        # Load metadata - decode bytes to strings
        metadata = {}
        for field_name in hf['metadata'].keys():
            field_data = hf['metadata'][field_name][:]
            decoded_values = []
            
            for val in field_data:
                if isinstance(val, bytes):
                    decoded_val = val.decode('utf-8')
                    # Convert 'nan' strings to None
                    if decoded_val.lower() == 'nan' or decoded_val.strip() == '':
                        decoded_values.append(None)
                    else:
                        decoded_values.append(decoded_val)
                elif val is None:
                    decoded_values.append(None)
                else:
                    # Handle NaN values for numeric fields like lat/lon
                    if str(val).lower() == 'nan':
                        decoded_values.append(None)
                    else:
                        decoded_values.append(val)
            
            metadata[field_name] = decoded_values
        
        # Get model info
        model_name = hf['attributes'].attrs['model_name'].decode('utf-8') if isinstance(hf['attributes'].attrs['model_name'], bytes) else hf['attributes'].attrs['model_name']
    
    # Determine how many records to process
    total_records = len(embeddings)
    records_to_process = max_records if max_records is not None else total_records
    records_to_process = min(records_to_process, total_records)
    
    # Calculate number of batches
    num_batches = math.ceil(records_to_process / batch_size)
    
    print(f"Processing {records_to_process} out of {total_records} total records...")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {num_batches}")
    
    # Create CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        # CSV headers matching PostgreSQL table
        headers = [
            'specimen_uuid', 'media_uuid', 'collection_date', 'higher_taxon',
            'common_name', 'scientific_name', 'recorded_by', 'location',
            'tax_kingdom', 'tax_phylum', 'tax_class', 'tax_order', 'tax_family',
            'tax_genus', 'catalog_number', 'earliest_epoch_or_lowest_series',
            'earliest_age_or_lowest_stage', 'external_media_uri', 'embedding',
            'model', 'pretrained', 'embed_version'
        ]
        
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        # Process records in batches
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, records_to_process)
            
            # Process current batch
            for i in range(start_idx, end_idx):
                # Get source data
                species = metadata.get('species', [None] * total_records)[i]
                url = metadata.get('url', [None] * total_records)[i]
                mask_name = metadata.get('mask_name', [None] * total_records)[i]
                lat = metadata.get('lat', [None] * total_records)[i]
                lon = metadata.get('lon', [None] * total_records)[i]
                
                # Generate unique UUID for media_uuid
                media_uuid = str(uuid.uuid4())
                
                # Format scientific name from species (e.g., "apatura_iris" -> "Apatura iris")
                scientific_name = None
                tax_genus = None
                if species:
                    parts = species.split('_')
                    if len(parts) >= 2:
                        genus = parts[0].capitalize()
                        species_part = parts[1].lower()
                        scientific_name = f"{genus} {species_part}"
                        tax_genus = genus
                    else:
                        scientific_name = species
                
                # Create location from lat/lon if both are available
                location = None
                if lat is not None and lon is not None:
                    try:
                        # Convert to float to ensure they're valid numbers
                        lat_val = float(lat)
                        lon_val = float(lon)
                        # PostgreSQL POINT format: POINT(longitude latitude)
                        location = f"POINT({lon_val} {lat_val})"
                    except (ValueError, TypeError):
                        # If conversion fails, leave location as None
                        location = None
                
                # Convert embedding to PostgreSQL vector format
                embedding_vector = embeddings[i].tolist()
                embedding_str = '[' + ','.join(map(str, embedding_vector)) + ']'
                
                # Create row with explicit None values for PostgreSQL
                row = [
                    None,                         # specimen_uuid (NULL)
                    media_uuid,                   # media_uuid (generated UUID)
                    None,                         # collection_date (NULL)
                    None,                         # higher_taxon (NULL)
                    None,                         # common_name (NULL)
                    scientific_name,              # scientific_name (from species)
                    None,                         # recorded_by (NULL)
                    location,                     # location (from lat/lon)
                    'Animalia',                   # tax_kingdom (fixed)
                    'Arthropoda',                 # tax_phylum (fixed)
                    'Insecta',                    # tax_class (fixed)
                    'Lepidoptera',                # tax_order (fixed)
                    'Nymphalidae',                # tax_family (fixed)
                    tax_genus,                    # tax_genus (from species)
                    mask_name,                    # catalog_number (from mask_name)
                    None,                         # earliest_epoch_or_lowest_series (NULL)
                    None,                         # earliest_age_or_lowest_stage (NULL)
                    url,                          # external_media_uri (from url)
                    embedding_str,                # embedding (converted to vector format)
                    model_name,                   # model (from H5 attributes)
                    'bioclip',                    # pretrained (fixed)
                    'v1'                          # embed_version (fixed)
                ]
                
                writer.writerow(row)
            
            # Print progress after each batch
            records_processed = min(end_idx, records_to_process)
            print(f"Batch {batch_num + 1} / {num_batches} completed ({records_processed} records processed)")
    
    print(f"Simple CSV generated: {output_path}")
    print(f"Records processed: {records_to_process}")

if __name__ == "__main__":
    h5_file_path = 'C:/Mani/learn/Courses/BioCosmos/Butterfly_Project/Artifacts/2025-07-09_08-03-13_FineTunedBioCLIP_bioclip_butterfly_embeddings_v1.h5'
    output_path = 'butterfly_embeddings.csv'
    
    # Process all records with configurable batch size
    # You can adjust batch_size (default: 1000) based on your memory constraints
    convert_h5_to_csv_simple(h5_file_path, output_path, max_records=None, batch_size=1000)