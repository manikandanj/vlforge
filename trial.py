import os
import pandas as pd

species_sublst = os.listdir("C:\\Mani\\learn\\Courses\\BioCosmos\\Butterfly_Project\\Data\\images")
print(species_sublst)


meta_data_dir = "C:\\Mani\\learn\\Courses\\BioCosmos\\Butterfly_Project\\Data\\metadata"
meta_data_file = "data_meta-nymphalidae_whole_specimen-v240606"
meta_data_file_ext = ".csv"
meta_data_path = os.path.join(meta_data_dir, meta_data_file + meta_data_file_ext)
df_meta = pd.read_csv(meta_data_path)
df_meta_selected = df_meta[df_meta["species"].isin(species_sublst)]
print(df_meta_selected.head(50)["species"])

meta_data_output_path = os.path.join(meta_data_dir, meta_data_file + "_subset" + meta_data_file_ext)
df_meta_selected.to_csv(meta_data_output_path, index=False)






