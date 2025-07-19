import torch

# --- Use the path to the file O provided ---
checkpoint_path = "C:/Mani/learn/Courses/BioCosmos/Butterfly_Project/Artifacts/best_img2img__v1_20250707.pt"

print(f"--- Inspecting Checkpoint File: {checkpoint_path} ---\n")

try:
    # Load the entire file into memory
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 1. Check the overall type of the saved object
    print(f"The saved object is a: {type(checkpoint)}\n")

    # 2. If it's a dictionary (it almost always is), print its top-level keys
    if isinstance(checkpoint, dict):
        print("Found a dictionary with the following top-level keys:")
        for key in checkpoint.keys():
            print(f"- {key}")

        # 3. Check if one of those keys is 'state_dict' (very common)
        if 'state_dict' in checkpoint:
            print("\nThis appears to be a complex checkpoint (e.g., from PyTorch Lightning).")
            print("The actual model weights are nested inside the 'state_dict' key.")
            # Optional: peek at the first 3 layer names inside the state_dict
            model_weights = checkpoint['state_dict']
            print("\nFirst 3 layer names inside 'state_dict':")
            for i, key in enumerate(model_weights.keys()):
                if i >= 3: break
                print(f"- {key}")
        else:
            print("\nThis appears to be a clean state dictionary already.")
            print("The top-level keys are the model's layer names.")

    else:
        # This would be unusual, but it's good to handle
        print("The file does not contain a dictionary. It might be a different type of object.")

except Exception as e:
    print(f"An error occurred while trying to load the file: {e}")