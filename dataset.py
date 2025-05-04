# dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class SkinDiseaseDataset(Dataset):
    """
    Custom PyTorch Dataset for loading skin disease images and conversations.
    Handles image preprocessing and text formatting using the Qwen processor's chat template.
    """
    def __init__(self, dataset_list, image_dir, processor, image_size=(448, 448)):
        self.data = []
        self.processor = processor
        self.image_dir = image_dir
        self.image_size = image_size
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),  # Converts image to [0, 1] range tensor
        ])

        print(f"Initializing dataset. Searching for images in: {self.image_dir}")
        found_count = 0
        missing_count = 0
        load_errors = 0

        for sample in dataset_list:
            image_filename = sample.get("image")
            if not image_filename:
                # print("Warning: Sample missing 'image' key.")
                continue

            image_path = os.path.join(self.image_dir, image_filename)
            human_question = None
            assistant_response = None

            conversations = sample.get("conversations", [])
            if not conversations:
                # print(f"Warning: Sample for image {image_filename} has no 'conversations'.")
                continue

            for conv in conversations:
                sender = conv.get("from", "").lower()
                value = conv.get("value", "")
                if sender == "human":
                    human_question = value.replace("<image>\n", "").strip()
                elif sender in ["gpt", "assistant"]:
                    assistant_response = value.strip()

            if human_question and assistant_response:
                if os.path.exists(image_path):
                    # Basic image check during init (optional, can slow down init)
                    # try:
                    #     Image.open(image_path).verify() # Quick check if image is valid
                    self.data.append({
                        "image_path": image_path,
                        "question": human_question,
                        "answer": assistant_response
                    })
                    found_count += 1
                    # except Exception as e:
                    #     print(f"Warning: Image file {image_path} exists but is potentially corrupt: {e}")
                    #     load_errors += 1
                    #     missing_count += 1
                else:
                    # print(f"Warning: Image file not found: {image_path}")
                    missing_count += 1
            # else:
            #     print(f"Warning: Sample for image {image_filename} missing question or answer.")


        print(f"Dataset initialized. Found {found_count} valid samples.")
        if missing_count > 0:
            print(f"Warning: Could not find {missing_count} image files.")
        if load_errors > 0:
             print(f"Warning: Encountered {load_errors} potential image loading errors during init check.")
        if not self.data:
            print("Warning: No valid data loaded into the dataset!")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
             raise IndexError("Index out of bounds")
        item = self.data[idx]

        # Format prompts using the processor's chat template
        prompts = [
            {"role": "user", "content": [
                {"type": "image", "image": item["image_path"]}, # Pass path for processor
                {"type": "text", "text": item["question"]}
            ]},
            {"role": "assistant", "content": item["answer"]}
        ]
        # Note: apply_chat_template handles image loading if path is given here
        # However, we need the tensor for custom transforms/size, so we load separately
        prompt_text = self.processor.apply_chat_template(prompts, tokenize=False, add_generation_prompt=False)

        # Load and preprocess image manually
        try:
            image = Image.open(item["image_path"]).convert("RGB")
            image_tensor = self.transform(image) # Apply transformations
        except Exception as e:
            print(f"Error loading/transforming image {item['image_path']}: {e}. Using zero tensor.")
            image_tensor = torch.zeros(3, *self.image_size)  # Fallback

        # Tokenize text and combine with *preprocessed* image tensor
        inputs = self.processor(
            text=[prompt_text],
            images=[image_tensor], # Use the transformed tensor
            padding=True,
            return_tensors="pt",
            do_rescale=False # Image is already [0, 1]
        )

        # Ensure image_grid_thw is present if needed by the model
        if 'image_grid_thw' not in inputs and 'pixel_values' in inputs:
            # Get patch size from processor if possible, else use default
            patch_size = getattr(self.processor.image_processor, 'patch_size', 32)
            grid_h = self.image_size[0] // patch_size
            grid_w = self.image_size[1] // patch_size
            inputs['image_grid_thw'] = torch.tensor([[grid_h, grid_w]], dtype=torch.long)

        # Remove the batch dimension before returning
        return {k: v.squeeze(0) for k, v in inputs.items()}