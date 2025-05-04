# qwen_vl_finetune.py (Main Entry Point)

import argparse
import json
import os
import torch

# Import from our modules
from model_utils import load_model_and_processor, setup_lora, freeze_vision_encoder
from trainer import FineTuneTrainer
from utils import plot_training_metrics # Keep plotting separate from trainer logic

# Set PyTorch CUDA memory management (optional)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main(args):
    """Main function to orchestrate the fine-tuning process."""

    # --- Argument Validation and Setup ---
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset JSON file not found at: {args.dataset_path}")
    if not os.path.isdir(args.image_dir):
        raise NotADirectoryError(f"Image directory not found at: {args.image_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available, training on CPU. This will be very slow.")
        args.use_amp = False # AMP only works on CUDA
        args.bf16 = False  # BF16 only works on CUDA

    # --- Load Dataset List ---
    print(f"Loading dataset metadata from: {args.dataset_path}")
    with open(args.dataset_path, "r") as f:
        dataset_list = json.load(f)
    print(f"Loaded {len(dataset_list)} samples configuration.")

    # --- Load Model and Processor ---
    model, processor = load_model_and_processor(
        args.model_name_or_path,
        device=device, # Pass determined device
        use_quantization=args.use_4bit, # Use quantization argument
        bf16=args.bf16
    )

    # --- Setup LoRA ---
    # Define target modules (adjust if needed for specific model variants)
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    model = setup_lora(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules
    )

    # --- Freeze Layers ---
    if args.freeze_vision_encoder:
        model = freeze_vision_encoder(model)

    # --- Configuration Object ---
    # Use argparse namespace directly or create a simple config class/dict
    config_dict = vars(args) # Convert argparse args to a dictionary
    class Config:
         def __init__(self, **kwargs):
              self.__dict__.update(kwargs)
              self.device = device # Ensure the determined device is in config
              self.use_amp = args.use_amp and device.startswith("cuda") # Final check
              self.bf16 = args.bf16 and device.startswith("cuda") and torch.cuda.is_bf16_supported()

    config = Config(**config_dict)

    # --- Initialize Trainer ---
    print("Initializing Trainer...")
    trainer = FineTuneTrainer(
        model=model,
        processor=processor,
        config=config,
        train_dataset_list=dataset_list, # Pass the list, dataset object created inside trainer
        image_dir=args.image_dir
    )

    # --- Start Training ---
    print("Starting Training...")
    try:
        trainer.train()
        print("Training finished successfully.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        print("Attempting to save final state despite error...")
        # Try saving even if training failed mid-way
        error_save_path = os.path.join(args.output_dir, "error_save_model")
        try:
            os.makedirs(error_save_path, exist_ok=True)
            model.save_pretrained(error_save_path)
            processor.save_pretrained(error_save_path)
            print(f"Model state saved to {error_save_path}")
        except Exception as save_e:
            print(f"Could not save model state after error: {save_e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL model using LoRA.")

    # Paths
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Hugging Face model identifier or local path.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the formatted JSON dataset file.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing images referenced in the dataset.")
    parser.add_argument("--output_dir", type=str, default="./qwen_vl_finetuned_lora", help="Directory to save checkpoints and final LoRA adapters.")

    # Training Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Peak learning rate for the optimizer.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of linear warmup steps for the learning rate scheduler.")
    parser.add_argument("--max_steps", type=int, default=-1, help="If set to a positive number, overrides num_train_epochs. Total number of training steps to perform.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log training metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=250, help="Save a checkpoint every N steps.")

    # LoRA Hyperparameters
    parser.add_argument("--lora_r", type=int, default=8, help="Rank of the LoRA update matrices.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA scaling factor (alpha).")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers.")

    # Hardware & Precision
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device for training ('cuda:0', 'cpu').")
    parser.add_argument("--use_4bit", action='store_true', default=True, help="Enable 4-bit quantization (QLoRA style).") # Default True based on notebook
    parser.add_argument("--no_4bit", dest='use_4bit', action='store_false', help="Disable 4-bit quantization.")
    parser.add_argument("--bf16", action='store_true', help="Use bfloat16 precision if available (requires Ampere+ GPU).")
    parser.add_argument("--use_amp", action='store_true', default=True, help="Use Automatic Mixed Precision (FP16/BF16 based on --bf16 flag and hardware).") # Default True
    parser.add_argument("--no_amp", dest='use_amp', action='store_false', help="Disable Automatic Mixed Precision.")


    # Freezing
    parser.add_argument("--freeze_vision_encoder", action='store_true', default=True, help="Freeze the vision encoder weights during training.")
    parser.add_argument("--no-freeze_vision_encoder", dest='freeze_vision_encoder', action='store_false', help="Do not freeze the vision encoder weights.")

    args = parser.parse_args()

    # Call main function
    main(args)