# Qwen2.5-VL Fine-Tuning for Skin Disease Image Analysis

This repository demonstrates the fine-tuning of the **Qwen2.5-VL-7B-Instruct** Vision-Language Model (VLM) for a specialized medical task: analyzing skin disease images and generating descriptive text or potential classifications based on user questions.

The fine-tuning process utilizes Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA) and 4-bit quantization (QLoRA-style) for efficient training, making it feasible to adapt large VLMs on consumer-grade or cloud-based GPUs.

## Features & Skills Demonstrated

This project showcases proficiency in the following areas of VLM fine-tuning:

*   **Vision-Language Model Fine-Tuning:** Adapting a state-of-the-art multimodal model (Qwen2.5-VL-7B-Instruct) to a specific, complex domain (dermatology image analysis).
*   **Parameter-Efficient Fine-Tuning (PEFT):** Implementing Low-Rank Adaptation (LoRA) to significantly reduce the number of trainable parameters, enabling efficient fine-tuning with limited computational resources.
*   **Quantization (QLoRA-style):** Leveraging 4-bit quantization via `bitsandbytes` (using the NF4 type and double quantization) to further decrease the memory footprint during training and inference.
*   **Multimodal Data Handling:** Creating a custom PyTorch `Dataset` (`dataset.py`) to effectively load, preprocess (resize, normalize), and format paired image-text data according to the model's requirements, including handling image paths and using the `transformers` `AutoProcessor`.
*   **Custom Training Loop:** Implementing a robust training loop (`trainer.py`) featuring:
    *   **Gradient Accumulation:** Simulating larger effective batch sizes to stabilize training.
    *   **Automatic Mixed Precision (AMP):** Accelerating training and reducing memory usage on compatible GPUs (FP16/BF16).
    *   **Gradient Clipping:** Preventing exploding gradients for more stable training.
*   **Learning Rate Scheduling:** Utilizing a cosine learning rate scheduler with a warmup phase for optimized convergence.
*   **Selective Layer Freezing:** Strategically freezing the parameters of the pre-trained vision encoder (`model_utils.py`) to preserve its powerful feature extraction capabilities while focusing fine-tuning resources on the language components and LoRA adapters.
*   **Configuration Management:** Using `argparse` (`qwen_vl_finetune.py`) to manage hyperparameters and paths, making the script flexible and reproducible.
*   **Checkpointing:** Implementing periodic saving of LoRA adapter weights and processor states during training for fault tolerance and resuming runs.
*   **Metric Logging & Visualization:** Tracking essential metrics (loss, learning rate) during training and generating plots (`utils.py`) for analysis.
*   **Modular Code Structure:** Organizing the code into logical Python modules (`dataset.py`, `model_utils.py`, `trainer.py`, `utils.py`) for better readability, maintainability, and reusability.
*   **Error Handling:** Incorporating basic error handling for missing files and potential issues during data loading or training steps.

## Project Structure

qwen-vl-finetuning/
├── qwen_vl_finetune.py # Main script (entry point) - Handles args, orchestrates training
├── dataset.py # Defines the SkinDiseaseDataset class for data loading
├── model_utils.py # Utility functions for loading model, processor, and setting up LoRA/freezing
├── trainer.py # Contains the FineTuneTrainer class implementing the training loop
├── utils.py # Helper functions, e.g., plotting training metrics
├── requirements.txt # Python dependencies
├── README.md # This file
└── .gitignore # Git ignore configuration


## Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/qwen-vl-finetuning.git # Replace with your repo URL
    cd qwen-vl-finetuning
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *   **Note:** `bitsandbytes` installation can sometimes be tricky depending on your CUDA version. If you encounter issues, consult the `bitsandbytes` documentation or try installing a specific version compatible with your setup. Ensure you have the NVIDIA drivers and CUDA toolkit installed if using a GPU.

4.  **Prepare Data:**
    *   Place your training dataset JSON file (e.g., `formatted_dataset.json`) in an accessible location.
    *   Place the corresponding image directory (e.g., `merged_total-4874img/`) in an accessible location.
    *   You will need to provide the paths to these via command-line arguments when running the script.

## Usage

Execute the main fine-tuning script from your terminal:

```bash
python qwen_vl_finetune.py \
    --dataset_path "/path/to/your/formatted_dataset.json" \
    --image_dir "/path/to/your/image_directory" \
    --output_dir "./qwen_vl_finetuned_output" \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --logging_steps 10 \
    --save_steps 250 \
    --warmup_steps 10 \
    --freeze_vision_encoder \
    --use_amp \
    # --bf16  # Add this flag if your GPU supports bfloat16 (Ampere or newer)
    # --max_steps 1000 # Optionally set a maximum number of steps
    # --no_4bit # Add this flag to disable 4-bit quantization (requires much more VRAM)

Key Command-Line Arguments:
    --dataset_path: Required. Path to the training JSON file.
    --image_dir: Required. Path to the directory containing training images.
    --output_dir: Directory to save checkpoints and the final model adapters.
    --model_name_or_path: Base model identifier from Hugging Face Hub or local path.
    --learning_rate: Peak learning rate for the AdamW optimizer.
    --num_train_epochs: Total number of times to iterate over the dataset.
    --batch_size: Number of samples processed per device before gradient accumulation.
    --gradient_accumulation_steps: Number of steps to accumulate gradients before an optimizer step. Effective batch size is batch_size * num_gpus * gradient_accumulation_steps.
    --lora_r, --lora_alpha, --lora_dropout: LoRA configuration parameters.
    --freeze_vision_encoder / --no-freeze_vision_encoder: Control freezing of the vision encoder (default is frozen).
    --use_amp / --no_amp: Enable/disable Automatic Mixed Precision (default enabled on CUDA).
    --use_4bit / --no_4bit: Enable/disable 4-bit quantization (default enabled).
    --bf16: Use bfloat16 for AMP/quantization if supported.
    --logging_steps: How often to print training logs.
    --save_steps: How often to save model checkpoints.
    --warmup_steps: Number of warmup steps for the learning rate scheduler.
    --max_steps: Optional maximum number of training steps (overrides epochs).
    --device: Device to use (e.g., cuda:0, cpu).
    Run python qwen_vl_finetune.py --help for a full list of options.
    Model & Fine-Tuning Details
    Base Model: Qwen2.5-VL-7B-Instruct
    Fine-Tuning Method: LoRA (Parameter-Efficient Fine-Tuning)
    Quantization: 4-bit NF4 with double quantization (QLoRA-style) via bitsandbytes.
    LoRA Config:
    Rank (r): 8 (configurable via --lora_r)
    Alpha (lora_alpha): 16 (configurable via --lora_alpha)
    Dropout (lora_dropout): 0.05 (configurable via --lora_dropout)
    Target Modules: Attention projections (q_proj, k_proj, v_proj, o_proj) and Feed-Forward layers (gate_proj, up_proj, down_proj).
    Frozen Layers: The vision encoder (model.visual) weights are frozen by default to preserve robust image understanding capabilities learned during pre-training. Only the LoRA adapters and potentially language model components (depending on LoRA target modules) are trained.