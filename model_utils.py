# model_utils.py
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def load_model_and_processor(model_name_or_path, device="cuda:0", use_quantization=True, bf16=False):
    """Loads the Qwen VL model and processor."""
    print(f"Loading processor for {model_name_or_path}...")
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

    quantization_config = None
    if use_quantization and torch.cuda.is_available():
        print("Setting up 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 and torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        quantization_config = bnb_config
        print("Using BitsAndBytes 4-bit NF4 quantization.")
    elif use_quantization:
        print("Warning: Quantization requested but CUDA not available. Loading in full precision.")

    print(f"Loading model {model_name_or_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        device_map=device if quantization_config else None, # device_map only works well with quantization
        torch_dtype=torch.bfloat16 if bf16 and torch.cuda.is_bf16_supported() else torch.float16 if use_quantization else torch.float32, # Match dtype
        trust_remote_code=True
    )

    # If not using device_map (e.g., no quantization or CPU), move model manually
    if not quantization_config and device != 'cpu':
         model.to(device)

    print("Model and processor loaded.")
    return model, processor

def setup_lora(model, lora_r, lora_alpha, lora_dropout, target_modules):
    """Applies LoRA configuration to the model."""
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM" # Specific to causal language models like Qwen
    )
    model = get_peft_model(model, lora_config)
    print("LoRA applied.")
    model.print_trainable_parameters() # Shows which parameters will be trained
    return model

def freeze_vision_encoder(model):
    """Freezes the parameters of the vision encoder part of the model."""
    print("Attempting to freeze vision encoder parameters...")
    freeze_count = 0
    unfreeze_count = 0
    if hasattr(model, 'visual'): # Standard attribute name in many VLMs
        for name, param in model.visual.named_parameters():
            param.requires_grad = False
            freeze_count += 1
        print(f"Froze {freeze_count} parameters in model.visual.")

        # Check if the base model (if using PEFT) also has it
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'visual'):
             print("Also checking base_model.visual for freezing...")
             for name, param in model.base_model.visual.named_parameters():
                  if 'lora' not in name: # Don't freeze LoRA layers if they somehow ended up here
                       param.requires_grad = False
                       freeze_count += 1
                  else:
                       unfreeze_count +=1
             print(f"Processed base_model.visual: Froze {freeze_count} additional, kept {unfreeze_count} unfrozen (likely LoRA).")

    else:
        print("Warning: 'visual' attribute not found directly on the model or base_model. Vision encoder might not be frozen.")

    # Verify by checking requires_grad status
    # vision_params_frozen = True
    # if hasattr(model, 'visual'):
    #     for param in model.visual.parameters():
    #         if param.requires_grad:
    #             vision_params_frozen = False
    #             break
    # print(f"Verification: Vision parameters frozen: {vision_params_frozen}")

    return model