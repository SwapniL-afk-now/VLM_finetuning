# trainer.py
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import get_cosine_schedule_with_warmup
import time
import os
import matplotlib.pyplot as plt
from dataset import SkinDiseaseDataset # Import from local dataset.py

class FineTuneTrainer:
    """
    Trainer class to handle the fine-tuning loop, optimization, scheduling,
    gradient accumulation, checkpointing, and metric logging.
    """
    def __init__(self, model, processor, config, train_dataset_list, image_dir):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model # Assumes model is already LoRA-wrapped and potentially frozen
        self.processor = processor

        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
            print(f"Set tokenizer pad_token_id to eos_token_id: {self.processor.tokenizer.pad_token_id}")

        print("Initializing SkinDiseaseDataset...")
        self.train_dataset = SkinDiseaseDataset(train_dataset_list, image_dir, processor)
        if len(self.train_dataset) == 0:
            raise ValueError("Training dataset is empty. Check dataset path, image directory, and JSON format.")

        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=True,
            num_workers=4 if config.device.startswith("cuda") else 0,
            pin_memory=config.device.startswith("cuda"),
            collate_fn=self.collate_fn
        )
        print(f"DataLoader initialized with {len(self.dataloader)} batches.")

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Calculate total training steps based on dataloader length
        steps_per_epoch = len(self.dataloader) // config.gradient_accumulation_steps
        self.total_steps = min(steps_per_epoch * config.num_train_epochs, config.max_steps) \
                           if config.max_steps > 0 else steps_per_epoch * config.num_train_epochs

        if self.total_steps <= 0:
             raise ValueError(f"Total steps calculated to be {self.total_steps}. Check batch size ({config.per_device_train_batch_size}), accumulation steps ({config.gradient_accumulation_steps}), dataset size ({len(self.train_dataset)}), epochs ({config.num_train_epochs}), or max_steps ({config.max_steps}).")
        print(f"Calculated total training steps: {self.total_steps}")

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=self.total_steps
        )

        self.scaler = torch.amp.GradScaler(device_type='cuda') if config.device.startswith("cuda") and config.use_amp else None
        if self.scaler:
            print("Using Automatic Mixed Precision (AMP).")

        self.step = 0
        self.metrics = {"loss": [], "lr": []}
        self.start_time = time.time()

    def collate_fn(self, batch):
        """Pads sequences within a batch."""
        input_ids = [item['input_ids'] for item in batch if 'input_ids' in item]
        attention_mask = [item['attention_mask'] for item in batch if 'attention_mask' in item]

        if not input_ids: # Handle empty batch case
             return {}

        # Pad input_ids and attention_mask
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        collated_batch = {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask_padded,
        }

        # Stack pixel_values and image_grid_thw if they exist and are valid
        if 'pixel_values' in batch[0] and batch[0]['pixel_values'] is not None:
            try:
                 pixel_values = torch.stack([item['pixel_values'] for item in batch])
                 collated_batch['pixel_values'] = pixel_values
            except Exception as e:
                 print(f"Warning: Error stacking pixel_values: {e}. Skipping pixel_values for this batch.")

        if 'image_grid_thw' in batch[0] and batch[0]['image_grid_thw'] is not None:
             try:
                  image_grid_thw = torch.stack([item['image_grid_thw'] for item in batch])
                  collated_batch['image_grid_thw'] = image_grid_thw
             except Exception as e:
                  print(f"Warning: Error stacking image_grid_thw: {e}. Skipping image_grid_thw for this batch.")


        return collated_batch


    def train(self):
        self.model.train()
        accumulation_counter = 0
        global_step_time_accumulator = 0.0
        log_step_start_time = time.time()

        print(f"Starting training for {self.config.num_train_epochs} epochs ({self.total_steps} effective steps)...")

        for epoch in range(self.config.num_train_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.config.num_train_epochs} ---")
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(self.dataloader):
                if not batch: # Skip empty batches from collate_fn
                     print(f"Skipping empty batch at index {batch_idx}")
                     continue

                if self.step >= self.total_steps:
                    print("Max steps reached. Finishing training.")
                    break

                # Move batch to device
                try:
                    batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                except Exception as e:
                    print(f"Error moving batch {batch_idx} to device: {e}")
                    continue # Skip this batch

                # Check if essential keys are present after moving
                if 'input_ids' not in batch or 'attention_mask' not in batch:
                     print(f"Warning: Batch {batch_idx} missing essential keys after moving to device. Skipping.")
                     continue

                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.scaler is not None,
                    dtype=torch.bfloat16 if self.config.bf16 and torch.cuda.is_bf16_supported() else torch.float16
                ):
                    try:
                        outputs = self.model(**batch, labels=batch["input_ids"])
                        loss = outputs.loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"Warning: NaN or Inf loss encountered at step {self.step}, batch {batch_idx}. Skipping update.")
                            # Optionally clear gradients if accumulated
                            if accumulation_counter > 0:
                                 self.optimizer.zero_grad(set_to_none=True)
                                 accumulation_counter = 0
                            continue

                        loss = loss / self.config.gradient_accumulation_steps
                    except Exception as e:
                         print(f"Error during forward pass at step {self.step}, batch {batch_idx}: {e}")
                         # Optionally try to recover or just skip
                         if accumulation_counter > 0:
                             self.optimizer.zero_grad(set_to_none=True)
                             accumulation_counter = 0
                         continue

                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accumulation_counter += 1

                if accumulation_counter >= self.config.gradient_accumulation_steps:
                    step_start_time = time.time() # Time the optimizer step + logging/saving

                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    current_loss = loss.item() * self.config.gradient_accumulation_steps
                    self.metrics["loss"].append(current_loss)
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.metrics["lr"].append(current_lr)

                    # Increment step counter *after* successful optimizer step
                    self.step += 1

                    step_duration = time.time() - step_start_time
                    global_step_time_accumulator += step_duration

                    if self.step % self.config.logging_steps == 0:
                         avg_step_time_since_log = (time.time() - log_step_start_time) / self.config.logging_steps
                         print(f"Step {self.step}/{self.total_steps} | "
                               f"Loss: {current_loss:.4f} | "
                               f"LR: {current_lr:.6f} | "
                               f"Avg Step Time (last {self.config.logging_steps}): {avg_step_time_since_log:.2f}s")
                         log_step_start_time = time.time() # Reset timer for next log interval


                    if self.step > 0 and self.step % self.config.save_steps == 0:
                        checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint-{self.step}")
                        os.makedirs(checkpoint_path, exist_ok=True)
                        print(f"Saving checkpoint to {checkpoint_path}...")
                        self.model.save_pretrained(checkpoint_path)
                        self.processor.save_pretrained(checkpoint_path)
                        print(f"Checkpoint saved.")

                    # Reset accumulation counter for the next set
                    accumulation_counter = 0

                if self.step >= self.total_steps:
                    break # Exit inner loop

            epoch_duration = time.time() - epoch_start_time
            print(f"--- Epoch {epoch + 1} completed in {epoch_duration / 3600:.2f} hours ---")

            if self.step >= self.total_steps:
                break # Exit outer loop

        # --- Final Save & Wrap-up ---
        final_model_path = os.path.join(self.config.output_dir, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        print(f"\nSaving final model adapters and processor to {final_model_path}...")
        self.model.save_pretrained(final_model_path)
        self.processor.save_pretrained(final_model_path)
        print("Final model saved.")

        total_time = time.time() - self.start_time
        avg_step_time_overall = global_step_time_accumulator / self.step if self.step > 0 else 0
        print(f"\nTotal training time: {total_time / 3600:.2f} hours")
        print(f"Average step time (optimizer steps): {avg_step_time_overall:.2f}s")

        # --- Plotting ---
        from utils import plot_training_metrics # Import plotting utility
        plot_training_metrics(self.metrics, self.config.output_dir, self.step)