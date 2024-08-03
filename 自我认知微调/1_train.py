"""
自我认知训练
"""

# Experimental environment: 3090, V100, ...
# 24GB GPU memory
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from swift.llm import DatasetName, ModelType, SftArguments, sft_main

model_dir = "/root/home/my_model/Qwen2-7B-Instruct"

sft_args = SftArguments(
    model_type=ModelType.qwen2_7b_instruct,
    model_id_or_path=model_dir,
    dataset=[f"{DatasetName.alpaca_zh}#500", f"{DatasetName.alpaca_en}#500", f"{DatasetName.self_cognition}#500"],
    max_length=2048,
    learning_rate=1e-4,
    output_dir="output",
    lora_target_modules=["ALL"],
    model_name=["迪小乐", "dd"],
    model_author=["迪迪", "dd"],
)
output = sft_main(sft_args)
best_model_checkpoint = output["best_model_checkpoint"]
print(f"best_model_checkpoint: {best_model_checkpoint}")

"""Out[0]
[INFO:swift] The logging file will be saved in: /xxx/output/qwen2-7b-instruct/v2-20240607-101038/logging.jsonl
{'loss': 1.8210969, 'acc': 0.6236614, 'grad_norm': 2.75, 'learning_rate': 2e-05, 'memory(GiB)': 16.79, 'train_speed(iter/s)': 0.155172, 'epoch': 0.01, 'global_step': 1}
{'loss': 1.75309932, 'acc': 0.63371617, 'grad_norm': 3.765625, 'learning_rate': 0.0001, 'memory(GiB)': 18.48, 'train_speed(iter/s)': 0.210486, 'epoch': 0.05, 'global_step': 5}
{'loss': 1.42493172, 'acc': 0.65476351, 'grad_norm': 1.671875, 'learning_rate': 9.432e-05, 'memory(GiB)': 18.48, 'train_speed(iter/s)': 0.221159, 'epoch': 0.11, 'global_step': 10}
{'loss': 1.16402645, 'acc': 0.69853611, 'grad_norm': 2.3125, 'learning_rate': 8.864e-05, 'memory(GiB)': 18.48, 'train_speed(iter/s)': 0.223072, 'epoch': 0.16, 'global_step': 15}
{'loss': 1.18519087, 'acc': 0.68314366, 'grad_norm': 1.7578125, 'learning_rate': 8.295e-05, 'memory(GiB)': 18.48, 'train_speed(iter/s)': 0.224677, 'epoch': 0.21, 'global_step': 20}
{'loss': 1.09617777, 'acc': 0.69949636, 'grad_norm': 1.4296875, 'learning_rate': 7.727e-05, 'memory(GiB)': 19.46, 'train_speed(iter/s)': 0.225241, 'epoch': 0.27, 'global_step': 25}
{'loss': 1.09035854, 'acc': 0.70226536, 'grad_norm': 1.34375, 'learning_rate': 7.159e-05, 'memory(GiB)': 19.46, 'train_speed(iter/s)': 0.226112, 'epoch': 0.32, 'global_step': 30}
{'loss': 1.04421387, 'acc': 0.71705227, 'grad_norm': 1.65625, 'learning_rate': 6.591e-05, 'memory(GiB)': 19.46, 'train_speed(iter/s)': 0.225783, 'epoch': 0.38, 'global_step': 35}
{'loss': 0.97917967, 'acc': 0.73127871, 'grad_norm': 1.2265625, 'learning_rate': 6.023e-05, 'memory(GiB)': 19.46, 'train_speed(iter/s)': 0.226212, 'epoch': 0.43, 'global_step': 40}
{'loss': 0.94920969, 'acc': 0.74032536, 'grad_norm': 0.9140625, 'learning_rate': 5.455e-05, 'memory(GiB)': 19.46, 'train_speed(iter/s)': 0.225991, 'epoch': 0.48, 'global_step': 45}
{'loss': 0.99205322, 'acc': 0.73348026, 'grad_norm': 1.1640625, 'learning_rate': 4.886e-05, 'memory(GiB)': 19.46, 'train_speed(iter/s)': 0.224141, 'epoch': 0.54, 'global_step': 50}
Train:  54%|███████████████████████████████████▍                              | 50/93 [03:42<03:19,  4.64s/it]
{'eval_loss': 1.03679836, 'eval_acc': 0.67676003, 'eval_runtime': 1.2396, 'eval_samples_per_second': 8.874, 'eval_steps_per_second': 8.874, 'epoch': 0.54, 'global_step': 50}
Val: 100%|████████████████████████████████████████████████████████████████████| 11/11 [00:01<00:00, 10.15it/s]
[INFO:swift] Saving model checkpoint to /xxx/output/qwen2-7b-instruct/v2-20240607-101038/checkpoint-50
{'loss': 0.98644152, 'acc': 0.73600368, 'grad_norm': 2.0625, 'learning_rate': 4.318e-05, 'memory(GiB)': 20.5, 'train_speed(iter/s)': 0.220983, 'epoch': 0.59, 'global_step': 55}
{'loss': 0.97522211, 'acc': 0.7305594, 'grad_norm': 1.1640625, 'learning_rate': 3.75e-05, 'memory(GiB)': 20.5, 'train_speed(iter/s)': 0.218717, 'epoch': 0.64, 'global_step': 60}
{'loss': 1.02459459, 'acc': 0.71822615, 'grad_norm': 1.125, 'learning_rate': 3.182e-05, 'memory(GiB)': 20.5, 'train_speed(iter/s)': 0.216185, 'epoch': 0.7, 'global_step': 65}
{'loss': 0.90719929, 'acc': 0.73806977, 'grad_norm': 1.078125, 'learning_rate': 2.614e-05, 'memory(GiB)': 20.5, 'train_speed(iter/s)': 0.21451, 'epoch': 0.75, 'global_step': 70}
{'loss': 0.88519163, 'acc': 0.74690943, 'grad_norm': 1.3359375, 'learning_rate': 2.045e-05, 'memory(GiB)': 20.5, 'train_speed(iter/s)': 0.21366, 'epoch': 0.81, 'global_step': 75}
{'loss': 0.95856657, 'acc': 0.72634115, 'grad_norm': 1.359375, 'learning_rate': 1.477e-05, 'memory(GiB)': 20.5, 'train_speed(iter/s)': 0.213132, 'epoch': 0.86, 'global_step': 80}
{'loss': 0.88609543, 'acc': 0.75917048, 'grad_norm': 0.90625, 'learning_rate': 9.09e-06, 'memory(GiB)': 20.5, 'train_speed(iter/s)': 0.211609, 'epoch': 0.91, 'global_step': 85}
{'loss': 0.97113533, 'acc': 0.73501945, 'grad_norm': 2.40625, 'learning_rate': 3.41e-06, 'memory(GiB)': 20.5, 'train_speed(iter/s)': 0.210918, 'epoch': 0.97, 'global_step': 90}
Train: 100%|██████████████████████████████████████████████████████████████████| 93/93 [07:21<00:00,  5.05s/it]
{'eval_loss': 1.03077412, 'eval_acc': 0.68508706, 'eval_runtime': 1.2226, 'eval_samples_per_second': 8.997, 'eval_steps_per_second': 8.997, 'epoch': 1.0, 'global_step': 93}
Val: 100%|████████████████████████████████████████████████████████████████████| 11/11 [00:01<00:00, 10.26it/s]
[INFO:swift] Saving model checkpoint to /xxx/output/qwen2-7b-instruct/v2-20240607-101038/checkpoint-93
{'train_runtime': 443.3746, 'train_samples_per_second': 3.358, 'train_steps_per_second': 0.21, 'train_loss': 1.07190883, 'epoch': 1.0, 'global_step': 93}
Train: 100%|██████████████████████████████████████████████████████████████████| 93/93 [07:23<00:00,  4.77s/it]
[INFO:swift] last_model_checkpoint: /xxx/output/qwen2-7b-instruct/v2-20240607-101038/checkpoint-93
[INFO:swift] best_model_checkpoint: /xxx/output/qwen2-7b-instruct/v2-20240607-101038/checkpoint-93
[INFO:swift] images_dir: /xxx/output/qwen2-7b-instruct/v2-20240607-101038/images
[INFO:swift] End time of running main: 2024-06-07 10:18:41.386561
best_model_checkpoint: /xxx/output/qwen2-7b-instruct/v2-20240607-101038/checkpoint-93
"""
