import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from swift.llm import AppUIArguments, merge_lora, app_ui_main

best_model_checkpoint = (
    "/root/home/ms_swift_demo/自我认知微调/output/qwen2-7b-instruct/v3-20240803-144855/checkpoint-93"
)
if os.path.exists(best_model_checkpoint + "-merged"):
    print("已经合并过模型")
    best_model_checkpoint += "-merged"
    app_ui_args = AppUIArguments(ckpt_dir=best_model_checkpoint)
else:
    # 需要先合并模型
    print("先合并模型")
    app_ui_args = AppUIArguments(ckpt_dir=best_model_checkpoint)
    merge_lora(best_model_checkpoint, device_map="cpu")

result = app_ui_main(app_ui_args)
