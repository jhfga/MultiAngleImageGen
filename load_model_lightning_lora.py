import math
import os
import time
import torch
from PIL import Image
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
)
from diffusers.models import QwenImageTransformer2DModel


def load_model_lightning_lora(
    model_path: str = "./models/Qwen-Image-Edit-2511-4bit",
    lightning_lora_path: str = "./models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    multi_angle_lora_path: str | None = "./models/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/qwen-image-edit-2511-multiple-angles-lora.safetensors",
):
    """
    不支持FP8的GPU可以使用，结合了Lightning LoRA和多角度 LoRA。

    Args:
        model_path: 基础模型目录路径
        lightning_lora_path: Lightning 蒸馏 LoRA 权重路径
        multi_angle_lora_path: 多角度 LoRA 权重路径

    Returns:
        QwenImageEditPlusPipeline 实例
    """
    torch_dtype = torch.bfloat16

    # Lightning LoRA 需要配合特定的 scheduler 配置
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    # 先加载 transformer，再加载 LoRA
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch_dtype
    )

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch_dtype,
    )
    pipe.to("cuda")

    # 加载 Lightning LoRA
    pipe.load_lora_weights(lightning_lora_path, adapter_name="lightning")

    # 加载多角度 LoRA（与 Lightning LoRA 同时启用）
    if multi_angle_lora_path is not None:
        pipe.load_lora_weights(multi_angle_lora_path, adapter_name="multi_angle")
        pipe.set_adapters(["lightning", "multi_angle"])

    return pipe


if __name__ == "__main__":
    MAX_SIZE = 1024

    def resize_image_if_needed(img: Image.Image, max_size: int = MAX_SIZE) -> Image.Image:
        """当图片宽或高超限时，等比例缩放到范围内。"""
        w, h = img.size
        if w <= max_size and h <= max_size:
            return img

        ratio = min(max_size / w, max_size / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"图片缩放: {w}x{h} -> {new_w}x{new_h}")
        return img

    def benchmark_inference(
        pipe: QwenImageEditPlusPipeline,
        image_path: str | list[str],
        prompt: str,
        output_path: str,
        seed: int = 42,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
    ):
        """
        使用 Lightning LoRA 模型执行图像编辑推理。

        Args:
            pipe: 已加载的 QwenImageEditPlusPipeline 实例
            image_path: 输入图片路径，支持单张或多张
            prompt: 编辑指令，如 "侧面视角，仰角15度"
            output_path: 输出图片保存路径
            seed: 随机种子
            num_inference_steps: 推理步数（Lightning LoRA 推荐 4 步）
            guidance_scale: 引导系数（Lightning LoRA 推荐 1.0）
        """
        # 加载输入图片
        if isinstance(image_path, list):
            input_images = [resize_image_if_needed(Image.open(p).convert("RGB")) for p in image_path]
        else:
            input_images = resize_image_if_needed(Image.open(image_path).convert("RGB"))

        # 获取输出尺寸，对齐到 vae_scale_factor * 2 的倍数
        if isinstance(input_images, list):
            final_w, final_h = input_images[0].size
        else:
            final_w, final_h = input_images.size
        required_div = pipe.vae_scale_factor * 2
        final_w = round(final_w / required_div) * required_div
        final_h = round(final_h / required_div) * required_div

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        generator = torch.Generator(device="cpu").manual_seed(seed)

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                image=input_images,
                width=final_w,
                height=final_h,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                true_cfg_scale=1.0,
                negative_prompt=" ",
                generator=generator,
            )

        output_image = result.images[0]
        output_image.save(output_path)
        print(f"图片已保存至 {output_path}")

    pipe = load_model_lightning_lora()

    prompts = [
        "<sks> front view eye-level shot medium shot",
        "<sks> right side view high-angle shot close-up",
        "<sks> back view low-angle shot wide shot",
        "<sks> front-left quarter view elevated shot medium shot",
    ]

    total_time = 0.0
    for i, prompt in enumerate(prompts):
        output_path = f"./output_{i + 1}.png"
        start = time.time()
        benchmark_inference(
            pipe=pipe,
            image_path="./test.png",
            prompt=prompt,
            output_path=output_path,
            seed=42,
        )
        elapsed = time.time() - start
        total_time += elapsed
        print(f"第 {i + 1} 次推理耗时: {elapsed:.2f}s")

    avg = total_time / len(prompts)
    print(f"\n平均每次推理耗时: {avg:.2f}s")
