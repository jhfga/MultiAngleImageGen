import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline


def load_model_4bit(
    model_path: str = "./models/Qwen-Image-Edit-2511-4bit",
    lora_path: str | None = "./models/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
):
    """
    加载预量化 4-bit 版本的 Qwen-Image-Edit-2511 模型。
    Args:
        model_path: 本地 4-bit 模型路径
        lora_path: LoRA 权重路径

    Returns:
        QwenImageEditPlusPipeline 实例
    """
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )

    # 模型全部加载到 GPU（显存足够时性能更好）
    pipe.to("cuda")

    # 启用推理进度条
    pipe.set_progress_bar_config(disable=None)

    # 加载 LoRA
    if lora_path is not None:
        pipe.load_lora_weights(lora_path)

    return pipe


def run_inference(
    pipe: QwenImageEditPlusPipeline,
    image_path: str | list[str],
    prompt: str,
    output_path: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    max_image_size: int,
) -> Image.Image:
    """
    使用 Qwen-Image-Edit-2511 执行图像编辑推理。

    Args:
        pipe: 已加载的 DiffusionPipeline 实例
        image_path: 输入图片路径，支持单张或多张（用于多人融合等场景）
        prompt: 编辑指令，如 "把背景换成海滩" 或 "侧面视角，仰角15度"
        output_path: 输出图片保存路径
        seed: 随机种子，保证可复现
        num_inference_steps: 推理步数（使用 Lightning LoRA 时可设为 4，标准推理通常 30-50）
        guidance_scale: 引导系数，越高越贴合指令，通常 3.0-7.0
        max_image_size: 输入图像最大边长，超出会等比缩放；显存越大可设越大

    Returns:
        生成的 PIL Image 对象
    """
    # 加载输入图片
    if isinstance(image_path, str):
        input_images = Image.open(image_path).convert("RGB")
    else:
        input_images = [Image.open(p).convert("RGB") for p in image_path]

    # 限制图像尺寸，防止 OOM
    if isinstance(input_images, Image.Image):
        w, h = input_images.size
        if max(w, h) > max_image_size:
            ratio = max_image_size / max(w, h)
            w, h = int(w * ratio), int(h * ratio)
            input_images = input_images.resize((w, h))
    else:
        resized = []
        for img in input_images:
            w, h = img.size
            if max(w, h) > max_image_size:
                ratio = max_image_size / max(w, h)
                w, h = int(w * ratio), int(h * ratio)
                img = img.resize((w, h))
            resized.append(img)
        input_images = resized

    # 获取最终输入图像尺寸，并指定为输出尺寸（对齐到 vae_scale_factor*2 的倍数）
    if isinstance(input_images, list):
        final_w, final_h = input_images[0].size
    else:
        final_w, final_h = input_images.size
    required_div = pipe.vae_scale_factor * 2
    final_w = round(final_w / required_div) * required_div
    final_h = round(final_h / required_div) * required_div

    generator = torch.Generator(device="cpu").manual_seed(seed)

    inputs = {
        "prompt": prompt,
        "image": input_images,
        "width": final_w,
        "height": final_h,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_images_per_prompt": 1,
        "generator": generator,
    }

    with torch.inference_mode():
        result = pipe(**inputs)

    output_image = result.images[0]
    output_image.save(output_path)
    print(f"图片已保存至 {output_path}")
    return output_image


if __name__ == "__main__":
    pipe = load_model_4bit()

    prompts = [
        "<sks> front view eye-level shot medium shot",
        "<sks> right side view high-angle shot close-up",
        "<sks> back view low-angle shot wide shot",
        "<sks> front-left quarter view elevated shot medium shot",
    ]

    for i, prompt in enumerate(prompts):
        output_path = f"./output_{i + 1}.png"
        run_inference(
            pipe=pipe,
            image_path="./test.png",
            prompt=prompt,
            output_path=output_path,
            seed=42,
            num_inference_steps=20,
            guidance_scale=5.0,
            max_image_size=1024,
        )
