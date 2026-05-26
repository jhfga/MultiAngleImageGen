import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline, BitsAndBytesConfig


def load_model_nf4(
    model_path: str = "./models/Qwen-Image-Edit-2511",
    lora_path: str | None = "./models/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
):
    """
    以 NF4 量化 + CPU 卸载方式加载 Qwen-Image-Edit-2511 模型。
    Args:
        model_path: 本地模型路径
        lora_path: LoRA 权重路径

    Returns:
        QwenImageEditPlusPipeline 实例
    """
    # 仅量化 Transformer 为 NF4，Text Encoder 和 VAE 保持 BF16
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        transformer_4bit_quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
    )

    # 根据CUDA可用性选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("警告：未检测到CUDA，将使用CPU运行，速度会非常慢")
    pipe.to(device)

    # 启用推理进度条
    pipe.set_progress_bar_config(disable=None)

    # 启用内存优化，减少显存占用
    pipe.enable_model_cpu_offload()

    # 加载 LoRA（不 fuse，推理时通过 lora_scale 动态控制强度）
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
    lora_scale: float = 1.0,
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
        lora_scale: LoRA 强度，1.0 为满强度，0.0 等于不启用 LoRA

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
            input_images = input_images.resize((int(w * ratio), int(h * ratio)))
    else:
        resized = []
        for img in input_images:
            w, h = img.size
            if max(w, h) > max_image_size:
                ratio = max_image_size / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)))
            resized.append(img)
        input_images = resized

    generator = torch.Generator(device="cpu").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=input_images,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        cross_attention_kwargs={"scale": lora_scale},
    )

    output_image = result.images[0]
    output_image.save(output_path)
    print(f"图片已保存至 {output_path}")
    return output_image


if __name__ == "__main__":
    pipe = load_model_nf4()

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
            lora_scale=1.0,
        )
