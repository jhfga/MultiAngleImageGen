import torch
from PIL import Image
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler


def load_model_fp8_lightning(
    model_path: str = "./models/Qwen-image-edit-2511-fp8-4steps/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors",
    lora_path: str | None = "./models/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
):
    """
    加载 FP8 量化 + Lightning 4步加速版本的 Qwen-Image-Edit-2511 模型。
    Args:
        model_path: 本地 FP8 safetensors 模型路径
        lora_path: LoRA 权重路径

    Returns:
        DiffusionPipeline 实例
    """
    pipe = DiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        trust_remote_code=True,
        load_safety_checker=False,
    ).to("cuda")

    # Lightning 4步专属调度器
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        prediction_type="v_prediction",
    )

    # 启用推理进度条
    pipe.set_progress_bar_config(disable=None)

    # 加载 LoRA
    if lora_path is not None:
        pipe.load_lora_weights(lora_path)

    return pipe


def run_inference(
    pipe: DiffusionPipeline,
    image_path: str | list[str],
    prompt: str,
    output_path: str,
    seed: int,
    num_inference_steps: int = 4,
    guidance_scale: float = 7.5,
    strength: float = 0.8,
    max_image_size: int = 1024,
) -> Image.Image:
    """
    使用 FP8 Lightning 模型执行图像编辑推理。

    Args:
        pipe: 已加载的 DiffusionPipeline 实例
        image_path: 输入图片路径，支持单张或多张（用于多人融合等场景）
        prompt: 编辑指令，如 "把背景换成海滩" 或 "侧面视角，仰角15度"
        output_path: 输出图片保存路径
        seed: 随机种子，保证可复现
        num_inference_steps: 推理步数（Lightning 模型固定 4 步）
        guidance_scale: 引导系数，越高越贴合指令
        strength: 图像编辑强度，0-1 之间
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
        "strength": strength,
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
    pipe = load_model_fp8_lightning()

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
            num_inference_steps=4,
            guidance_scale=7.5,
            strength=0.8,
            max_image_size=1024,
        )
