import os
import tempfile
import time
from PIL import Image
from lightx2v import LightX2VPipeline


def load_model_fp8_lightning(
    model_path: str = "./models/Qwen-Image-Edit-2511",
    fp8_ckpt: str = "./models/Qwen-image-edit-2511-fp8-4steps/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors",
    lora_path: str | None = "./models/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
):
    """
    Args:
        model_path: 基础模型目录路径
        fp8_ckpt: FP8+Lightning 合并权重文件路径
        lora_path: 多角度 LoRA 权重路径

    Returns:
        LightX2VPipeline 实例
    """
    pipe = LightX2VPipeline(
        model_path=model_path,
        model_cls="qwen-image-edit-2511",
        task="i2i",
    )

    # 加载 FP8+Lightning 合并权重
    pipe.enable_quantize(
        dit_quantized=True,
        dit_quantized_ckpt=fp8_ckpt,
        quant_scheme="fp8-sgl",
        text_encoder_quantized=True,
        text_encoder_quantized_ckpt="models/Qwen25-VL-4bit-GPTQ",
        text_encoder_quant_scheme="int4",
    )

    # 加载多角度 LoRA
    if lora_path is not None:
        pipe.enable_lora(
            [{"path": lora_path, "strength": 1.0}],
            lora_dynamic_apply=False,
        )

    # 创建生成器
    pipe.create_generator(
        attn_mode="flash_attn3",
        resize_mode="adaptive",
        infer_steps=4,
        guidance_scale=1,
    )

    pipe.CONDITION_IMAGE_SIZE = 1048576

    return pipe


MAX_SIZE = 1024
_temp_dir = None


def _get_temp_dir():
    global _temp_dir
    if _temp_dir is None:
        _temp_dir = tempfile.mkdtemp(prefix="resized_")
    return _temp_dir


def resize_image_if_needed(image_path: str, max_size: int = MAX_SIZE) -> str:
    """当图片宽或高超限时，等比例缩放到范围内，返回缩放后的临时路径；无需缩放则返回原路径。"""
    img = Image.open(image_path)
    w, h = img.size
    if w <= max_size and h <= max_size:
        return image_path

    ratio = min(max_size / w, max_size / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    save_path = os.path.join(_get_temp_dir(), f"resized_{os.path.basename(image_path)}")
    img.save(save_path)
    print(f"图片缩放: {w}x{h} -> {new_w}x{new_h}")
    return save_path


def benchmark_inference(
    pipe,
    image_path: str | list[str],
    prompt: str,
    output_path: str,
    seed: int = 42,
):
    """
    使用 FP8 Lightning 模型执行图像编辑推理。

    Args:
        pipe: 已加载的 LightX2VPipeline 实例
        image_path: 输入图片路径，支持单张或多张（逗号分隔）
        prompt: 编辑指令，如 "侧面视角，仰角15度"
        output_path: 输出图片保存路径
        seed: 随机种子
    """
    # LightX2V 支持多图用逗号分隔
    if isinstance(image_path, list):
        image_path = [resize_image_if_needed(p) for p in image_path]
        image_path = ",".join(image_path)
    else:
        image_path = resize_image_if_needed(image_path)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    pipe.generate(
        seed=seed,
        image_path=image_path,
        prompt=prompt,
        negative_prompt="",
        save_result_path=output_path,
    )
    print(f"图片已保存至 {output_path}")


if __name__ == "__main__":
    pipe = load_model_fp8_lightning()

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
