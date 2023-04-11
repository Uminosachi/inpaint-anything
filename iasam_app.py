import os
import torch
import numpy as np
from PIL import Image
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import hashlib
from get_dataset_colormap import create_pascal_label_colormap

IASAM_DEBUG = bool(int(os.environ.get("IASAM_DEBUG", "0")))

device = "cuda" if torch.cuda.is_available() else "cpu"

sam_checkpoint = os.path.join(os.path.dirname(__file__), "sam_vit_h_4b8939.pth")
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

masks = None

cm_pascal = create_pascal_label_colormap()

model_ids = [
    "runwayml/stable-diffusion-inpainting",
    "stabilityai/stable-diffusion-2-inpainting",
    "saik0s/realistic_vision_inpainting",
    ]

def run_sam(input_image):
    print("input_image:", input_image.shape, input_image.dtype)

    global masks
    masks = mask_generator.generate(input_image)

    canvas_image = np.zeros_like(input_image)

    masks = sorted(masks, key=lambda x: np.sum(x.get("segmentation").astype(int)))
    for idx, seg_dict in enumerate(masks):
        seg_mask = np.expand_dims(seg_dict.get("segmentation").astype(int), axis=-1)
        canvas_mask = np.logical_not(np.sum(canvas_image, axis=-1, keepdims=True).astype(bool)).astype(int)
        seg_color = cm_pascal[idx] * seg_mask * canvas_mask
        canvas_image = canvas_image + seg_color
    seg_image = canvas_image.astype(np.uint8)
    
    if IASAM_DEBUG:
        save_name = hashlib.md5(input_image.tobytes()).hexdigest()[0:16] + "_" + os.path.splitext(os.path.basename(sam_checkpoint))[0] + ".png"
        Image.fromarray(seg_image).save(save_name)

    return seg_image

def select_mask(masks_image):
    if masks_image is None or masks is None:
        return None

    image = masks_image["image"]
    mask = masks_image["mask"][:,:,0:3]
    
    canvas_image = np.zeros_like(image)
    mask_region = np.zeros_like(image)
    for idx, seg_dict in enumerate(masks):
        seg_mask = np.expand_dims(seg_dict["segmentation"].astype(int), axis=-1)
        canvas_mask = np.logical_not(np.sum(canvas_image, axis=-1, keepdims=True).astype(bool)).astype(int)
        if (seg_mask * canvas_mask * mask).astype(bool).any():
            mask_region = mask_region + (seg_mask * canvas_mask * 255)
        seg_color = cm_pascal[idx] * seg_mask * canvas_mask        
        canvas_image = canvas_image + seg_color
    seg_image = mask_region.astype(np.uint8)

    if IASAM_DEBUG:
        save_name = hashlib.md5(mask.tobytes()).hexdigest()[0:16] + "_" + "selected_mask" + ".png"
        Image.fromarray(seg_image).save(save_name)

    return seg_image

def run_inpaint(input_image, sel_mask, prompt, n_prompt, ddim_steps, scale, seed, model_id):
    print(model_id)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.safety_checker = None

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()
    
    mask_image = np.sum(sel_mask, axis=-1, keepdims=True).astype(bool).astype(np.uint8) * 255
    mask_image = np.repeat(mask_image, 3, axis=-1)
    
    init_image = Image.fromarray(input_image).convert("RGB")
    mask_image = Image.fromarray(mask_image).convert("RGB")
    print(init_image.size, mask_image.size)
    width, height = init_image.size
    
    generator = torch.Generator(device).manual_seed(seed)
    
    pipe_args_dict = {
        "prompt": prompt,
        "image": init_image,
        "width": width,
        "height": height,
        "mask_image": mask_image,
        "num_inference_steps": ddim_steps,
        "guidance_scale": scale,
        "negative_prompt": n_prompt,
        "generator": generator,
        }
    
    output_image = pipe(**pipe_args_dict).images[0]
    
    if IASAM_DEBUG:
        save_name = hashlib.md5(np.array(output_image).tobytes()).hexdigest()[0:16] + "_" + os.path.basename(model_id) + "_" + str(seed) + ".png"
        output_image.save(save_name)
    
    return output_image

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Inpainting with Segment Anything")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input image", elem_id="input_image", source="upload", type="numpy", interactive=True)
            sam_btn = gr.Button("Run Segment Anything")
            
            model_id = gr.Dropdown(label="Model ID", elem_id="model_id", choices=model_ids, value=model_ids[0])
            prompt = gr.Textbox(label="Prompt")
            n_prompt = gr.Textbox(label="Negative prompt")
            with gr.Accordion("Advanced options", open=False):
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=2147483647,
                    step=1,
                    randomize=True,
                )
            inpaint_btn = gr.Button("Run Inpainting")
            
            out_image = gr.Image(label="Inpainted image", elem_id="out_image", interactive=False).style(height=480)
            
        with gr.Column():
            sam_image = gr.Image(label="Segment Anything image", elem_id="sam_image", type="numpy", tool="sketch", brush_radius=4)
            select_btn = gr.Button("Determine mask")
            
            sel_mask = gr.Image(label="Selected mask image", elem_id="sel_mask", type="numpy", interactive=False).style(height=480)
            
        sam_btn.click(run_sam, inputs=[input_image], outputs=[sam_image])
        select_btn.click(select_mask, inputs=[sam_image], outputs=[sel_mask])
        inpaint_btn.click(run_inpaint, inputs=[input_image, sel_mask, prompt, n_prompt, ddim_steps, scale, seed, model_id],
                            outputs=[out_image])

block.launch()
