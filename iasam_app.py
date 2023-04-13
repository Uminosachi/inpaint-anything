import os
import torch
import numpy as np
from PIL import Image
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
#import hashlib
from get_dataset_colormap import create_pascal_label_colormap
from torch.hub import download_url_to_file
from torchvision import transforms
from datetime import datetime

IASAM_DEBUG = bool(int(os.environ.get("IASAM_DEBUG", "0")))

device = "cuda" if torch.cuda.is_available() else "cpu"

sam_checkpoint = os.path.join(os.path.dirname(__file__), "sam_vit_h_4b8939.pth")

url_sam_vit_h_4b8939 = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
if not os.path.isfile(sam_checkpoint):
    download_url_to_file(url_sam_vit_h_4b8939, sam_checkpoint)

model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

output_dir = os.path.join(os.path.dirname(__file__), "outputs", datetime.now().strftime("%Y-%m-%d"))

masks = None

cm_pascal = create_pascal_label_colormap()
colormap = cm_pascal
colormap = [c for c in colormap if max(c) >= 64]
#print(len(colormap))

model_ids = [
    "stabilityai/stable-diffusion-2-inpainting",
    "runwayml/stable-diffusion-inpainting",
    "saik0s/realistic_vision_inpainting",
    "parlance/dreamlike-diffusion-1.0-inpainting",
    ]

def run_sam(input_image):
    if input_image is None:
        return None
    print("input_image:", input_image.shape, input_image.dtype)
    
    global masks
    masks = mask_generator.generate(input_image)

    canvas_image = np.zeros_like(input_image)

    masks = sorted(masks, key=lambda x: np.sum(x.get("segmentation").astype(int)))
    masks = masks[:len(colormap)]
    for idx, seg_dict in enumerate(masks):
        seg_mask = np.expand_dims(seg_dict.get("segmentation").astype(int), axis=-1)
        canvas_mask = np.logical_not(np.sum(canvas_image, axis=-1, keepdims=True).astype(bool)).astype(int)
        seg_color = colormap[idx] * seg_mask * canvas_mask
        canvas_image = canvas_image + seg_color
    seg_image = canvas_image.astype(np.uint8)
    
    if IASAM_DEBUG:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        save_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + os.path.splitext(os.path.basename(sam_checkpoint))[0] + ".png"
        save_name = os.path.join(output_dir, save_name)
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
        seg_color = colormap[idx] * seg_mask * canvas_mask        
        canvas_image = canvas_image + seg_color
    
    canvas_mask = np.logical_not(np.sum(canvas_image, axis=-1, keepdims=True).astype(bool)).astype(int)
    if (canvas_mask * mask).astype(bool).any():
        mask_region = mask_region + (canvas_mask * 255)
    
    seg_image = mask_region.astype(np.uint8)

    # if IASAM_DEBUG:
    #     if not os.path.isdir(output_dir):
    #         os.makedirs(output_dir, exist_ok=True)
    #     save_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + "created_mask" + ".png"
    #     save_name = os.path.join(output_dir, save_name)
    #     Image.fromarray(seg_image).save(save_name)

    return seg_image

def run_inpaint(input_image, sel_mask, prompt, n_prompt, ddim_steps, scale, seed, model_id):
    if input_image is None or sel_mask is None:
        return None

    sel_mask_image = sel_mask["image"]
    sel_mask_mask = np.logical_not(sel_mask["mask"][:,:,0:3].astype(bool)).astype(np.uint8)
    sel_mask = sel_mask_image * sel_mask_mask

    if IASAM_DEBUG:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        save_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + "created_mask" + ".png"
        save_name = os.path.join(output_dir, save_name)
        Image.fromarray(sel_mask).save(save_name)

    print(model_id)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.safety_checker = None

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    #pipe = pipe.to(device)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    
    mask_image = sel_mask
        
    init_image = Image.fromarray(input_image).convert("RGB")
    mask_image = Image.fromarray(mask_image).convert("RGB")
    assert init_image.size == mask_image.size
    #print(init_image.size, mask_image.size)
    width, height = init_image.size

    new_height = (height // 8) * 8
    new_width = (width // 8) * 8
    if new_width < width or new_height < height:
        if (new_width / width) < (new_height / height):
            scale = new_height / height
        else:
            scale = new_width / width
        print("resize:", f"({height}, {width})", "->", f"({int(height*scale+0.5)}, {int(width*scale+0.5)})")
        init_image = transforms.functional.resize(init_image, (int(height*scale+0.5), int(width*scale+0.5)), transforms.InterpolationMode.LANCZOS)
        mask_image = transforms.functional.resize(mask_image, (int(height*scale+0.5), int(width*scale+0.5)), transforms.InterpolationMode.LANCZOS)
        print("center_crop:", f"({int(height*scale)}, {int(width*scale)})", "->", f"({new_height}, {new_width})")
        init_image = transforms.functional.center_crop(init_image, (new_height, new_width))
        mask_image = transforms.functional.center_crop(mask_image, (new_height, new_width))
        assert init_image.size == mask_image.size
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
    
    if True:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        save_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + os.path.basename(model_id) + "_" + str(seed) + ".png"
        save_name = os.path.join(output_dir, save_name)
        output_image.save(save_name)
    
    return output_image

block = gr.Blocks().queue()
block.title = "Inpaint Anything"
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
            sam_image = gr.Image(label="Segment Anything image", elem_id="sam_image", type="numpy", tool="sketch", brush_radius=8).style(height=480)
            select_btn = gr.Button("Create mask")
            
            sel_mask = gr.Image(label="Selected mask image", elem_id="sel_mask", type="numpy", tool="sketch", brush_radius=16).style(height=480)
            
        sam_btn.click(run_sam, inputs=[input_image], outputs=[sam_image])
        select_btn.click(select_mask, inputs=[sam_image], outputs=[sel_mask])
        inpaint_btn.click(run_inpaint, inputs=[input_image, sel_mask, prompt, n_prompt, ddim_steps, scale, seed, model_id],
                            outputs=[out_image])

block.launch()
