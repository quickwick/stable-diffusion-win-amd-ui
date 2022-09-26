"""
Original Source/Inspiration:
https://www.travelneil.com/stable-diffusion-windows-amd.html
https://www.travelneil.com/stable-diffusion-updates.html

Don't forget to use correct virtual environment!
"""

#import os
from pathlib import Path
import tkinter as tk # GUI
import configparser
from datetime import datetime
from PIL import ImageTk # must install Pillow module
from diffusers import StableDiffusionOnnxPipeline, DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
import numpy as np

def get_latents_from_seed(passed_seed: int, width: int, height:int) -> np.ndarray:
    # 1 is batch size
    latents_shape = (1, 4, height // 8, width // 8)
    # Gotta use numpy instead of torch, because torch's randn() doesn't support DML
    rng = np.random.default_rng(passed_seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents

def change_scheduler(*args):
    print (strvar_scheduler.get())
    if strvar_scheduler.get() == 'DDIM':
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, num_train_timesteps=1000, tensor_format="np")
        entry_steps.delete(0, 'end')
        entry_steps.insert(0, '8')
    elif strvar_scheduler.get() == 'LMSDiscrete':
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, tensor_format="np")
        entry_steps.delete(0, 'end')
        entry_steps.insert(0, '25')
    elif strvar_scheduler.get() == 'PNDM':
        scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True, num_train_timesteps=1000, tensor_format="np")
        entry_steps.delete(0, 'end')
        entry_steps.insert(0, '25')

    global pipe
    canvas.delete("all")
    try:
        canvas.create_text(250, 30, text="Attempting to delete existing pipeline...", fill="black")
        canvas.update()
        del pipe
        canvas.create_text(250, 45, text="Pipeline deleted", fill="black")
        canvas.update()
    except NameError:
        canvas.create_text(250, 45, text="No pipeline exists yet", fill="black")
        canvas.update()
    finally:
        canvas.create_text(250, 60, text="Ok to create new pipeline now", fill="black")
        canvas.update()
    
    canvas.create_text(250, 75, text="Generating Stable Diffusion Onnx Pipeline...", fill="black")
    canvas.update()
    pipe = StableDiffusionOnnxPipeline.from_pretrained("./stable_diffusion_onnx", provider="DmlExecutionProvider", scheduler=scheduler)
    # Remove the safety_checker (NSFW filter) which speeds things up a lot
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    # Show the Generate button
    btn_generate.pack()
    canvas.create_text(250, 90, text="Pipeline created, go for Generate", fill="black")
    canvas.update()

def cmd_randomize():
    entry_seed.delete(0, 'end')
    entry_seed.insert(0,str(np.random.randint(1,9223372036854775807, dtype=np.int64)))

def cmd_generate():
    prompt = entry_prompt.get()
    seed = int(entry_seed.get())
    cfg = float(entry_cfg.get())
    steps = int(entry_steps.get())

    latents = get_latents_from_seed(seed, 512, 512)

    # Determine output filenames
    dt_obj = datetime.now()
    dt_cust = dt_obj.strftime("%Y-%m-%d_%H-%M-%S")
    image_name = dt_cust + "_" + str(seed) + ".png"
    text_name = dt_cust + "_" + str(seed) + "_info.txt"
    image_path = output_base_path / image_name
    text_path = output_base_path / text_name
    
    """
    Reference: Arguments taken by pipe
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_onnx.py#L45

    prompt: Union[str, List[str]],
    height: Optional[int] = 512, # Must be divisible by 8
    width: Optional[int] = 512, # Must be divisible by 8
    num_inference_steps: Optional[int] = 50, # The number of iterations to perform. Generally higher means better quality
    guidance_scale: Optional[float] = 7.5, # This is also sometimes called the CFG value.
        How heavily the AI will weight prompt, versus being creative.
        CFG 2 - 6: Let the AI take the wheel.
        CFG 7 - 11: Let's collaborate, AI!
        CFG 12 - 15: No, seriously, this is a good prompt. Just do what I say, AI.
        CFG 16 - 20: DO WHAT I SAY OR ELSE, AI.
    eta: Optional[float] = 0.0,
    latents: Optional[np.ndarray] = None,
    output_type: Optional[str] = "pil",
    """

    canvas.delete("all")
    canvas.create_text(250, 50, text="Generating image...", fill="black")
    canvas.update()

    # Generate and Save Image
    image = pipe(prompt, num_inference_steps=steps, guidance_scale=cfg, latents=latents).images[0]
    image.save(image_path)

    # Save Info to Text File
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write("Prompt: " + prompt + "\n")
        f.write("Seed: " + str(seed) + "\n")
        f.write("Inference Steps: " + str(steps) + "\n")
        f.write("Guidance Scale (CFG): " + str(cfg) + "\n")
        f.write("Scheduler: " + strvar_scheduler.get())

    # Display image in GUI
    global img_for_canvas # avoid garbage collection bug(?)
    img_for_canvas = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.create_image(0, 0, anchor='nw', image=img_for_canvas)
    canvas.update()

# Get config setting(s)
text2img_config = configparser.ConfigParser()
text2img_config_path = Path(__file__).resolve().parent / 'text2img_ui.cfg'
text2img_config.read(text2img_config_path)
output_base_path = Path(text2img_config['output']['output_path'])

# Create GUI
window = tk.Tk()
window.geometry("520x600")

frame_labels = tk.Frame(relief="groove",borderwidth=2)
frame_entries = tk.Frame(relief="groove",borderwidth=2)
frame_buttons = tk.Frame(relief="groove",borderwidth=2)
frame_image = tk.Frame(relief="groove",borderwidth=2)

label_prompt = tk.Label(master=frame_labels, text="Prompt:")
label_prompt.pack()
label_seed = tk.Label(master=frame_labels, text="Seed:")
label_seed.pack()
label_cfg = tk.Label(master=frame_labels, text="CFG:")
label_cfg.pack()
label_steps = tk.Label(master=frame_labels, text="Inference steps:")
label_steps.pack()

entry_prompt = tk.Entry(master=frame_entries)
entry_prompt.insert(0,"")
entry_prompt.pack(expand=True,fill=tk.X)
entry_seed = tk.Entry(master=frame_entries)
entry_seed.insert(0,"33055")
entry_seed.pack(expand=True,fill=tk.X)
entry_cfg = tk.Entry(master=frame_entries)
entry_cfg.insert(0,"7.5")
entry_cfg.pack(expand=True,fill=tk.X)
entry_steps = tk.Entry(master=frame_entries)
entry_steps.insert(0,"8")
entry_steps.pack(expand=True,fill=tk.X)

btn_generate = tk.Button(master=frame_buttons, text="Generate", command=cmd_generate)
#btn_generate.pack()
btn_randomize = tk.Button(master=frame_buttons, text="Random Seed", command=cmd_randomize)
btn_randomize.pack()
schedulers = {'DDIM','LMSDiscrete','PNDM'}
strvar_scheduler = tk.StringVar()
strvar_scheduler.set('Scheduler')
optmnu_scheduler = tk.OptionMenu(frame_buttons, strvar_scheduler, *schedulers)
optmnu_scheduler.pack()

# link function to change dropdown
strvar_scheduler.trace('w', change_scheduler)

canvas = tk.Canvas(master=frame_image, width=512, height=512)
canvas.create_text(250, 50, text="Select a Scheduler using the dropdown", fill="black")
canvas.pack()

frame_image.pack(side=tk.BOTTOM)
frame_buttons.pack(side=tk.RIGHT)
frame_entries.pack(side=tk.RIGHT,expand=True,fill=tk.X)
frame_labels.pack(side=tk.RIGHT)

window.winfo_toplevel().title('Stable Diffusion on Windows with AMD - Tkinter GUI')

window.mainloop()
