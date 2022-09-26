# Stable Diffusion Tkinter GUI

This is a quickly hacked-together Tkinter-based GUI for running Stable Diffusion in Windows with an AMD GPU.

You must first set up a Python virtual environment, install the dependencies, and get/convert the Stable Diffusion Model to Onnx format.

Stable Diffusion Windows AMD Guides:
- https://www.travelneil.com/stable-diffusion-windows-amd.html
- https://gist.github.com/harishanand95/75f4515e6187a6aa3261af6ac6f61269#file-stable_diffusion-md

Next, there is a small tweak you need to make to `virtualenv\Lib\site-packages\diffusers\pipelines\stable-diffusion\pipeline_stable_diffusion_onny.py`

Change line 133 from:
```
sample=latent_model_input, timestep=np.array([t]), encoder_hidden_states=text_embeddings
```
to
```
sample=latent_model_input, timestep=np.array([t], dtype=np.int64), encoder_hidden_states=text_embeddings
```

Reference: https://www.travelneil.com/stable-diffusion-updates.html

Finally, you need to install scipy (to use the LMSDiscreteScheduler):

```
pip install scipy
```

## Configuring/Using the GUI

Once you have a working Stable Diffusion setup (confirmed with the basic test scripts from the guides), you should be able to use this GUI.

Copy/rename `text2img_ui.cfg.template` to `text2img_ui.cfg`, and set the **output_path** value to your desired image save location.

After that, run `python text2img_ui.py` (make sure you do so from the venv you set up previously)

## Advantages/features of this GUI:
- Generate more than one image in a row, without having to re-initialize the pipline each time
- Select between the three known working Schedulers
- It saves a .txt file with each image, containing all the settings used to generate the image (Prompt, Seed, Inference Steps, Guidance Scale, Scheduler) for future reference
