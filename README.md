# Stable Diffusion Tkinter GUI

This is a quickly hacked-together Tkinter-based GUI for running Stable Diffusion in Windows with an AMD GPU.

You must first set up a Python virtual environment, install the dependencies, and get/convert the Stable Diffusion Model to Onnx format.

Stable Diffusion Windows AMD Guides:
- https://www.travelneil.com/stable-diffusion-windows-amd.html
- https://gist.github.com/harishanand95/75f4515e6187a6aa3261af6ac6f61269#file-stable_diffusion-md

Once you have a working Stable Diffusion setup (confirmed with the basic test scripts from the guides), you should be able to use this GUI.

You'll need to copy/rename **text2img_ui.cfg.template** to **text2img_ui.cfg**, and set the **output_path** value to your desired image save location.

## Advantages/features of this GUI:
- Generate more than one image in a row, without having to re-initialize the pipline each time
- Select between the three known working Schedulers
- It saves a .txt file with each image, containing all the settings used to generate the image (Prompt, Seed, Inference Steps, Guidance Scale, Scheduler) for future reference
