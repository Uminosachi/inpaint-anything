# Inpaint Anything (Inpainting with Segment Anything)

Inpaint Anything performs stable diffusion inpainting on a browser UI using any mask selected from the output of [Segment Anything](https://github.com/facebookresearch/segment-anything).

![Explanation image](images/inpaint_anything_explanation_image_1.png)

## Installation

```bash
conda create -n inpaint python=3.10
conda activate inpaint
git clone https://github.com/Uminosachi/inpaint-anything.git
cd inpaint-anything
pip install -r requirements.txt
```

## Download model

* Download `sam_vit_h_4b8939.pth` from the [model checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints).
* Place the downloaded file in the same directory as the `iasam_app.py` file.
* If the file does not exist, it will be automatically downloaded after running the application.

## Running the application

```bash
python iasam_app.py
```

* Open http://127.0.0.1:7860/ in your browser.
* Recommended browsers: Microsoft Edge or Mozilla Firefox (as mask selection may not work properly with Google Chrome).

## Usage

* Drag and drop your image onto the input image area.
* Click the "Run Segment Anything" button.
* Use sketching to define the area you want to inpaint. You can undo and adjust the pen size.
* Click the "Create mask" button (the mask will appear in the selected mask image area).
* Choose the Model ID, enter the Prompt and Negative Prompt.
* Click the "Run Inpainting" button (**Please note that it may take some time to download the model for the first time**).

Inpainting is performed using [diffusers](https://github.com/huggingface/diffusers).

![UI image](images/inpaint_anything_ui_image_1.png)

## Auto-saving images

* Set the environment variable before running.
* Images will be saved in the current directory.

```bash
export IASAM_DEBUG=1
```

## License

The source code is licensed under the [Apache 2.0 license](LICENSE).