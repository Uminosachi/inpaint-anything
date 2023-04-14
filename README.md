# Inpaint Anything (Inpainting with Segment Anything)

Inpaint Anything performs stable diffusion inpainting on a browser UI using any mask selected from the output of [Segment Anything](https://github.com/facebookresearch/segment-anything).


Using Segment Anything enables users to specify masks by simply pointing to the desired areas, instead of manually filling them in. This can increase the efficiency and accuracy of the mask creation process, leading to potentially higher-quality inpainting results while saving time and effort.

![Explanation image](images/inpaint_anything_explanation_image_1.png)

## Installation

Please follow these steps to install the software:

1. Create a new conda environment with Python:

```bash
conda create -n inpaint python=3.10
conda activate inpaint
```

2. Clone the software repository:

```bash
git clone https://github.com/Uminosachi/inpaint-anything.git
cd inpaint-anything
```

3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

If you are using CUDA 11.7 or CUDA 11.8, please install the following package instead of the previous one:

```bash
pip install -r requirements_cu117.txt
```

```bash
pip install -r requirements_cu118.txt
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
* You can change the seed in the Advanced options.

Inpainting is performed using [diffusers](https://github.com/huggingface/diffusers).

![UI image](images/inpaint_anything_ui_image_1.png)

## Auto-saving images

* The inpainted image will be automatically saved in the current date folder within the `outputs` directory.
* If you would like to save the segmented and masked images as well, please set the environment variable before running.

```bash
export IASAM_DEBUG=1
```

## License

The source code is licensed under the [Apache 2.0 license](LICENSE).