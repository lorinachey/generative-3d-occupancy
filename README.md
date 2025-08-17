# Generative 3D Occupancy Mapping using A Diffusion Model

This repository provides a simple demo of how to train a diffusion model for 3D occupancy generation.

## Motivation
The core idea, introduced [1], is to predict and fill in missing areas in a 3D map of the environment. For example, a robot might not be able to see the floor right in front of it, or areas around a corner might be occluded. By using a generative model, we can synthesize a plausible reconstruction of the missing parts of the map. This can significantly improve a robot's navigation and interaction capabilities. The methods have been enhanced and adapted for new platforms in subsequent works [2, 3].

### Examples

![Generative Prediction of a narrow hallway](images/narrow_hallway_process.png)
*Figure 1: The model generating predictions in a cluttered environment, probabilistically merging the predictions into the running occupancy map, and then traversing that area.*

![Spot robot turning a corner and filling in the map](images/corner_diff.png)
*Figure 2: This image shows a real-world example of how this technology can be used with a quadruped robot to fill in unobserved areas around a corner. The green voxels are the direct observation voxels generated from the LiDAR sensor on the top of the Spot. The red voxels are predicted to be occupied by the 3D occupancy model and fill the area that Spot could not detect. This data was collected on a real Spot robot but we show the approximate location with the Spot image in this photo.*

## Getting Started

Follow these instructions to set up your environment and run the training notebook.

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (for training)
- `conda` or `venv` for environment management.

### Environment Setup

You can use either Conda or a standard Python virtual environment.

**Option 1: Using Conda (My Preference)**

```bash
# Create and activate the conda environment from the environment.yml file
conda env create -f environment.yml
conda activate generative-occupancy
```

**Option 2: Using venv**

```bash
# Create a new virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Once you have set up the environment and installed the dependencies, you can run the training process.

This notebook can be run using the classic Jupyter Notebook interface in your browser or directly within Visual Studio Code.

### Option 1: Running in Jupyter Notebook
1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  **Open the Notebook:** In the Jupyter interface, navigate to and open `training-2DUnet.ipynb`.
3.  **Run the Cells:** You can run all the cells in the notebook to:
    - Load the training data.
    - Define and train the 2D UNet diffusion model.
    - Save the trained model to `saved_models/unet2d_model.pth`.
    - Load the model and generate a new 3D occupancy map sample.

### Option 2: Running in Visual Studio Code

1.  **Open the project folder** in Visual Studio Code.
2.  **Select the Python Interpreter:** Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`) and type `Python: Select Interpreter`. Choose the conda or venv environment you created earlier (e.g., `generative-occupancy` or `.venv`).
3.  **Open the Notebook File:** Open `training-2DUnet.ipynb` in the editor.
4.  **Run the Cells:** You can now run the cells individually or all at once using the icons in the notebook toolbar.

## Inspiration & Cited Works
If this work inspires yours, please consider citing the relevant papers.

[1]
```
@INPROCEEDINGS{10802589,
  author={Reed, Alec and Crowe, Brendan and Albin, Doncey and Achey, Lorin and Hayes, Bradley and Heckman, Christoffer},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={SceneSense: Diffusion Models for 3D Occupancy Synthesis from Partial Observation}, 
  year={2024},
  pages={7383-7390},
  keywords={Geometry;Three-dimensional displays;Runtime;Source coding;Diffusion models;Cameras;Real-time systems;Planning;Trajectory;Intelligent robots},
  doi={10.1109/IROS58592.2024.10802589}}
```

[2]
```
@article{reed2024online,
  title={Online diffusion-based 3d occupancy prediction at the frontier with probabilistic map reconciliation},
  author={Reed, Alec and Achey, Lorin and Crowe, Brendan and Hayes, Bradley and Heckman, Christoffer},
  journal={arXiv preprint arXiv:2409.10681},
  year={2024}
}
```
[3]
```
@misc{achey2025robustroboticexplorationmapping,
      title={Robust Robotic Exploration and Mapping Using Generative Occupancy Map Synthesis}, 
      author={Lorin Achey and Alec Reed and Brendan Crowe and Bradley Hayes and Christoffer Heckman},
      year={2025},
      eprint={2506.20049},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.20049}, 
}
```
