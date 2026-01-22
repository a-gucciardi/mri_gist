# Project README

## Folder Structure

```
project-root/
│
├── code/                  # Root Folder for experiments
├── code/3d                # 3d learning notebooks, from MONAI, more to come
├── code/siamese_torch     # first paper data load and models on 2D images
├── code/synthseg          # Module from SynthSeg for brain generation
├── code/vol_vis           # Three.js visualizer, lots of tests
├── code/volumes           # Latest tests on 3D dataset, fractal dimension and segmentation
├── requirements.txt       # List of Python dependencies
└── README.md              # This file
```

General structure : all code subfolders contain python script, notebooks, and other useful files.  

## Tasks
 - [x] General repo 
 - [ ] Read me for each part
 - [ ] List of stuff to install (envs, freesurfer, path, etc)
 - [ ] Share data 
 - [ ] Clean

1. **Data Preparation**:
   - Clean and preprocess the data in the `data/` folder.
   - *Currently done but on the server*
   - Data, data registered, data skull-stripped, data segmented, (data super-resolution)

2. **Data Statistics**:
   - Using the results from segmentation
   - List volumes and FD (to compute)

3. **Data Augmentation**:
   - Data augmentation generation
   - Model architectures test, SOTA implementation and runs
   - *dummy* dataset : half brain vs blobs
   - Data splits for training and tesing

4. **Exploration and Analysis**:
   - Test of models architecture
   - More data exploration and model evaluation

5. **Testing**:
   - Write unit tests in a new folder to ensure complete code reliability

6. **Documentation**:
   - Update the READMEs and add comments to the code/notebooks.

---

## Setup Instructions

### 1. Create a Conda Environment with Python 3.11

1. Open a terminal or command prompt.
2. Run the following command to create a new Conda environment:

   ```bash
   conda create -n myenv python=3.11
   ```

3. Activate the environment:

   ```bash
   conda activate myenv
   ```

### 2. Install Dependencies

1. Ensure you are in the `project-root` directory where the `requirements.txt` file is located.
2. Install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

### 3. Verify Installation

1. Check that all dependencies are installed correctly by running:

   ```bash
   pip list
   ```

2. Ensure the environment is ready for development or execution of the project.

---

## Cool tutorials and links
 - [Bobs dataset](https://bobsrepository.readthedocs.io/en/latest/) 
 - [Nibabel](https://nipy.org/nibabel/gettingstarted.html)
 - [MONAI](https://github.com/Project-MONAI/tutorials) ( will be added to 3d, some cool stuff already done on the server)

## Additional Notes

- Replace `myenv` with your preferred environment name.
- If you encounter any issues during installation, ensure your Conda and pip are up to date.
- For GPU support, install the appropriate versions of libraries like `tensorflow` or `pytorch` as needed. ( to be tested )