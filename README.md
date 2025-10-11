Implementation of CI/CD pipeline is coming soon!

The project is developed in VSCode.

### Steps to run the workflow:

1. Clone the repository: 
```bash
https://github.com/vasusriram6/Semiconductor-Wafer-Defect-Classification/tree/master 
```

2. If it doesn't exist beforehand, create and activate a base environment with python version 3.13.5:

```bash
conda create -n py313 python==3.13.5
conda activate py313
```
3. Create and activate the project virtual environment:
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```
4. Once the virtual environment is activated, install the dependencies:
```bash
pip install -r requirements.txt
```
Optional installation for GPU acceleration (with CUDA 12.6 installed on your device):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
Note: You can adjust configurations for training and evaluation in configs/config.yaml

5. In order to train a model (EfficientNet-B0), run:
```bash
python -m src.train
```
Note: If you want to train a model from the beginning, make sure the "outputs/pkl_exp1/" directory is empty 

6. For model evaluation on the testing dataset, run:
```bash
python -m src.eval
```
7. To run the application:
```bash
streamlit run src/app.py
```
This will open the app in a web browser. You can drag and drop wafer images and get the output results. Some sample wafer images are available in “data/extracted_samples” directory. These samples are extracted from data/wafer_test.pkl

### Additional information:
1. The raw dataset is in archive.zip. It was extracted, put in "data/" directory and renamed as "wafer_maps.pkl"
2. "wafer_train.pkl" and "wafer_test.pkl" were extracted based on split label "Testing" and "Training" in the raw dataset. The reason for the swap is because there were more samples with "Testing" label than "Training" label.
3. The model evaluation metrics are saved in "outputs/pkl_exp1/classification_report.csv"
