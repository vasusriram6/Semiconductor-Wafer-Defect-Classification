
# Semiconductor Wafer Defect Classification

The project is developed in VSCode, with CI/CD implementation using Github Actions and Docker.

### Sample app screenshots:

Idle app:

<img width="800" height="400" alt="App idle" src="https://github.com/user-attachments/assets/a94b25d4-330f-47c7-8d43-2ee40917c01d" />

App output after dragging and dropping wafer images:

<img width="800" height="400" alt="App inference output softmax" src="https://github.com/user-attachments/assets/a3f8f45c-0ec8-4509-b3d2-3c43f0ee8e48" />

Some sample wafer images are available in “data/extracted_samples” directory. These samples are extracted from data/wafer_test.pkl

### Steps to run the core modules of the project on local device:

1. Clone the repository: 
```bash
https://github.com/vasusriram6/Semiconductor-Wafer-Defect-Classification/tree/main 
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
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126; pip install --no-cache-dir -r requirements.txt
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
Note: If you want to train a model from the beginning, make sure the "outputs/main_output/" directory is empty
The existing model was trained on a GPU with 6GB VRAM. For lower-end devices, try reducing batch size in "config.yaml"

6. For model evaluation on the testing dataset, run:
```bash
python -m src.eval
```
7. To run the application:
```bash
streamlit run src/app.py
```
This will open the app in a web browser.

### Steps to run the app directly via Docker container:
1.	Make sure Docker Desktop is installed and is running
2.	Then in command prompt, type:
```bash
docker pull noobmlengineer/streamlit-wafer-app:latest
```
3.	To run the container, use any one of the following methods:
    - Open Docker Desktop window, click on “Images” on left sidebar, then click the run button of the downloaded image. In the optional settings, set the host port to “8501”.
    - Run in command prompt:
```bash
docker run -p 8501:8501 noobmlengineer/streamlit-wafer-app:latest
```
4.	Then in a web browser, open the website:
```bash
localhost:8501
```

### Additional information:
1. The raw dataset is in archive.zip. It was extracted, put in "data/" directory and renamed as "wafer_maps.pkl"
2. "wafer_train.pkl" and "wafer_test.pkl" were extracted based on split label "Testing" and "Training" in the raw dataset. The reason for the swap is because there were more samples with "Testing" label than "Training" label.
3. The model evaluation metrics are saved in "outputs/main_output/classification_report.csv"
