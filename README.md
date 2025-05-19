# YOLOv8 + CBAM Integration

This repository demonstrates how to integrate **CBAM** (Convolutional Block Attention Module) into Ultralytics’ YOLOv8 instance-segmentation model. It logs which layers are wrapped with CBAM both at integration time and during each forward pass. You can also explore the fully annotated Jupyter notebook in `yolocbam.ipynb`.

## 📖 Project Overview

- **Goal**  
  Enhance YOLOv8’s `C2f` blocks with channel and spatial attention to help the model focus on more informative features.

- **How It Works**  
  1. Recursively scan the YOLOv8 model for every `C2f` block.  
  2. For each `C2f`, read its output channel count and create a matching CBAM module.  
  3. Replace the original `C2f` with an `nn.Sequential(C2f, CBAM)` wrapper.  
  4. Print log messages:  
     - At integration:  
       ```  
       ✅ CBAM added: C2f @ <module_path>, channels=<N>  
       ```  
     - At each forward call during training:  
       ```  
       ✅ CBAM forward triggered  
       ```

## 🚀 Key Features

- **Automatic Detection**  
  All `C2f` blocks are located and wrapped with CBAM without manual layer indexing.

- **Real-Time Logging**  
  Clearly see which layers receive attention modules and when they execute.

- **Plug-and-Play**  
  Easily drop into any YOLOv8-segmentation training pipeline.

- **Notebook Included**  
  A step-by-step Jupyter notebook `yolocbam.ipynb` is provided for interactive exploration and demonstration.

## ⚙️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/<your-username>/yolov8-cbam.git
   cd yolov8-cbam

**Create & activate a virtual environment (optional)**

python3 -m venv venv
source venv/bin/activate

**Install dependencies**
pip install ultralytics torch

