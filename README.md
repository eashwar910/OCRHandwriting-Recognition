
# OCR Based Handwriting Recognition System

> ⚠️ **Before you clone the repo**, please download all necessary files from the following Dropbox link and add them to your working directory.
> This way, you can directly run `paint.py` and test the model without needing to train it from scratch.
>
> 📥 **[Download the files here](#https://www.dropbox.com/scl/fo/pjx7a5ui72si8n8l3l4lz/AMuaRHfzwYd2hftoOmX_SLk?rlkey=cdh3lmaap3b2j59vchjcfxzi4&st=6yu7ex8f&dl=0)**

---

This is a simple handwriting recognition system based on OCR (Optical Character Recognition). It uses a CRNN (Convolutional Recurrent Neural Network) architecture built with TensorFlow/Keras to classify drawn characters and words from images.

---

## Features

*  **Trainable CRNN model with full source code**
*  **Data preprocessing** using grayscale normalization and (optional) augmentation
*  **Prediction and inference pipeline** for any input image
*  **Built-in character set** for all digits and alphabets
*  **Label encoding/decoding** for character classification

---

## 🔠 Supported Characters

* All digits: `0-9`
* Uppercase alphabets: `A-Z`
* Lowercase alphabets: `a-z`

---

## ▶️ How to Run

### Option 1: Train your own model from scratch

1. **Install dependencies**
   Create a virtual environment (optional) and install:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the notebook**
   Open `OCR_Model.ipynb` in Jupyter or VS Code and execute all the cells in order to:

   * Preprocess the data
   * Train the model
   * Predict and evaluate results

---

### Option 2: Skip training and use pre-trained model

1. **Clone the repository**:

   ```bash
   git clone https://github.com/eashwar910/Handwriting-Recognition.git
   cd Handwriting-Recognition
   ```

2. **Add the pre-downloaded files from Dropbox** to the cloned folder, including:

   * `model.h5`
   * `dataset/`
   * Any other assets needed for `paint.py`

3. **Run the paint app (drawing UI)**:

   ```bash
   python paint.py
   ```

   This will open a basic drawing canvas. Draw a character, save it, and run the model to predict the character.

---
