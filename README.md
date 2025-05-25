# 🐶🐱 Dog vs Cat Image Classifier

This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow** and **Keras** to classify images as either a **dog** or a **cat**. The model is trained on the **"Dogs vs. Cats"** dataset from Kaggle and deployed via a simple **Streamlit** web app.

---

## 📁 Project Structure

- `DogVsCat.keras`: Pre-trained Keras model file.
- `DogVsCatClassifier.ipynb`: Jupyter Notebook containing code for model training, evaluation, and visualization.
- `app.py`: Streamlit application for image classification.
- `README.md`: Project documentation.

---

## 🧠 Model Overview

The CNN model includes:

- Convolutional layers with **ReLU activation**, **batch normalization**, and **max-pooling**
- Fully connected dense layers with **dropout** to reduce overfitting
- Final dense layer with **sigmoid** activation for binary classification (dog vs. cat)

---

## ⚙️ Setup Instructions

### ✅ Prerequisites

Ensure you have the following installed:

- Python 3.x
- `pip` (Python package manager)
- Kaggle API key (`kaggle.json`) _(only if training from scratch)_

### 📦 Install Dependencies

```bash
pip install tensorflow keras numpy matplotlib pillow streamlit
```

---

## 🚀 Running the Streamlit App

To run the image classifier locally:

### 1. Clone the repository

```bash
git clone https://github.com/BlackJack-14/CatVsDog.git
cd DogVsCat
```

### 2. Ensure the model file is present

Make sure the file `DogVsCat.keras` is in the same directory as `app.py`.

If not, download it from the GitHub repository or the release section.

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

### 4. Upload an image

Once the app launches in your browser:

- Upload an image of a dog or a cat (JPG/PNG).
- The model will predict and show whether it’s a dog or a cat with confidence.

---

## 🧪 Model Performance

- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Accuracy Achieved:** 81.42% (varies based on training duration and batch size)

Training and validation accuracy and loss are visualized during training to monitor overfitting and model performance.

---

## 📦 Dataset

- **Source:** [Dogs vs. Cats - Kaggle Competition](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
- Contains 25,000 labeled images of cats and dogs.

> _Note:_ You need to accept competition rules and use a Kaggle API key if training from scratch.

---

## 🙌 Acknowledgments

- [Kaggle](https://www.kaggle.com) for the dataset
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for the deep learning framework
- [Streamlit](https://streamlit.io/) for the easy-to-use UI

---

Made with ❤️ by **Rudra Gupta**
