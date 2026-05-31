# 🌿 Plant Disease Detection using CNN (Keras + TensorFlow)

This project aims to detect plant diseases using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. A simple and interactive web interface is also provided for easy testing via image uploads.

---

## 🧠 Project Structure

```
├── CNN/                  # Contains the CNN architecture and training code
├── WEB/                  # Simple web interface for local image testing
├── manual_testing_images/ # Test images for manual validation
├── info/               # Accuracy
├── Dataset/             # Dataset download link file
├── run_web_app.bat       # Windows batch file to start the web interface easily
```

---

## 📁 Dataset

* The dataset contains **24 balanced classes** with **500 images per class**.
* Due to storage constraints, the dataset is **not included** in this repository.

📥 **Download the dataset**:
Navigate to the `data set/` folder and open the `dataset_link.txt` file. It contains a direct download link.
After downloading, place the dataset in the appropriate directory as mentioned in the CNN code.

---

## 🧪 Model Details

* Developed using **TensorFlow** and **Keras**.
* CNN architecture is defined in the `CNN/` directory.
* Includes image augmentation and preprocessing pipelines.
* Designed for **multi-class classification** of plant leaf diseases.

---

## 🌐 Web Interface

* The `web/` folder includes a simple Flask-based interface.
* Users can upload plant leaf images via the browser.
* The system predicts the disease based on the uploaded image.

🚀 **To launch the web app (on Windows)**:
Double-click the `run_web_app.bat` file. This will automatically:

* Set up the environment (if required)
* Start the server
* Open the web interface in your default browser

---

## 🔧 Requirements

Before running, ensure you have the following Python packages installed:

```bash
tensorflow
flask
opencv-python
numpy
Pillow
```

You can install them via:

```bash
pip install -r requirements.txt
```

---

## 📸 Manual Testing

Use the `manual_testing_images/` folder to test the model without using the web interface. This can help in validating predictions quickly using CLI or notebook.

---

## 📝 Notes

* The system works locally and does not require internet once the model and dataset are set up.
* Dataset download is manual due to size limitations.
* The batch file (`run_web_app.bat`) is for **Windows only**.
* The model "plant_disease_final_model.h5" is not available in this repo use this drive link to download it and then simply copy paste it into the WEB folder

---

## 🤝 Contributions

Feel free to fork, improve, or report any issues via GitHub.

---
