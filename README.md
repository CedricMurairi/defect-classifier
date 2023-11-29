# Image Defect Classifier

## Overview
The Image Defect Classifier is a specialized tool designed for classifying defects in submersible pump impellers. It categorizes impellers into two primary groups: Defective or Non-Defective. This project is particularly useful in manufacturing quality control, where accurate defect detection is crucial.

## Features
- **Image Similarity Assessment**: Utilizes Structural Similarity Index (SSI) for comparing the uploaded image with a reference image to ensure the image is of a submersible pump impeller.
- **Defect Classification**: Classifies the impeller as either defective or non-defective based on the image analysis.
- **User-Friendly Interface**: Built with Streamlit, the application offers an easy-to-use interface for uploading and analyzing images.

## Installation and Usage
To set up and run the Image Defect Classifier, follow these steps:

1. **Clone the Repository**: 
```bash
git clone https://github.com/CedricMurairi/defect-classifier.git
```
2. **Navigate to the Project Directory**: 
```bash
cd defect-classifier
```
3. **Create a Virtual Environment**: 
```bash
python -m venv env
```
4. **Activate the Virtual Environment**: 
```bash
source env/bin/activate
```
5. **Install the Dependencies**: 
```bash
pip install -r requirements.txt
```

6. **Run the Application**: 
```bash
streamlit run app.py
```

## Notes [Important]!

For misclassification, the model tends to misclassify defective images. This issue was partly due to the limited free RAM provided by Google Colab, as mentioned in the model document. More extensive image training could potentially improve accuracy. The new images pushed to GitHub are classified correctly and can be used for testing purposes.



## How It Works
The application uses a pre-trained VGG16 model for feature extraction and a Support Vector Classifier for the final classification. The user uploads an image of the impeller, and the system processes it to determine if it's defective.

## Dataset and Model Training
Details about the dataset and the model training process can be found in the Jupyter Notebook included in the repository.
[View Notebook](https://github.com/CedricMurairi/defect-classifier/blob/master/ManufacturingData.ipynb "Manufacturing Data Notebook")

## Contributing
Contributions to the project are welcome! Just open an issue or submit a pull request.

## License
This project is licensed under the MIT License - So feel free to use it however you like as long as you include the license file.
