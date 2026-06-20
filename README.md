Work from my Coventry Coursework put on my main Github
# ANN-6009CMD
# Social Bias Detection in Social Media Text

**Module:** 6009CMD - Artificial Neural Networks  
**Student:** Muhammad Shayan | 14678222  

## About

Multi-output CNN models for detecting social bias in social media posts using the SBIC dataset (144,649 annotated posts). The models predict three labels simultaneously: offensiveness, intent, and sexual content.

- **Part A** Human-guided model (single Conv1D, shared encoder, loss weighting)
- **Part B** GenAI-assisted model (TextCNN with parallel kernels, focal loss, per-head dense layers)

## Requirements

- Python 3.12
- Jupyter Notebook

### Dependencies

```
tensorflow-cpu==2.16.1
scikit-learn>=1.4
numpy>=1.26
pandas>=2.1
matplotlib>=3.8
seaborn>=0.13
liac-arff>=2.5
```

Install all at once:

```bash
!pip install tensorflow-cpu scikit-learn numpy pandas matplotlib seaborn liac-arff
```

## How to Run

1. Clone the repository
2. Install dependencies (see above)
3. Open `ANN part A.ipynb` and run all cells - this does the full EDA, builds and trains Part A, and saves the model
4. Open `ANN part B.ipynb` and run all cells - this loads the same data, builds and trains Part B, and saves the model
5. Both notebooks save trained models to the `models/` folder
6. Might need to place the Dataset in the same folder as the ipynb as that is how it was when i was running it originally

**Note:** The notebooks use `tensorflow-cpu` and run fine without a GPU. Part A trains in ~3 minutes, Part B in ~5 minutes, could take less or more time

## Dataset

Social Bias Inference Corpus (SBIC) - 144,649 social media posts with annotations for offensiveness, intent, and sexual content.

Source: [OpenML Dataset 46823](https://www.openml.org/search?type=data&id=46823)  
Paper: Sap(2020) [Social Bias Frames](https://aclanthology.org/2020.acl-main.486/)
