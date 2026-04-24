# Interactive Data Visualization with Streamlit

### A hands-on tutorial using the Fashion-MNIST dataset

This repository contains a step-by-step tutorial on building **interactive data visualizations** using [Streamlit](https://streamlit.io/), a Python library that turns scripts into shareable web apps with minimal overhead.

Most Streamlit tutorials either:
- Teach widgets in isolation, or
- Show a finished dashboard without explaining the *design decisions* behind it

This tutorial bridges the gap between the two. It is aimed at students and practitioners who already know some Python and want to build interactive, shareable visualizations without learning a full web framework. 

I was planning on creating this tutorial using the Palmer Penguins, but after the pre-class that used its own Streamlit dashboard for the dataset, I decided to pivot to Fashion-MNIST.

## How to Run Locally

Please make sure you have Miniconda or Anaconda installed! The program is also best run using Python 3.10.

### 1. Clone the repository
```bash
git clone https://github.com/allison-diebol/streamlit-data-viz-tutorial
cd streamlit-data-viz-tutorial
```

### 2. Create & Activate the conda environment
```bash
conda env create -f environment.yml
conda activate fashion-mnist-dashboard
```
The tutorial uses TensorFlow/Keras to gain access to the dataset. There are several specific dependencies that the environment file resolves! It may take a little bit for the new environment to be created.

### 3. Run the app
```bash
streamlit run tutorial.py
```

The app will open automatically in a local URL (usually `http://localhost:8501`)

## Repository Files
```
├── tutorial.py                  # main Streamlit file
├── environment.yml              # conda environment spec
├── tutorial_subset.pdf          # PDFs of part of the tutorial in case you are unable to locally run it
├── writeup.pdf                  # written report
└── README.md                    # you are here!
```

## Acknowledgements
Dataset: Fashion-MNIST by Zalando Research, released under MIT licence.

## Author
Built as an Honors Option project for CMSE 402: Data Visualization Principles and Techniques

**Allison Diebol** · diebolal@msu.edu · github.com/allison-diebol
