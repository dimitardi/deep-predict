# Convolutional Neural Network for damage detection

## Motivation 

Implementing the approach proposed by the research article 
[Deep Learning Enabled Fault Diagnosis Using Time-Frequency Image Analysis of Rolling Element Bearings](https://www.hindawi.com/journals/sv/2017/5067651/) 
in a tool for predictive maintenance using deep neural network.

## Setup environment

We use Python with a standard `virtualenv` setup. 
The file `requirements.txt` contains all the dependencies needed to run the code.

For the Jupyter Notebooks we recommend running them on [Google Colaboratory](https://colab.research.google.com), utilizing GPU acceleration.
Make sure the Notebook runtime is GPU accelerated by clicking "Runtime" > "Change runtime type" from the top level menu.

Alternatively one can use any Jupyter Environment. 
In this case the first cells in the Notebooks must be ignored/adjusted, as they are specific to the Colab environment. 

## See it in action on Google Colaboration

Just follow the Jupyter Notebooks, starting with `colab/deeppredict_0_download_raw_data.ipynb`.
Make sure you have a writable `DEEPPREDICT_HOME` folder with the same contents and structure as this repository.

