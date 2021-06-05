# GaussianProcessor
### Interative guide and tool for Gaussian Processes in Python

[Click here to launch the app](https://share.streamlit.io/gotticketsto360tour/gaussianprocessor/main/GaussianProcessor.py)

**NOTE:** The app relies on Github and Streamlit for hosting. The current limit for RAM is 800 Mb on the app, and will therefore not support larger files for uploading. 

The backend is however ready to handle larger files. For larger datasets, clone this repository and install the requirements for running the app:

```
pip install -r requirements.txt
```

After this, you should be able to navigate to *Gaussian_Processor.py* and run the following script in the terminal:

```
streamlit run Gaussian_Processor.py
```

which will open the app. 

If you do not have a good data-set available, the dataset *AirPassengers.csv* is included here, which is ideal for modelling. 