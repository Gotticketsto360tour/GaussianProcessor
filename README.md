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

### Data
Due to the limited memory offered by Streamlit, **the app will break if users exceed the limit of 800 Mb**. To test the functionality of the app, I recommend using the dataset *AirPassengers.csv*, which is included in the repository.