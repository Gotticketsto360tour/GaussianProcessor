## IMPORTS
import warnings
warnings.filterwarnings("ignore") #somewhat questionable practice of ignoring all warnings.
import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import os
import re
import helper_functions
import matplotlib.pyplot as plt
import base64
from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic, ExpSineSquared, DotProduct, RBF

## SETUP STYLES

sns.set_style(
    style='darkgrid', 
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')

st.set_page_config(
    page_title="Gaussian Processor",
    page_icon="bayes_bois.png",
    initial_sidebar_state="expanded",
    )

st.image("bayes_bois.png", width = 100)

## TITLE

'''
# Gaussian Processor
'''

### GLOBAL PARAMS
radio_checker = st.selectbox("What do you want to do?", ["Learn and simulate", "Upload my own data"])

## SIMULATION OPTION:

if radio_checker == "Learn and simulate":

    '''
    # Learning and simulating
    '''
    '''
    Welcome to the Gaussian Processor! To begin simulating, simply adjust the sliders below and press the *Simulate Data*-button. 
    '''
        
    '''
    ## Simulating Data 
    '''
    with st.form("make_data"):
            
        expander = st.beta_expander("Trends")
        with expander:
            lin_trend = st.slider("Linear trend", -20.0, 20.0, 0.4)
            poly_trend = st.slider("Polynomial Trend", -1.0, 1.0, 0.0)

        expander = st.beta_expander("Seasonality")
        with expander: 
            sinus = st.slider("Sinus", 0.0, 10.0, 1.0)
            sinus_2 = st.slider("Sinus 2", 0.0, 10.0, 1.0)
            sinus_2_period = st.slider("Sinus 2 Period", 0.0, 10.0, 0.4)

        expander = st.beta_expander("Noise")

        with expander:
            noise_amount = st.slider("Noise Amount", 1.0, 100.0, 1.0)

    ### GLOBAL PARAMS

        '''
        ## Choose values for train and test values
        '''

        expander = st.beta_expander("Set parameters for X")

        with expander:
            '''
            #### Training parameters
            '''
            n = st.number_input("Select number of points", 20)
            x_end = st.number_input("Select end for X", 10)
            '''
            #### Test parameters
            '''
            after_end = st.number_input("Select end after X", 10)
            test_n = st.number_input("Select number of points in test-set", 20)

        submitted = st.form_submit_button("Simulate Data") 

        if submitted:
            x = np.atleast_2d(np.linspace(0, x_end, 1000)).T
            x_test = np.atleast_2d(np.linspace(x_end, x_end+after_end, 500)).T
            x = np.append(x, x_test)
            fig, ax = plt.subplots(1, 1, figsize=(12, 5)) 
            ax.plot(x, helper_functions.f(x, lin_trend, sinus, sinus_2, sinus_2_period, poly_trend), 'r:', label=rf'$f(x) = {lin_trend} \cdot x + {poly_trend} \cdot x^2 +{sinus} \cdot \sin(x) + {sinus_2} \cdot \sin({sinus_2_period} \cdot x)$')
            ax.axvline(x_end, label = "training test split")
            ax.set(xlabel = "$x$", ylabel = "$f(x)$")
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                fancybox=True, shadow=True, ncol=2)
            st.pyplot(fig)

            ## success button
            st.success("Data has been simulated!")
            st.success("Try exploring different kernels in the sidebar to the left. When you are ready, press the *Fit the Model*-button.")

            '''
            ## Guide to exploring the app
            '''

            expander = st.beta_expander("Click for a guided tour")

            with expander:
                f'''
                ### Data Generation
                This app was made to showcase $sklearn$'s GaussianProcesses - an extremely useful method for time-series forecasting.
                Above you will find sliders, which specify the data generating function. Currently, the data-generating function is

                $$f(x) = {lin_trend} \cdot x + {poly_trend} \cdot x^2 + {sinus} \cdot \sin(x) + {sinus_2} \cdot \sin({sinus_2_period} \cdot x)$$

                Optionally, the data generating function is not perfect. This can be used to simulate measurement errors. This is specified in
                the "Noise Amount"-slider. The data is sampled from a Gaussian distribution around the point ($x$), with a standard deviation equal to 
                the Noise Amount:

                $$x \sim N(f(x), {noise_amount})$$

                ### Train and Test
                The model currently sees {n} points between 0 and {x_end}. The MSE is calculated for the test set. 

                ### Known Issues
                Currently, when parameters in the kernels are specified, for which there is no good solution, the kernel will default to the last value,
                which gave valid responses. If nothing you do changes the plot, please try to reset the kernels and start over. Another alternative fix is to 
                play around with the data by adding additional points or extending the range of the training data. 
                '''

    with st.sidebar.form("my_sidebar_form"):

            ### initialize interpret dict

            interpret_dict = {}
            string_dict = {}

            kernel_select = st.sidebar.multiselect("Select Kernels", ["RationalQuadratic", "ExpSineSquared", "ExpSineSquared_2", "DotProduct", "WhiteKernel", "RBF", "Matern", "DotProduct Squared"], "DotProduct")

            if kernel_select:
                st.sidebar.header("Kernels")

            if "RationalQuadratic" in kernel_select:

                expander = st.sidebar.beta_expander("Rational Quadratic")
                with expander:
                    rational_weight = st.slider("Rational Quadratic Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
                    rational = st.slider("Rational Quadratic Kernel L", min_value = 0.1, max_value = 100.0)
                    alpha = st.slider("Rational Quadratic Alpha", min_value = 0.1, max_value = 100.0)
                
                interpret_dict["RationalQuadratic"] = rational_weight * RationalQuadratic(length_scale = rational, alpha = alpha)
                string_dict["RationalQuadratic"] = '''### RationalQuadratic
The *Rational Quadratic* can be seen as an infinite sum of RBF kernels. If you are trying to capture a non-linear trend, this is a good option.

#### Mathematical equation
$$k(x_i, x_j) = \\left(1 + \\frac{d(x_i, x_j)^2}{2 \\alpha l^2} \\right)$$

where $d(.,.)$ is the Euclidean distance.  
'''

            if "DotProduct" in kernel_select:

                expander = st.sidebar.beta_expander("Dot Product")
                with expander:
                    dot_weight = st.slider("Dot Product Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
                    sigma_dot = st.slider("Sigma for DotProduct", min_value = 0.0, max_value = 100.0)
                
                interpret_dict["DotProduct"] = dot_weight * DotProduct(sigma_0 = sigma_dot)
                string_dict["DotProduct"] = '''### DotProduct 
The *Dot Product* can be viewed as a linear kernel. It can be used to sample from models that capture the linear trend in the data.

#### Mathematical equation
$$k(x_i, x_j) = \sigma ^2 _0 + x_i \cdot x_j$$
'''

            if "DotProduct Squared" in kernel_select:
                expander = st.sidebar.beta_expander("Dot Product Squared")
                with expander:
                    dot_squared_weight = st.slider("Dot Product Squared Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
                    squared_sigma = st.slider("Sigma for DotProduct Squared", min_value = 0.0, max_value = 100.0)

                interpret_dict["DotProduct Squared"] = dot_squared_weight * DotProduct(sigma_0=squared_sigma) * DotProduct(sigma_0=squared_sigma)
                string_dict["DotProduct Squared"] = '''### DotProduct Squared
The *Dot Product Squared* is maybe not surprisingly a polynomial kernel. It can be used to sample from models that capture a quadratic trend in the data.

#### Mathematical equation
$$k(x_i, x_j) = (\sigma ^2 _0 + x_i \cdot x_j)^2$$
'''
                
            if "ExpSineSquared" in kernel_select:
                expander = st.sidebar.beta_expander("Exponential Sine Squared")
                with expander:
                    sine_weight = st.slider("Exp Sine Squared Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
                    exp_sine_1 = st.slider("First Exp Sine Squared L", min_value = 0.1, max_value = 100.0)

                interpret_dict["ExpSineSquared"] = sine_weight * ExpSineSquared(length_scale = exp_sine_1)
                string_dict["ExpSineSquared"] = '''### ExpSineSquared
The *Exponential Sine Squared* kernels can be used to sample models that capture the periodic/cyclical relations in the data. The reason that there are two is to model two different cycles. The second Exponential Sine Squared Kernel can adjusted in terms of its periodicity.

#### Mathematical equation
$$k(x_i, x_j) = \exp \\left(- \\frac{2 \sin(\pi d (x_i, x_j)/p)}{l^2}\\right)$$
'''

            if "ExpSineSquared_2" in kernel_select:
                expander = st.sidebar.beta_expander("Exponential Sine Squared 2")
                with expander:
                    sine_weight_2 = st.slider("Exp Sine Squared Weight 2", min_value = 0.1, max_value = 100.0, value = 1.0)
                    exp_sine_2 = st.slider("Second Exp Sine Squared L", min_value = 0.1, max_value = 100.0)
                    exp_sine_2_period = st.slider("Second Exp Sine Squared Period", min_value = 0.1, max_value = 100.0)
                interpret_dict["ExpSineSquared_2"] = sine_weight_2 * ExpSineSquared(length_scale = exp_sine_2, periodicity= exp_sine_2_period)

            if "RBF" in kernel_select:
                expander = st.sidebar.beta_expander("RBF")
                with expander:
                    rbf_weight = st.slider("RBF Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
                    rbf = st.slider("RBF L", min_value = 0.1, max_value = 100.0)

                interpret_dict["RBF"] = rbf_weight * RBF(length_scale=rbf)
                string_dict["RBF"] = '''### RBF
The *RGF* kernel treats points that are close to each other in $x$ as being close in $y$. Sometimes this is also called the squared exponential kernel.

#### Mathematical equation
$$k(x_i, x_j) = \exp \\left(- \\frac{d(x_i, x_j)^2}{2l^2})\\right)$$
'''

            if "Matern" in kernel_select:
                expander = st.sidebar.beta_expander("Matern")
                with expander:
                    matern_weight = st.slider("Mattern Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
                    matern_length = st.slider("Mattern Length", min_value = 0.1, max_value = 100.0, value = 1.0)
                    matern_nu = st.slider("Mattern Nu", min_value = 0.1, max_value = 10.0, value = 1.5) #NOTE: Parameter changed to max 10, else it can break
                
                interpret_dict["Matern"] = matern_weight * Matern(length_scale=matern_length, nu = matern_nu)
                string_dict["Matern"] = '''### Matern

#### Mathematical equation
$$C_{\\nu }(d)=\\sigma ^{2}{\\frac {2^{1-\\nu }}{\Gamma (\\nu )}}{\Bigg (}{\sqrt {2\\nu }}{\\frac {d}{\\rho }}{\Bigg )}^{\\nu }K_{\\nu }{\Bigg (}{\sqrt {2\\nu }}{\\frac {d}{\\rho }}{\Bigg )}$$'''
            
            if "WhiteKernel" in kernel_select:
                expander = st.sidebar.beta_expander("WhiteKernel")
                with expander:
                    white_weight = st.slider("White Noise Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
                    white_noise = st.slider("NoiseLevel", min_value = 0.1, max_value = 10.0)

                interpret_dict["WhiteKernel"] = white_weight * WhiteKernel(white_noise)
                string_dict["WhiteKernel"] = ''' ### White Kernel
The *White Kernel* is white noise and can be used to model noise (i.e. measurement errors) in the data. Furthermore, it is useful to make the model fit the data.'''

            submitted_sidebar = st.form_submit_button("Fit the Model")

    if submitted_sidebar:

        if kernel_select:
            # ----------------------------------------------------------------------

            # Instantiate a Gaussian Process model

            for i, ele in enumerate(kernel_select):
                element = interpret_dict.get(ele)
                if i == 0:
                    kernel = element
                else:
                    kernel += element


            # ----------------------------------------------------------------------
            X = np.linspace(0.1, x_end, n)
            X = np.atleast_2d(X).T

            x = np.atleast_2d(np.linspace(0, x_end, 1000)).T

            np.random.seed(1)
            
            # Observations and noise
            y = helper_functions.f(X, lin_trend, sinus, sinus_2, sinus_2_period, poly_trend).ravel()
            dy = 0.5 + noise_amount * np.random.random(y.shape)
            noise = np.random.normal(0, dy)
            y += noise

            # Instantiate a Gaussian Process model
            gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2,
                                        n_restarts_optimizer=9)

            # Fit to data using Maximum Likelihood Estimation of the parameters
            with st.spinner("Fitting the model..."):
                gp.fit(X, y)

                # Make the prediction on the meshed x-axis (ask for MSE as well)
                y_pred, sigma = gp.predict(x, return_std=True)

                X_test = np.linspace(x_end, x_end + after_end, test_n)
                X_test = np.atleast_2d(X_test).T

                y_test = helper_functions.f(X_test, lin_trend, sinus, sinus_2, sinus_2_period, poly_trend).ravel()
                dy_test = 0.5 + noise_amount * np.random.random(y_test.shape)
                noise = np.random.normal(0, dy_test)
                y_test += noise

                x_test = np.atleast_2d(np.linspace(x_end, x_end+after_end, 500)).T

                y_pred_test, sigma_test = gp.predict(x_test, return_std=True) #??

                y_predding, sigma_predding = gp.predict(X_test, return_std=True)

                errors = y_test - y_predding
                errors = errors.flatten()
                errors_mean = errors.mean()
                errors_std = errors.std()

            '''
            ### Kernels
            '''
            expander = st.beta_expander("Read about your kernels")

            with expander:
                '''
                ### General introduction to Kernels
                Kernels are the way Gaussian Processes are specified. As a starter intuition, these kernels can be seen as the distribution of functions,
                that the model is sampling from. In other words, the kernels specifies which family of functions the model should think the data has been generated from.
                
                The kernels can be combined by either multiplying or adding them together to create
                more complex kernels. In this app, all kernels are combined by adding them. 
                
                For all kernels, $k_i$, you can specify their weight, $w_i$:

                $$k = \sum_i w_i \cdot k_i$$

                Almost all kernels have an $l$-parameter, which can be understood as how local or global the patterns should be estimated. The larger $l$, the more global the kernel is. Larger $l$ will normally result in more smooth functions.
                
                Different kernels have different optional parameters, for instance $l$ and $w$. These can be tweaked for each kernel in the sidebar to the left.
                
                In the following paragraph, only your choices of kernels will be explained. If you want to explore other kernels, try including them in your model to see what they do and when they are useful.

                '''
                string = ""
                for i in string_dict:
                    string += string_dict.get(i) + "\n"
                string

            ### PLOT KERNELS:
            expander = st.beta_expander("Visualize your kernels")
            with st.spinner("Plotting kernels..."):
                with expander:
                    '''
                    #### Covariance matrices
                    Below are the covariance matrices for your selected kernels. The covariance matrices have $x$ on both axis. Each point on this matrice can be defined as $k(x_i, x_j)$. 

                    A good intuition to have here, is that if $k(x_i, x_j)$ has a bright color, the kernel estimates that these points are similar, and should therefore learn from each other's value.
                    '''
                    st.pyplot(helper_functions.plot_kernels(kernel_select, interpret_dict, kernel, X))

            # Plot the function, the prediction and the 95% confidence interval based on
            # the MSE
            fig, ax = plt.subplots(2, 1, figsize=(12, 10)) 
            ax[0].plot(x, helper_functions.f(x, lin_trend, sinus, sinus_2, sinus_2_period, poly_trend), 'r:', label=rf'$f(x) = {lin_trend} \cdot x + {poly_trend} \cdot x^2 +{sinus} \cdot \sin(x) + {sinus_2} \cdot \sin({sinus_2_period} \cdot x)$')
            ax[0].errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')
            ax[0].plot(x, y_pred, 'b-', label='Prediction')
            ax[0].fill(np.concatenate([x, x[::-1]]),
                    np.concatenate([y_pred - 1.9600 * sigma,
                                    (y_pred + 1.9600 * sigma)[::-1]]),
                    alpha=.5, fc='b', ec='None', label='95% confidence interval')
            ax[0].plot(x_test, helper_functions.f(x_test, lin_trend, sinus, sinus_2, sinus_2_period, poly_trend), 'r:', label='_nolegend_')
            ax[0].errorbar(X_test.ravel(), y_test, dy_test, fmt='r.', markersize=10, label='_nolegend_')
            ax[0].plot(x_test, y_pred_test, 'b-', label='_nolegend_')
            ax[0].fill(np.concatenate([x_test, x_test[::-1]]),
                    np.concatenate([y_pred_test - 1.9600 * sigma_test,
                                    (y_pred_test + 1.9600 * sigma_test)[::-1]]),
                    alpha=.5, fc='b', ec='None', label='_nolegend_')
            #ax[0].xlabel('$x$')
            #ax[0].ylabel('$f(x)$')
            ax[0].axvline(x_end)
            ax[0].set(xlabel = "$x$", ylabel = "$f(x)$")
            ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                fancybox=True, shadow=True, ncol=2)
            ### HISTOGRAM:
            sns.distplot(a=errors, bins = 20, ax=ax[1])
            ax[1].axvline(x=errors_mean, color=sns_c[3], linestyle='--', label=f'$\mu$')
            ax[1].axvline(x=errors_mean + 2*errors_std, color=sns_c[4], linestyle='--', label=f'$\mu \pm 2\sigma$')
            ax[1].axvline(x=errors_mean - 2*errors_std, color=sns_c[4], linestyle='--')
            ax[1].axvline(x=errors_mean, color=sns_c[3], linestyle='--')
            ax[1].legend()
            ax[1].set(title='Distribution of Error', xlabel='error', ylabel=None);
            st.pyplot(fig)


            col1, col2, col3 = st.beta_columns([1,6,1])

            with col1:
                st.write("")

            with col2:
                stringing = str(gp.kernel)
                stringing = stringing.replace("**", "^")
                stringing = stringing.replace("*", "\cdot")
                stringing = stringing.replace("_", " ")

                #stringing = re.sub(r"**", "^", stringing)
                '''### Current settings:'''


                f'''$${stringing}$$'''

                expander = st.beta_expander("Accuracy")
                with expander:
                    '''Here, we are using the mean squared error 
                    to get an accuracy measurement of our model.
                    Mathematically, this is given by:'''
                    r''' ### $$\frac{1}{n} \sum ^{n}_{i=1} (y-\hat{y})^2$$'''

                    f'''
                    ### $$MSE = {helper_functions.get_RMSE(y_test, y_predding)}$$
                    '''

            with col3:
                st.write("")

else:

    '''
    # Upload your own data
    '''

    data_file = st.file_uploader("Upload CSV",type=['csv'])

    if data_file is not None:
        st.success('''File loaded!''')
        #file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
        #st.write(file_details)

        #read the data
        df = pd.read_csv(data_file) 

        expander = st.beta_expander("Show dataframe")

        with expander:
            df

        #form for batching inputs 
        with st.form("my_form"):

            '''
            ### Choose your variables 
            '''
            #select x
            x_select = st.selectbox("What is your x-variable?", df.columns)
            ### NOTE: Can this be made "pop-uppy"?
            #select y
            y_select = st.selectbox("What is your y-variable?", df.columns)

            date_time_checker = st.selectbox("Is your x-axis a date?", ["Yes", "No"])

            if date_time_checker == "Yes":
                df[x_select] = pd.to_datetime(df[x_select])

            df["x"] = list(range(len(df[x_select])))

            def time_split(df, splitter):
                percent = len(df["x"]) / 100
                index = round(percent * splitter)

                train = df[df["x"] <= index]
                test = df[df["x"] > index] #consider resetting index
                return train, test

            percent_split = st.number_input("What percent of the data should be used for training?", 50.0, 90.0)

            train, test = time_split(df, percent_split)

            submitted = st.form_submit_button("Submit Values") 

            if submitted:
                st.success("Parameters chosen! Select kernels and start fitting your model on the sidebar to the left.")


        ### TESTING:

        with st.sidebar.form("my_sidebar_form"):

            ### initialize interpret dict

            interpret_dict = {}

            kernel_select = st.sidebar.multiselect("Select Kernels", ["RationalQuadratic", "ExpSineSquared", "ExpSineSquared_2", "DotProduct", "WhiteKernel", "RBF", "Matern", "DotProduct Squared"], "DotProduct")

            if kernel_select:
                st.sidebar.header("Kernels")

            if "RationalQuadratic" in kernel_select:

                expander = st.sidebar.beta_expander("Rational Quadratic")
                with expander:
                    rational_weight = st.slider("Rational Quadratic Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
                    rational = st.slider("Rational Quadratic Kernel L", min_value = 0.1, max_value = 100.0)
                    alpha = st.slider("Rational Quadratic Alpha", min_value = 0.1, max_value = 100.0)
                
                interpret_dict["RationalQuadratic"] = rational_weight * RationalQuadratic(length_scale = rational, alpha = alpha)

            if "DotProduct" in kernel_select:

                expander = st.sidebar.beta_expander("Dot Product")
                with expander:
                    dot_weight = st.slider("Dot Product Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
                    sigma_dot = st.slider("Sigma for DotProduct", min_value = 0.0, max_value = 100.0)
                
                interpret_dict["DotProduct"] = dot_weight * DotProduct(sigma_0 = sigma_dot)

            if "DotProduct Squared" in kernel_select:
                expander = st.sidebar.beta_expander("Dot Product Squared")
                with expander:
                    dot_squared_weight = st.slider("Dot Product Squared Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
                    squared_sigma = st.slider("Sigma for DotProduct Squared", min_value = 0.0, max_value = 100.0)
                interpret_dict["DotProduct Squared"] = dot_squared_weight * DotProduct(sigma_0=squared_sigma) * DotProduct(sigma_0=squared_sigma)

            if "ExpSineSquared" in kernel_select:
                expander = st.sidebar.beta_expander("Exponential Sine Squared")
                with expander:
                    sine_weight = st.slider("Exp Sine Squared Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
                    exp_sine_1 = st.slider("First Exp Sine Squared L", min_value = 0.1, max_value = 100.0)
                    exp_sine_1_period = st.slider("First Exp Sine Squared Period", min_value = 0.1, max_value = 100.0)
                interpret_dict["ExpSineSquared"] = sine_weight * ExpSineSquared(length_scale = exp_sine_1, periodicity = exp_sine_1_period)

            if "ExpSineSquared_2" in kernel_select:
                expander = st.sidebar.beta_expander("Exponential Sine Squared 2")
                with expander:
                    sine_weight_2 = st.slider("Exp Sine Squared Weight 2", min_value = 0.1, max_value = 100.0, value = 1.0)
                    exp_sine_2 = st.slider("Second Exp Sine Squared L", min_value = 0.1, max_value = 100.0)
                    exp_sine_2_period = st.slider("Second Exp Sine Squared Period", min_value = 0.1, max_value = 100.0)
                interpret_dict["ExpSineSquared_2"] = sine_weight_2 * ExpSineSquared(length_scale = exp_sine_2, periodicity= exp_sine_2_period)

            if "RBF" in kernel_select:
                expander = st.sidebar.beta_expander("RBF")
                with expander:
                    rbf_weight = st.slider("RBF Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
                    rbf = st.slider("RBF L", min_value = 0.1, max_value = 100.0)
                interpret_dict["RBF"] = rbf_weight * RBF(length_scale=rbf)

            if "Matern" in kernel_select:
                expander = st.sidebar.beta_expander("Matern")
                with expander:
                    matern_weight = st.slider("Mattern Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
                    matern_length = st.slider("Mattern Length", min_value = 0.1, max_value = 100.0, value = 1.0)
                    matern_nu = st.slider("Mattern Nu", min_value = 0.1, max_value = 10.0, value = 1.5) #NOTE: Parameter changed to max 10, else it can break
                interpret_dict["Matern"] = matern_weight * Matern(length_scale=matern_length, nu = matern_nu)
            
            if "WhiteKernel" in kernel_select:
                expander = st.sidebar.beta_expander("WhiteKernel")
                with expander:
                    white_weight = st.slider("White Noise Weight", min_value = 0.1, max_value = 100.0, value = 1.0)
                    white_noise = st.slider("NoiseLevel", min_value = 0.1, max_value = 10.0)
                interpret_dict["WhiteKernel"] = white_weight * WhiteKernel(white_noise)

            submitted_sidebar = st.form_submit_button("Fit the Model")

        if submitted_sidebar:

            '''
            ## Fitted values
            '''
                
            np.random.seed(1)

            if kernel_select:
                # ----------------------------------------------------------------------

                # Instantiate a Gaussian Process model

                for i, ele in enumerate(kernel_select):
                    element = interpret_dict.get(ele)
                    if i == 0:
                        kernel = element
                    else:
                        kernel += element

                ## Unpack data 
                ### Most likely here the problem with mattern and RBF is. Needs to be numeric things in order to divide.
                x_train, y_train = train["x"].values, train[y_select].values

                x_test, y_test = test["x"].values, test[y_select].values

                #get right format

                x_train = np.atleast_2d(x_train).T
                x_test = np.atleast_2d(x_test).T

                expander = st.beta_expander("Visualize your kernels")

                with st.spinner("Plotting kernels..."):
                    with expander:
                        '''
                    #### Covariance matrices
                    Below are the covariance matrices for your selected kernels. The covariance matrices have $x$ on both axis. Each point on this matrice can be defined as $k(x_i, x_j)$. 

                    A good intuition to have here, is that if $k(x_i, x_j)$ has a bright color, the kernel estimates that these points are similar, and should therefore learn from each other's value.
                    '''
                        ## Warning message for kernels.
                        st.warning('Explanations of the different kernels are not shown here. Navigate to the *Learn and simulate* for complete explanations.')

                        st.pyplot(helper_functions.plot_kernels(kernel_select, interpret_dict, kernel, x_train))

                #initialize the processor
                gp = GaussianProcessRegressor(kernel=kernel,
                                            n_restarts_optimizer=9)

                # Fit to data using Maximum Likelihood Estimation of the parameters
                with st.spinner("Fitting the model..."):

                    gp.fit(x_train, y_train)

                    # Make the prediction on the meshed x-axis (ask for MSE as well)
                    y_pred, sigma = gp.predict(x_train, return_std=True)

                    # predict test data

                    y_pred_test, sigma_test = gp.predict(x_test, return_std=True)

                    # get errors for test data
                    errors = y_test - y_pred_test
                    errors = errors.flatten()
                    errors_mean = errors.mean()
                    errors_std = errors.std()
                st.success("Done fitting!")

                ## PLOT
                with st.spinner("Plotting predictions... "):
                    fig, ax = plt.subplots(2, 1, figsize=(12, 10)) 
                    ax[0].plot(x_train, y_train, 'r:', label='Observations')
                    ax[0].plot(x_train, y_pred, 'b-', label='Prediction')
                    ax[0].fill(np.concatenate([x_train, x_train[::-1]]),
                            np.concatenate([y_pred - 1.9600 * sigma,
                                            (y_pred + 1.9600 * sigma)[::-1]]),
                            alpha=.5, fc='b', ec='None', label='95% confidence interval')
                    ax[0].plot(x_test, y_test, 'r:', label='_nolegend_')
                    ax[0].plot(x_test, y_pred_test, 'b-', label='_nolegend_')
                    ax[0].fill(np.concatenate([x_test, x_test[::-1]]),
                            np.concatenate([y_pred_test - 1.9600 * sigma_test,
                                            (y_pred_test + 1.9600 * sigma_test)[::-1]]),
                            alpha=.5, fc='b', ec='None', label='_nolegend_')
                    #ax[0].xlabel('$x$')
                    #ax[0].ylabel('$f(x)$')
                    ax[0].axvline(x_train[-1]) #maybe not flexible enough
                    ax[0].set(xlabel = "$x$", ylabel = "$y$")
                    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                        fancybox=True, shadow=True, ncol=2)
                    ### HISTOGRAM:
                    sns.distplot(a=errors, bins = 20, ax=ax[1])
                    ax[1].axvline(x=errors_mean, color=sns_c[3], linestyle='--', label=f'$\mu$')
                    ax[1].axvline(x=errors_mean + 2*errors_std, color=sns_c[4], linestyle='--', label=f'$\mu \pm 2\sigma$')
                    ax[1].axvline(x=errors_mean - 2*errors_std, color=sns_c[4], linestyle='--')
                    ax[1].axvline(x=errors_mean, color=sns_c[3], linestyle='--')
                    ax[1].legend()
                    ax[1].set(title='Distribution of Error', xlabel='error', ylabel=None);
                    st.pyplot(fig)
                st.success("Done plotting!")

                col1, col2, col3 = st.beta_columns([1,6,1])

                with col1:
                    st.write("")

                with col2:
                    stringing = str(gp.kernel)
                    stringing = stringing.replace("**", "^")
                    stringing = stringing.replace("*", "\cdot")
                    stringing = stringing.replace("_", " ")

                    #stringing = re.sub(r"**", "^", stringing)
                    '''### Current settings:'''


                    f'''$${stringing}$$'''

                    expander = st.beta_expander("Accuracy")
                    with expander:
                        '''Here, we are using the mean squared error 
                        to get an accuracy measurement of our model.
                        Mathematically, this is given by:'''
                        r''' ### $$\frac{1}{n} \sum ^{n}_{i=1} (y-\hat{y})^2$$'''

                        f'''
                        ### $$MSE = {helper_functions.get_RMSE(y_test, y_pred_test)}$$
                        '''

                with col3:
                    st.write("")
            
            '''
            ## Code for running Gaussian Processes
            '''
            ### CODE
            ### NOTE: Currently, the code doesn't include real names and specifications.
            expander = st.beta_expander("Show code")
            with expander:
                code_block = f'''
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic, ExpSineSquared, DotProduct, RBF

#ensure reproducibility by having a random seed
np.random.seed(1)

#define function for splitting in a train and test set
def time_split(df, splitter):
        percent = len(df['{x_select}']) / 100
        index = round(percent * splitter)

        train = df[df['{x_select}'] <= index]
        test = df[df['{x_select}'] > index] 
        return train, test

#use the function
train, test = time_split(df, {percent_split})

#extract values
x_train, y_train = train['{x_select}'].values, train['{y_select}'].values

x_test, y_test = test['{x_select}'].values, test['{y_select}'].values

#specify the kernel
kernel = {kernel}

#Initialize the Gaussian processor with the kernel
gp = GaussianProcessRegressor(kernel=kernel,
                                    n_restarts_optimizer=9)

#make arrays 2d
x_train = np.atleast_2d(x_train).T
x_test = np.atleast_2d(x_test).T

## fit the model
gp.fit(x_train, y_train)

#predict on the training data
y_pred, sigma = gp.predict(x_train, return_std=True)

# predict on the test data
y_pred_test, sigma_test = gp.predict(x_test, return_std=True)
'''
                st.code(code_block)

            ### Downloadable link
            y_column = np.append(y_pred, y_pred_test) 
            sigma_column = np.append(sigma, sigma_test)

            df["y_pred"] = y_column
            df["sigma"] = sigma_column

            coded_data = base64.b64encode(df.to_csv(index = False).encode()).decode()
            st.markdown(
                f'<a href="data:file/csv;base64,{coded_data}" download="GP_predictions.csv">Click here to download the .csv file with predictions</a>',
                unsafe_allow_html = True
            )


## TODO:
# (3) Gentænk strukturen for "Guided Tour". Tror bare det skal komme som sin egen ting, men med containers indeni.
# (5) Read about human in the loop
# (6) Find dataset, where this is just brilliant
# (7) Add kernels to Upload your own data.
# (8) Consider having the second ExpSine as a multiplicative component
# (10) Check whether final kernel is really the right one - when it fits, does it look different? Does not!
# (12) Insert a "what to look out for in terms of mistakes"
# (13) Consider scaling everything - does that make things better? 
# (14) Insert period for exp1 as well
# (15) should all weights just be between 0 and 1?
# (16) Make explanations for all kernels finished in the current format.
# (17) Fix "Known Issues"
# (18) Explanations for plots?