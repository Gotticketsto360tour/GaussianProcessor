import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic, ExpSineSquared, DotProduct, RBF

#@st.cache(allow_output_mutation=True)
def get_RMSE(y, y_pred):
    listing = [np.sqrt((y[i]-y_pred[i])**2) for i in range(len(y))]
    return sum(listing)/len(listing)

#@st.cache(allow_output_mutation=True) - if true, the test aspect breaks in simulation, and its not that heavy
def f(x, lin_trend, sinus, sinus_2, sinus_2_period, poly_trend):
    return lin_trend * x + sinus * np.sin(x) + sinus_2 * np.sin(sinus_2_period * x) + poly_trend * x**2

#@st.cache(allow_output_mutation=True)
def get_kernel(kernel_select, interpret_dict):
    
    for i, ele in enumerate(kernel_select):
        element = interpret_dict.get(ele)
        if i == 0:
            kernel = element
        else:
            kernel += element
    
    return kernel

#@st.cache(allow_output_mutation=True)
def plot_kernels(kernel_select, interpret_dict, kernel, X):

        #plot kernels
        fig, ax = plt.subplots(len(kernel_select) + 1, 1, figsize = (12,10), constrained_layout = True)#figsize = (12, 10)
        for i, ele in enumerate(kernel_select):
            kernel_viz = interpret_dict.get(ele)
            kernel_viz = kernel_viz.__call__(X)
            sns.heatmap(kernel_viz, ax = ax[i], cbar = False)
            #ax[i].imshow(kernel_viz)
            ax[i].set(title = ele, xticklabels=[], yticklabels=[])
            ax[i].legend([],[], frameon=False)
        kernel_viz = kernel.__call__(X)
        sns.heatmap(kernel_viz, ax = ax[-1], cbar = False)
        #ax[-1].imshow(kernel_viz)
        ax[-1].set(title = "Combined", xticklabels=[], yticklabels=[])
        ax[-1].legend([], [], frameon=False)

        fig.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
        return fig

#@st.cache(allow_output_mutation=True)
def generate_interpret_dict(kernel_select):
    interpret_dict = {
                    "RationalQuadratic": rational_weight * RationalQuadratic(length_scale = rational, alpha = alpha),
                    "ExpSineSquared": sine_weight * ExpSineSquared(length_scale = exp_sine_1),
                    "ExpSineSquared_2": sine_weight_2 * ExpSineSquared(length_scale = exp_sine_2, periodicity= exp_sine_2_period),
                    "DotProduct": dot_weight * DotProduct(sigma_0 = sigma_dot),
                    "WhiteKernel": white_weight * WhiteKernel(white_noise),
                    "RBF": rbf_weight * RBF(length_scale=rbf),
                    "Matern": matern_weight * Matern(length_scale=matern_length, nu = matern_nu),
                    "DotProduct Squared": dot_squared_weight * DotProduct(sigma_0=squared_sigma) * DotProduct(sigma_0=squared_sigma)
                }
    