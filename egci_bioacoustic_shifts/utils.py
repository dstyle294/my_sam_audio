# imports

from multiprocessing import Pool

import array
import numpy as np

from scipy.linalg import toeplitz
from scipy.stats import zscore, entropy
import scipy.stats
import librosa

import statsmodels.api as sm
from statsmodels.tsa.stattools import acf

import matplotlib.pyplot as plt
import plotly.graph_objs as go
import csv
import pandas as pd

from datasets import load_dataset, Dataset, DatasetDict
import tensorflow_hub as hub
from torch import binary_cross_entropy_with_logits, Tensor

from tqdm import tqdm

from typing import Callable, Optional



# Data Processing Utils

def one_hot_encode(labels, classes: list[str], approved_classes: dict) -> np.array:
    """
    One-hot encoding for multiclass labels
    """
    
    one_hot = np.zeros(len(classes))
    for label in labels: 
        if label in approved_classes.keys():
            one_hot[approved_classes[label]] = 1
    return np.array(one_hot, dtype=float)

def class_names_from_csv(labels_path=None):
    """
    Returns list of class names corresponding to score vector from labels csv
    """
    with open(labels_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        class_names = [mid[0] for mid in csv_reader]
        return class_names[1:]

def process_data(data: dict, audio_processing: Callable = lambda x, y, z: (x, z), lag=256 ):
    """
    Processes data from dictionary of audio data (e.g. dataset row)
    
    Uses no preprocessing by default but can be supplied with a preprocessing
    function for audio array
    
    Parameters
    ----------
        data: `Dataset`
    
    Returns
    -------
    Dictionary of metadata, as well as entropy and complexity
    """
    
    path = data["audio"]["path"]
    i = 0
    if data["length"] is not None and data["length"] > 5:
        i = np.random.randint(0, data["length"] - 5)

    try:
        audio1, sr = librosa.load(path, sr=32_000, duration=5, offset=i)
        label = data["ebird_code_multilabel"]
        audio, label = audio_processing(audio1, sr, label)
        h, c, lag = EGCI(audio, lag=lag)
    except Exception as e:
        print("2", e)
        return None
    
    output_data = {
            "path": path,
            "offset_s": 0,
            "sr": sr,
            "gt": label,
            "entropy": h,
            "complexity": c,
            "lag": lag
        }
    return output_data

def process_data_with_identity(data):
    """
    Identity function for data processing (applies no processing)
    """
    return process_data(data, audio_processing=lambda audio, sr, label: (audio,label) )


def multiprocess_data(
        process_data_func: Callable = process_data, 
        data_repo: Optional[Dataset] = None,  
        processes: int = 1, 
        sample=50) -> tuple[list]:
    """
    Takes samples from an audio dataset split and calculates EGCI 
    for each sample, then returns the result as a list, along with
    a numpy array of the sample indices
    """
    
    if data_repo is None:
        data_repo = load_dataset("DBD-research-group/BirdSet", "PER", trust_remote_code=True, revision="b0c14a03571a7d73d56b12c4b1db81952c4f7e64")["test_5s"]
    
    if type(sample) == int:
        indx_sampled = np.random.randint(0, len(data_repo), sample)
    else:
        indx_sampled = sample

    with Pool(processes) as p:
        data = list(tqdm(p.imap(process_data_func, data_repo.select(indx_sampled))))
        
    return data, indx_sampled

# Information Theory Utils

def Entropy(p1: np.ndarray) -> np.ndarray:
    """
    From Colonna et. al., calculates von Neumann Entropy using SciPy's
    Shannon's Entropy implementation
    """
    p1 = p1/np.sum(p1)
    return entropy(p1)/np.log(len(p1))

def JSD(p: np.ndarray, q=None) -> tuple[float]:
    """
    Calculates Jensen-Shannon Divergence
    """
    
    n = len(p)
    if q is None:
        q = np.ones(n)/n # Uniform reference
    elif type(q) is not np.ndarray:
        raise "Bad type for equilibrium distributions"
    elif len(q) != n:
        raise "Distributions are not the same size"
    else:
        q = q/q.sum() # normalize q
    
    p = np.asarray(p)
    q = np.asarray(q)
    p = p/p.sum() # normalize
    m = (p + q) / 2
    
    jensen0 = -2*((((n+1)/n)*np.log(n+1)-2*np.log(2*n) + np.log(n))**(-1))
    
    return jensen0*(entropy(p, m) + entropy(q, m)) / 2

# https://github.com/juancolonna/EGCI/blob/master/Example_of_EGCI_calculation.ipynb
def EGCI(x: np.ndarray, lag: int = 512) -> tuple[float]:
    """
    Calculates EGCI according to the method outlined in Colonna et. al. (2020)
    
    Parameters
    ----------
    x : `np.ndarray`plot
        An audio 
    lag : `int`
        t_max from the paper, the maximum value of t, the number of
        milleseconds shifted in autocorrelation

    Returns
    -------
    von Neumann Entropy, EGCI for the given audio
    """
    
    x = zscore(x)
    
    # # Algorithm steps 
    rxx = acf(x, nlags=lag, adjusted=True, fft=True)
    
    # #https://github.com/blue-yonder/tsfresh/issues/902
    Sxx = toeplitz(rxx)
    s = np.linalg.svd(Sxx, compute_uv=False)
    
    return Entropy(s), Entropy(s)*JSD(s), lag           # (Entropy, Complexity)

# Plotting

def plot_EGCI(H: list[float], C: list[float], lag: int, axes:plt.Axes=None, label=None, color="lightblue", plot_boundries=True):
    """
    Plots a scatterplot of EGCI against von Neumann's Entropy given lag
    """
    if axes is None:
        axes = plt.subplot(1, figsize=(11,9), dpi=500)
    
    cotas = pd.read_csv('plotting_utils/Cotas_HxC_bins_' + str(int(lag)) + '.csv')
    if plot_boundries:
        noise = pd.read_csv('plotting_utils/coloredNoises_' + str(int(lag)) + '.csv')
        axes.plot(cotas['Entropy'],cotas['Complexity'], '--k', label = 'HxC boundaries')
        axes.plot(noise['Entropy'],noise['Complexity'], '--b', label = 'Colored noises')

    axes.scatter(H, C, c=color, marker='.', s=2, label=label)
    axes.set_xlim([0, 1])
    axes.set_ylim([0, np.max(cotas['Complexity'])+0.01])
    axes.set_ylabel('Complexity [Cf]', fontsize=18)
    axes.set_xlabel('Entropy [Hf]', fontsize=18)
    axes.legend(loc = 'upper left', fontsize=12)
    # plt.show()
    return axes
    
def load_EGCI(
        region = "PER", 
        dataset_sub="test_5s",
        fig=None, 
        indx=None,
        process_data_func=process_data, sample=2000, label="", ds=None, workers=12, lag=256):
    """
    Loads in BirdSet dataset split and performs data processing on it, then plots EGCI
    """
    if ds is None:
        ds = load_dataset("DBD-research-group/BirdSet", region, trust_remote_code=True, revision="b0c14a03571a7d73d56b12c4b1db81952c4f7e64")
    
    # if indx is None:
    #     indx = np.random.choice(np.arange(len(ds[dataset_sub])), sample, replace=True)
    out, indx_sampled = multiprocess_data(process_data_func=process_data_func, data_repo = ds[dataset_sub], processes=workers, sample=sample)
    # print(out)
    h, c, s = [], [], []

    
    for data in out:
        # Errors loading data can be common...
        if data is not None:
            # print(data)
            h.append(data["entropy"])
            c.append(data["complexity"])
            #s.append(gt[i])
            
    cotas = pd.read_csv('plotting_utils/Cotas_HxC_bins_' + str(int(lag)) + '.csv')
    noise = pd.read_csv('plotting_utils/coloredNoises_' + str(int(lag)) + '.csv')

    first_fig = fig is None
    if first_fig:
        fig = plt.figure(figsize=(11,9))
        plt.plot(cotas['Entropy'],cotas['Complexity'], '--k', label = 'HxC boundaries')
        plt.plot(noise['Entropy'],noise['Complexity'], '--b', label = 'Colored noises')
        
    

    # for i in range(len(H)):
    plt.scatter(h, c, marker='.', s=1, cmap='viridis', label=f"{region} {dataset_sub} {label}")
    plt.xlim([0, 1])
    plt.ylim([0, np.max(cotas['Complexity'])+0.01])
    plt.ylabel('Complexity [Cf]')
    plt.xlabel('Entropy [Hf]')
    lgnd = plt.legend(loc = 'best')
    # set sizes of points in legend to be the same
    for handle in lgnd.legend_handles:
        try:
            handle.set_sizes([200.0])
        except:
            continue
    fig.axes[0].set_title(f"Number of Species in Ground Truth By EGCI")

    if first_fig:
        plt.colorbar()
    return fig, out, (h, c, s), indx_sampled

def load_EGCI_losses(
        region = "PER", 
        dataset_sub="test_5s",
        fig=None, 
        indx=50,
        process_data_func=process_data_with_identity, label_fig="",
        lag: int = 512,
        workers: int = 12
    ):
    """
    Calculates and plots EGCI for the BirdSet data along with losses 
    """
    model=hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8')
    labels_path=hub.resolve('https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8') + "/assets/label.csv"
    cotas = pd.read_csv('plotting_utils/Cotas_HxC_bins_' + str(int(lag)) + '.csv')
    noise = pd.read_csv('plotting_utils/coloredNoises_' + str(int(lag)) + '.csv')
    
    ds = load_dataset("DBD-research-group/BirdSet", region, trust_remote_code=True, revision="b0c14a03571a7d73d56b12c4b1db81952c4f7e64")
    
    class_list = ds["test"].features["ebird_code"].names

    # In case model has a missing label from dataset
    blocked_classes = []
    approved_classes = {}
    classes = class_names_from_csv(labels_path)
    class_targets = []
    for ebird_code in ds[dataset_sub].features["ebird_code"].names:
        # print(class_list.index(ebird_code), ebird_code, ebird_code in classes)
        if ebird_code in classes: #Correct index if a class is missing in the middle
            class_targets.append(classes.index(ebird_code))
            approved_classes[class_list.index(ebird_code)] = class_list.index(ebird_code) - len(blocked_classes)
        else:
            blocked_classes.append(class_list.index(ebird_code))

    # Get EGCI metrics
    out, indx = multiprocess_data(process_data_func=process_data_func, data_repo = ds[dataset_sub], processes=workers, sample=indx)
    
    #print(out)
    h, c, preds, losses, labels = [], [], [], [], []

    for i in range(len(out)):
        if (out[i] is not None):
            try:
                audio, sr = librosa.load(out[i]["path"], sr=32_000)
            except Exception as e:
                print(e)
                continue
            try:
                model_outputs = model.infer_tf(audio[np.newaxis, :])
            except Exception as e:
                print(e)
                continue

            
            label = one_hot_encode(out[i]["gt"], class_targets, approved_classes)
            logits = np.array(model_outputs['label'])[:, class_targets]
            # print(Tensor(logits).shape, Tensor(label).unsqueeze(0).shape)
            losses.append(float(
                binary_cross_entropy_with_logits(
                Tensor(label).unsqueeze(0), Tensor(logits), reduction=1
            )))
            preds.append(logits)
            h.append(out[i]["entropy"])
            c.append(out[i]["complexity"])
            labels.append(label)

    first_fig = fig is None
    if first_fig:
        fig = plt.figure(figsize=(11,9))
        plt.plot(cotas['Entropy'],cotas['Complexity'], '--k', label = 'HxC boundaries')
        plt.plot(noise['Entropy'],noise['Complexity'], '--b', label = 'Colored noises')
        
    plt.scatter(h, c, marker='.', s=1, c='blue', label=f"{region} {dataset_sub} {label_fig}")
    plt.xlim([0, 1])
    plt.ylim([0, np.max(cotas['Complexity'])+0.01])
    plt.ylabel('Complexity [Cf]')
    plt.xlabel('Entropy [Hf]')
    lgnd = plt.legend(loc = 'best')
    
    # set sizes of points in legend to be the same
    for handle in lgnd.legend_handles:
        try:
            handle.set_sizes([200.0])
        except:
            continue
    fig.axes[0].set_title(f"Number of Species in Ground Truth By EGCI")

    if first_fig:
        plt.colorbar()
    
    return fig, out, (h, c, preds, losses, labels), indx


def plot_ols_plane(data, model, x_col, y_col, z_col):
    """
    Create a 3D scatter plot with regression plane from an OLS model.

    Parameters:
    - data: pd.DataFrame
    - model: fitted OLS model from statsmodels
    - x_col, y_col: names of independent variable columns
    - z_col: name of dependent variable column
    """
    # Scatter points
    scatter = go.Scatter3d(
        x=data[x_col],
        y=data[y_col],
        z=data[z_col],
        mode='markers',
        marker=dict(size=1, color='blue'),
        name='Data Points'
    )

    # Create meshgrid for the regression plane
    x_range = np.linspace(data[x_col].min(), data[x_col].max(), 20)
    y_range = np.linspace(data[y_col].min(), data[y_col].max(), 20)
    xx, yy = np.meshgrid(x_range, y_range)

    # Flatten and prepare input for prediction
    plane_df = pd.DataFrame({x_col: xx.ravel(), y_col: yy.ravel()})
    plane_df = sm.add_constant(plane_df)  # Add intercept

    zz = model.predict(plane_df).values.reshape(xx.shape)

    # Regression plane
    plane = go.Surface(
        x=xx, y=yy, z=zz,
        colorscale='Viridis',
        opacity=0.6,
        name='OLS Plane',
        showscale=False
    )

    layout = go.Layout(
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        title='OLS Regression Plane with Data Points'
    )

    fig = go.Figure(data=[scatter, plane], layout=layout)
    fig.show()
    
def measure_distrbution_metrics(focal, soundscapes, emd=True):
    q = np.array(focal)
    p = np.array(soundscapes)

    # Limits number if either distrbution had error'd samples
    min_len = min(len(p), len(q))
    q = q[:min_len]
    p = p[:min_len]
    if emd:
        return {
            "Kullback-Leibler divergence Xeno-canto to Soundscapes": scipy.stats.entropy(p, q, axis=(1,0)),
            "Kullback-Leibler divergence Soundscapes to Xeno-canto": scipy.stats.entropy(q, p, axis=(1,0)),
            "Wasserstein Distance 2 Dimensional": scipy.stats.wasserstein_distance_nd(q, p)
        }
    else:
        return {
            "Kullback-Leibler divergence Xeno-canto to Soundscapes": scipy.stats.entropy(p, q, axis=(1,0)),
            "Kullback-Leibler divergence Soundscapes to Xeno-canto": scipy.stats.entropy(q, p, axis=(1,0)),
        }
    

