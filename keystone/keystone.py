import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm


def calculate_keystone(j, sample, original_abundance, model):
    p = original_abundance[j]
    sample[j] = 0
    original_abundance[j] = 0

    null_abundance = original_abundance / original_abundance.sum()

    with torch.no_grad():
        predicted_abundance = model(torch.tensor(sample.reshape(1, -1), dtype=torch.float32)).numpy()
    
    k = distance.braycurtis(null_abundance, predicted_abundance.flatten()) * (1 - p)

    return k



def calculate_keystone_array(sample_array, sample_abundances, model):
    K = np.zeros((sample_array.shape[0], sample_array.shape[1]))

    for i, sample in enumerate(tqdm(sample_array)):
        original_abundance = sample_abundances[i]
        for j, _ in enumerate(sample):
            K[i, j] = calculate_keystone(j, sample, original_abundance, model)
            
    return K