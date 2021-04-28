'''
This code is the implementation of the texture descriptors presented in the paper
"COVID-index: a texture-based approach to classify lung lesions based on CT images".

The features were extracted from the databases mentioned in the article, which have images of covid-19 lesions,
solid lesions and healthy tissue (non-nodules).

Important notes:
    * label 0: solid
    * label 1: covid
    * label 2: non-nodule

Developed by: Patrick Sales and Vitória Carvalho.

Last update: April 2021.
'''

# -----------------------------------------------------------------------------------------------------------------------------

from glob import glob
import nibabel as nib
import numpy as np
import pandas as pd
import warnings
from skimage.io import imshow
from tqdm import tqdm
from joblib import Parallel, delayed
from pydicom import dcmread, read_file
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------------------------------------------------

def descriptor_from_hist(hist_dict):

    '''
    This function is responsible for calculating the indexes from a histogram.

    Parameters:
        hist_dict: dict, required
            A dictionary that has 3 keys: the 'data' key is a dictionary that stores some information
            calculated from the histogram and that is the basis for the calculation of the PSV and PSR indexes;
            the 'label' key is the histogram label; and the 'hist' key is the image histogram itself.

    Returns:
        feat: list
            A list containing the name of the image from which the histogram was extracted, the 8 calculated
            indexes and the label of the histogram.
    '''
  
    data = hist_dict['data']
    label = hist_dict['label']
    histogram = data['hist']
    sizeHistogram = len(histogram)
    
    n = np.sum(histogram)

    sumDelta = 0
    sumDelta_ = 0
    MNND = 0
    S1 = 0
    S2 = 0
    sumMPD1 = 0
    sumMPD2 = 0
    sum1 = 0
    
    qtdSpecies = data['qtdSpecies']
    sumValuesMatrix = data['sumValuesMatrix']
    sumDiagonal = data['sumDiagonal']
    
    PSV = ((3*sumDiagonal)-21)/((3*(3-1)))
    PSR = 3*PSV
    
    for i in range(0, sizeHistogram):
      for j in range(i + 1, sizeHistogram):
        
        # calculate distance
        distance = ((j - i) + 1)
        if(i != 0):
          distance = ((j - i) + 2)

        # calculate pipj
        pipj = 0
        if(histogram[i] > 0 and histogram[j] > 0):
          pipj = 1

        # Taxonomics
        sumDelta += distance * histogram[i] * histogram[j]
        sumDelta_ += histogram[i] * histogram[j]

        # MNND
        if(j == i+1 and i < sizeHistogram-1):
          # j == 0: the distance should be calculated only for the closest relative
          # i < sizeHistograma - 1: in this condition, the closest relative of the last species has already been calculated
          MNND += (distance * histogram[i])

        # PD
        histAi = histogram[i:j+1]
        sum1 += np.sum(histAi)
        Ai = np.sum(histAi) / len(histAi)
        S1 += (distance * Ai)
        S2 += Ai

        # MPD
        sumMPD1 += distance * pipj
        sumMPD2 += pipj


    delta = sumDelta/(n*(n-1)/2)
    delta_ = sumDelta/(sumDelta_+0.001)
    SPD = (sizeHistogram*(sizeHistogram-1)/2)*delta_
    MPD = sumMPD1/(sumMPD2+0.001)
    PD = (sizeHistogram-1)*(S1/(S2+0.001))

    feat = []  
    feat.append(data['name'])
    feat.append(PD)
    feat.append(SPD)
    feat.append(MNND)
    feat.append(PSV)
    feat.append(PSR)
    feat.append(MPD)
    feat.append(delta)
    feat.append(delta_)
    feat.append(label)

    return feat

# -----------------------------------------------------------------------------------------------------------------------------

def calc_hist(img, name):

    '''
    This function is responsible for calculating a histogram from a 3D image.

    Parameters:
        img: numpy array, required 
            A 3D image of type numpy array.

        name: str, required
            Image name.

    Returns:
        0: int
            If the volume has a 1x1x1 shape.

        data: dict
            If the volume does not have a 1x1x1 shape.
            The variable data is a dictionary that stores the histogram information.
    '''
    
    img= np.int_(img)
    img_min = img.min()    
    
    hist = [0]*(img.max()+1)

    sumDiagonal = 0
    sumValuesMatrix = 0

    if len(img.shape) < 3 or (img.shape[0] == 1 and img.shape[1] == 1  and img.shape[2] == 1):
        return 0

    else:
        for k in range(img.shape[0]):
            for i in range(img.shape[1]):
                for j in range(img.shape[2]):
                    #if img[i][j][k] != img_min:
                    if i == j:
                        sumDiagonal+= img[k][i][j]
                    else:
                        sumValuesMatrix+=img[k][i][j]
                    try:                    
                        hist[img[k][i][j]] += 1
                        

                    except Exception as e:
                        print(f'Exc: {e}, len_hist: {len(hist)}')

        data = {
            'hist': hist,
            'sumValuesMatrix': sumValuesMatrix,
            'sumDiagonal': sumDiagonal,
            'qtdSpecies': len(np.unique(img)),
            'shape': img.shape,
            'name': name
        }

        return data
    
# -----------------------------------------------------------------------------------------------------------------------------

def load_nii(path):

    '''
    This function is responsible for reading an image of type nii from a path.

    Parameters:
        path: str, required
            Path of the .nii file.

    Returns:
        data: numpy array
            Structured image in a numpy array.
    '''
    
    data = nib.load(path)
    
    return data.get_fdata()

# -----------------------------------------------------------------------------------------------------------------------------

def load_dcm(path):

    '''
    This function is responsible for reading an image of type dcm from a path.

    Parameters:
        path: str, required
            Path of the .dcm file.

    Returns:
        data: numpy array
            Structured image in a numpy array.
    '''
    
    data = dcmread(path)
    
    return np.asarray(read_file(path).pixel_array, dtype=np.int64)

# -----------------------------------------------------------------------------------------------------------------------------

def get_hist(data_path, verbose=0):

    '''
   This function is responsible for performing the functions for reading the image, depending on its type,
   and the function for extracting the histogram from the read image.

    Parameters:
        path: str, required
            Path of the image file.
        
        verbose: int, default 0
            If different from 0, shows the information extracted from the image.

    Returns:
        data: calc_hist(img, name)
            Return from the calc_hist function, that is, a dictionary with the extracted histogram information.
    '''
    
    load = {'nii': load_nii,
            'dcm': load_dcm}

    img_name = data_path.split('/')[-1]
    img = load[img_name.split('.')[-1]](data_path)
    img = (img - img.min()).astype(int)
    
    if verbose:
        print('---- img description ---')
        print(f'Name: {img_name}')
        print(f'Shape: {img.shape}')
        print(f'Max: {img.max()}')
        print(f'Min: {img.min()}')

    data = calc_hist(img, img_name)

    return data

# -----------------------------------------------------------------------------------------------------------------------------

def cal_hists_from_paths(paths, verbose=0):

    '''
    This function is responsible for generating a histogram for each image in the path list,
    using the get_hist function.

    Parameters:
        paths: list, required
            List of image paths.
        
        verbose: int, default 0
            If different from 0, shows the information extracted from the image. Parameter used by the
            get_hist function.

    Returns:
        hists: list
            List with the extracted histograms.
    '''

    hists = []

    for path in paths:
        var = get_hists(path, verbose)
        if var != 0:
            hists.append(var)
            
    return hists

# -----------------------------------------------------------------------------------------------------------------------------

def save(features, path):
    
    '''
    This method is responsible for saving the array of features as a csv file, using pandas.

    Parameters:
        features: numpy array, required
            A numpy array that stores, for each image, the name, the 8 calculated indices and the label.
        
        path: str, required
            Path where the file should be saved.
    '''

    df = pd.DataFrame(data={
                            "name": features[:,0],
                            "pd": features[:,1],
                            "spd": features[:,2],
                            "mnnd": features[:,3],
                            "psv": features[:,4],
                            "psr": features[:,5],
                            "mpd": features[:,6],
                            "delta": features[:,7],
                            "delta_": features[:,8],
                            "label": features[:,9]
                           }
                     )

    df.to_csv(path, sep=',',index=False)
    
# -----------------------------------------------------------------------------------------------------------------------------

print('Loading dataset...\n')

voi_lidc_idri = glob('./Covid-19/voi_lidc_idri/solid/*')
voi_infection_covid19_base1 = glob('./Covid-19/voi_infection_covid19_base1/*')
voi_infection_covid19_base2 = glob('./Covid-19/voi_infection_covid19_base2/*')
non_nodule = sorted(glob('./Covid-19/non-nodule/*'))

print(f'Solid: {len(voi_lidc_idri)}, covid 1: {len(voi_infection_covid19_base1)}, covid 2: {len(voi_infection_covid19_base2)}, non nodule: {len(non_nodule)}')

# -----------------------------------------------------------------------------------------------------------------------------

print('\n\nExtracting histograms...\n')

hist_solid = cal_hists_from_paths(voi_lidc_idri)
hist_covid = cal_hists_from_paths(voi_infection_covid19_base1 + voi_infection_covid19_base2)
hist_non_nodule = cal_hists_from_paths(non_nodule[4000:8000])

print(f'hist_covid: {len(hist_covid)}, hist_solid: {len(hist_solid)}, hist_non_nodule: {len(hist_non_nodule)}')

# -----------------------------------------------------------------------------------------------------------------------------

hists = hist_solid + hist_covid + hist_non_nodule
labels = [0]*len(hist_solid) + [1]*len(hist_covid) + [2]*len(hist_non_nodule)

data = []

for i in range(len(hists)):
    data.append({
        'data': hists[i],
        'label': labels[i]
    })
    
print(f'Number of histograms: {len(data)}\n\n')

print('\n\nExtracting indexes...\n')

features = Parallel(n_jobs=20)(delayed(descriptor_from_hist)(hist) for hist in tqdm(data))

save(np.array(features), './indexes_features.csv')

print('Done!')
