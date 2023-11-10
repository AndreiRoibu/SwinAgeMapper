import os
import numpy as np
import torch
import torch.nn  as nn
import torch.utils.data as data
import pandas as pd
from scipy.stats import norm
import nibabel as nib
import torchio as tio

class AddGaussianNoise(object):
    """
    Class adding gaussian noise to an input tensor
    """
    def __init__(self, 
                 mean: float = 0., 
                 std: float = 1.,
                 fix_seed: bool = False
                 ) -> None:
        """
        Add Gaussian Noise Initialiser

        Parameters:
        -----------
        mean : float
            Mean of the gaussian noise
        std : float
            Standard deviation of the gaussian noise

        Returns:
        --------
        None

        """

        self.std = std
        self.mean = mean
        self.fix_seed = fix_seed
        
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian Noise Call

        Parameters:
        -----------
        X : np.ndarray
            Input tensor

        Returns:
        --------
        torch.Tensor
            Input tensor with added gaussian noise
        """
        # return tensor + torch.randn(tensor.size()) * self.std + self.mean

        if self.std == 0 and self.mean == 0:
            return X
        else:
            if self.fix_seed:
                np.random.seed(0)
            return X + np.random.randn(X.size).reshape(X.shape) * self.std + self.mean
    
    def __repr__(self):
        """
        Add Gaussian Noise Representation

        Parameters:
        -----------
        None

        Returns:
        --------
        str
            String representation of the class

        """
        if self.std == 0 and self.mean == 0:
            return "No noise has been added"
        else:
            return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    

class AddSingleTransformation(object):
    """
    Class adding a single transformation to an input tensor. The transformation is selected from a list of possible transformations.
    """
    def __init__(self, 
                 transformationFlag: str = 'RandomNoise',
                 transformationMetric: float = 0.,
                 fix_seed: bool = False
                 ) -> None:
        """
        Add a class which applies a single transformation to the input tensor.

        Parameters:
        -----------
        transformationFlag : str
            Flag indicating the transformation to be applied. Possible values are: 'RandomNoise', 'RandomAnisotropy', 'RandomBiasField', 'RandomBlur', 'RandomGamma', 'RandomGhosting', 'RandomMotion', 'RandomSpike', 'RandomSwap', 'RandomElasticDeformation', 'RandomAffine', 'RandomFlip'. Default value is 'RandomNoise'. 
        transformationMetric : float
            Metric of the transformation. Default value is 0. 
        fix_seed : bool
            Flag indicating if the seed should be fixed. Default value is False.

        Returns:
        --------
        None

        """

        self.transformationFlag = transformationFlag
        self.transformationMetric = transformationMetric
        self.fix_seed = fix_seed

        if self.transformationFlag == 'RandomNoise':
            if self.fix_seed == True:
                torch.manual_seed(0)
                self.transformation = tio.RandomNoise(std=(self.transformationMetric, self.transformationMetric))
            else:
                self.transformation = tio.RandomNoise(std=(self.transformationMetric, self.transformationMetric))
        elif self.transformationFlag == 'RandomAnisotropy':
            if self.fix_seed == True:
                torch.manual_seed(0)
                self.transformation = tio.RandomAnisotropy(downsampling=(self.transformationMetric, self.transformationMetric))
            else:
                self.transformation = tio.RandomAnisotropy(downsampling=(self.transformationMetric, self.transformationMetric))
        elif self.transformationFlag == 'RandomBiasField':
            if self.fix_seed == True:
                torch.manual_seed(0)
                self.transformation = tio.RandomBiasField(coefficients=self.transformationMetric)
            else:
                self.transformation = tio.RandomBiasField(coefficients=self.transformationMetric)
        elif self.transformationFlag == 'RandomMotion':
            if self.fix_seed == True:
                torch.manual_seed(0)
                self.transformation = tio.RandomMotion(num_transforms=self.transformationMetric, image_interpolation='nearest')
            else:
                self.transformation = tio.RandomMotion(num_transforms=self.transformationMetric, image_interpolation='nearest')
        elif self.transformationFlag == 'RandomGhosting':
            if self.fix_seed == True:
                torch.manual_seed(0)
                self.transformation = tio.RandomGhosting(intensity=(self.transformationMetric, self.transformationMetric))
            else:
                self.transformation = tio.RandomGhosting(intensity=(self.transformationMetric, self.transformationMetric))
        elif self.transformationFlag == 'RandomSpike':
            if self.fix_seed == True:
                torch.manual_seed(0)
                self.transformation = tio.RandomSpike(intensity=(self.transformationMetric, self.transformationMetric))
            else:
                self.transformation = tio.RandomSpike(intensity=(self.transformationMetric, self.transformationMetric))
        elif self.transformationFlag == 'RandomGamma':
            if self.fix_seed == True:
                torch.manual_seed(0)
                self.transformation = tio.RandomGamma(log_gamma=self.transformationMetric)
            else:
                self.transformation = tio.RandomGamma(log_gamma=self.transformationMetric)
        elif self.transformationFlag == 'RandomAffine':
            if self.fix_seed == True:
                torch.manual_seed(0)
                self.transformation = tio.RandomAffine(scales = 0, degrees=self.transformationMetric)
            else:
                self.transformation = tio.RandomAffine(scales = 0, degrees=self.transformationMetric)
        elif self.transformationFlag == 'RandomNoiseMask':
            if self.fix_seed == True:
                torch.manual_seed(0)
                self.transformation = tio.RandomNoise(std=(self.transformationMetric, self.transformationMetric))
            else:
                self.transformation = tio.RandomNoise(std=(self.transformationMetric, self.transformationMetric))
            
            mask_path = 'datasets/MNI152_T1_1mm_brain_mask_dil.nii.gz'
            crop_values = [10, 170, 12, 204, 0, 160]
            self.mask = np.array(nib.load(mask_path).dataobj)
            self.mask = self.mask[crop_values[0]:crop_values[1],
                            crop_values[2]:crop_values[3], 
                            crop_values[4]:crop_values[5]]

        else:
            raise ValueError('The transformation selected is not available.')


        
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Perform the transformation on the input tensor. 

        Parameters:
        -----------
        X : np.ndarray
            Input tensor

        Returns:
        --------
        torch.Tensor
            Input tensor with added gaussian noise
        """

        if (self.transformationFlag in ['RandomNoise', 'RandomBiasField', 'RandomGhosting', 'RandomSpike', 'RandomGamma', 'RandomNoiseMask', 'RandomAffine'] and self.transformationMetric == 0) or (self.transformationFlag in ['RandomAnisotropy', ] and self.transformationMetric == 1):
            return X
        else:
            if self.transformationFlag == 'RandomNoiseMask':
                return np.multiply( self.mask, np.squeeze(self.transformation(np.expand_dims(X, 0))) )
            else:
                return np.squeeze(self.transformation(np.expand_dims(X, 0)))
    
    def __repr__(self):
        """
        Print the transformation applied. 

        Parameters:
        -----------
        None

        Returns:
        --------
        str
            String representation of the class

        """
        if (self.transformationFlag in ['RandomNoise', 'RandomBiasField', 'RandomGhosting', 'RandomSpike', 'RandomGamma'] and self.transformationMetric == 0) or (self.transformationFlag in ['RandomAnisotropy', ] and self.transformationMetric == 1):
            return "No transformation has been applied"
        else:
            return self.__class__.__name__ + '(Transformation Metric={0})'.format(self.transformationMetric)



class DynamicDataMapper(data.Dataset):

    def __init__(self, X_paths, y_ages, modality_flag, scale_factor, resolution, rsfmri_volume=None, shift_transformation=False,  mirror_transformation: bool = False,
                 transformationFlag: str = 'RandomNoise',
                 transformationMetric: float = 0.,
                 fix_seed: bool = False
                 ) -> None:

        self.X_paths = X_paths # List where all the file paths are already in the deisred order
        self.y_ages = y_ages
        self.scale_factor = scale_factor
        self.shift_transformation = shift_transformation
        self.mirror_transformation = mirror_transformation

        self.transformationFlag = transformationFlag

        if self.transformationFlag is None:
            self.additional_transformations = None
        elif self.transformationFlag == 'CustomRandomNoise':
            self.additional_transformations = AddGaussianNoise(mean=0., std=transformationMetric, fix_seed=fix_seed)
        else:
            self.additional_transformations = AddSingleTransformation(
                                                                    transformationFlag = transformationFlag,
                                                                    transformationMetric = transformationMetric, 
                                                                    fix_seed=fix_seed
                                                                    )

        self.rsfmri_volume = rsfmri_volume

        non_deterministic_modalities = ['T1_nonlinear', 'T1_linear', 'T2_nonlinear']
        if modality_flag in non_deterministic_modalities:
            self.non_deterministic_modality = True
        else:
            self.non_deterministic_modality = False

        if resolution == '2mm':
            self.crop_values = [5, 85, 6, 102, 0, 80]
        else:
            self.crop_values = [10, 170, 12, 204, 0, 160]
        
    def __getitem__(self, index):

        if self.rsfmri_volume != None:
            X_volume = np.array(nib.load(self.X_paths[index]).dataobj)[:,:,:,self.rsfmri_volume]
        else:
            X_volume = np.array(nib.load(self.X_paths[index]).dataobj)

        X_volume = X_volume[self.crop_values[0]:self.crop_values[1],
                            self.crop_values[2]:self.crop_values[3], 
                            self.crop_values[4]:self.crop_values[5]]

        if self.non_deterministic_modality == True:
            X_volume = X_volume / X_volume.mean()

        X_volume = X_volume / self.scale_factor

        if self.mirror_transformation==True:
            prob = np.random.rand(1)
            if prob < 0.5:
                X_volume = np.flip(X_volume,0)

        if self.shift_transformation==True:
            x_shift, y_shift, z_shift = np.random.randint(-2,3,3)
            X_volume = np.roll(X_volume,x_shift,axis=0)
            X_volume = np.roll(X_volume,y_shift,axis=1)
            X_volume = np.roll(X_volume,z_shift,axis=2)
            if z_shift < 0:
                X_volume[:,:,z_shift:] = 0

        if self.additional_transformations is not None:
            X_volume = self.additional_transformations(X_volume)

        X_volume = torch.from_numpy(X_volume)
        y_age = np.array(self.y_ages[index])

        return X_volume, y_age

    def __len__(self):
        return len(self.X_paths)


def select_datasets_path(data_parameters):
    dataset_sex = data_parameters['dataset_sex']
    dataset_size = data_parameters['dataset_size']
    data_folder_name = data_parameters['data_folder_name']

    X_train_list_path = data_parameters['male_train']
    y_train_ages_path = data_parameters['male_train_age']
    X_validation_list_path = data_parameters['male_validation']
    y_validation_ages_path = data_parameters['male_validation_age']

    if dataset_sex == 'female':
        X_train_list_path = "fe" + X_train_list_path
        y_train_ages_path = "fe" + y_train_ages_path
        X_validation_list_path = "fe" + X_validation_list_path
        y_validation_ages_path = "fe" + y_validation_ages_path

    if dataset_size == 'small':
        X_train_list_path += "_small.txt"
        y_train_ages_path += "_small.npy"
        X_validation_list_path += "_small.txt"
        y_validation_ages_path += "_small.npy"
    elif dataset_size == 'tiny':
        X_train_list_path += "_tiny.txt"
        y_train_ages_path += "_tiny.npy"
        X_validation_list_path += "_tiny.txt"
        y_validation_ages_path += "_tiny.npy"
    else:
        X_train_list_path += ".txt"
        y_train_ages_path += ".npy"
        X_validation_list_path += ".txt"
        y_validation_ages_path += ".npy"

    if dataset_size == 'han':
        X_train_list_path = "train_han.txt"
        y_train_ages_path = "train_age_han.npy"
        X_validation_list_path = "validation_han.txt"
        y_validation_ages_path = "validation_age_han.npy"

    if dataset_size == 'everything':
        X_train_list_path = "train_everything.txt"
        y_train_ages_path = "train_age_everything.npy"
        X_validation_list_path = "validation_everything.txt"
        y_validation_ages_path = "validation_age_everything.npy"

    if 'small' in dataset_size and dataset_size!='small':
        # ATTENTION! Cross Validation only enabled for male subjects at the moment!
        print('ATTENTION! CROSS VALIDATION DETECTED. This will only work for small male subject datasets ATM!')
        X_train_list_path = data_parameters['male_train'] + '_' + dataset_size + '.txt'
        y_train_ages_path = data_parameters['male_train_age'] + '_' + dataset_size + '.npy'
        X_validation_list_path = data_parameters['male_validation'] + '_' + dataset_size + '.txt'
        y_validation_ages_path = data_parameters['male_validation_age'] + '_' + dataset_size + '.npy'

    if dataset_size == 'testAB':
        X_train_list_path = 'male_testA.txt'
        y_train_ages_path = 'male_testA_age.npy'
        X_validation_list_path = 'male_testB.txt'
        y_validation_ages_path = 'male_testB_age.npy'
        if dataset_sex == 'female':
            X_train_list_path = "fe" + X_train_list_path
            y_train_ages_path = "fe" + y_train_ages_path
            X_validation_list_path = "fe" + X_validation_list_path
            y_validation_ages_path = "fe" + y_validation_ages_path

    X_train_list_path = os.path.join(data_folder_name, X_train_list_path)
    y_train_ages_path = os.path.join(data_folder_name, y_train_ages_path)
    X_validation_list_path = os.path.join(data_folder_name, X_validation_list_path)
    y_validation_ages_path = os.path.join(data_folder_name, y_validation_ages_path)

    return X_train_list_path, y_train_ages_path, X_validation_list_path, y_validation_ages_path


def select_test_datasets_path(data_parameters, mapping_evaluation_parameters):
    dataset_sex = data_parameters['dataset_sex']
    dataset_size = data_parameters['dataset_size']

    prediction_output_statistics_name = mapping_evaluation_parameters['prediction_output_statistics_name']

    if mapping_evaluation_parameters['dataset_type'] == 'validation':
        X_test_list_path = mapping_evaluation_parameters['male_validation']
        y_test_ages_path = mapping_evaluation_parameters['male_validation_age']
        prediction_output_statistics_name += '_validation.csv'
    elif mapping_evaluation_parameters['dataset_type'] == 'train':
        X_test_list_path = data_parameters['male_train']
        y_test_ages_path = data_parameters['male_train_age']
        prediction_output_statistics_name += '_train.csv'
    else:
        X_test_list_path = mapping_evaluation_parameters['male_test']
        y_test_ages_path = mapping_evaluation_parameters['male_test_age']
        prediction_output_statistics_name += '_test.csv'

    if dataset_sex == 'female':
        X_test_list_path = "fe" + X_test_list_path
        y_test_ages_path = "fe" + y_test_ages_path

    if dataset_size == 'small':
        X_test_list_path += "_small.txt"
        y_test_ages_path += "_small.npy"
    elif dataset_size == 'tiny':
        X_test_list_path += "_tiny.txt"
        y_test_ages_path += "_tiny.npy"
    elif 'small' in dataset_size and dataset_size!='small':
        X_test_list_path += "_small.txt"
        y_test_ages_path += "_small.npy"
    else:
        X_test_list_path += ".txt"
        y_test_ages_path += ".npy"

    if dataset_size == 'han':
        X_test_list_path = "test_han.txt"
        y_test_ages_path = "test_age_han.npy"

    if dataset_size == 'everything':
        X_test_list_path = "test_everything.txt"
        y_test_ages_path = "test_age_everything.npy"

    if dataset_size == 'testAB':
        X_test_list_path = 'male_testB.txt'
        y_test_ages_path = 'male_testB_age.npy'
        if dataset_sex == 'female':
            X_test_list_path = "fe" + X_test_list_path
            y_test_ages_path = "fe" + y_test_ages_path

    X_test_list_path = 'datasets/' + X_test_list_path
    y_test_ages_path = 'datasets/' + y_test_ages_path

    return X_test_list_path, y_test_ages_path, prediction_output_statistics_name



def get_datasets_dynamically(data_parameters):

    X_train_list_path, y_train_ages_path, X_validation_list_path, y_validation_ages_path = select_datasets_path(data_parameters)

    data_directory = data_parameters['data_directory']
    modality_flag = data_parameters['modality_flag']
    scaling_values_simple = pd.read_csv(data_parameters['scaling_values'], index_col=0)

    scale_factor = scaling_values_simple.loc[modality_flag].scale_factor
    resolution = scaling_values_simple.loc[modality_flag].resolution
    data_file = scaling_values_simple.loc[modality_flag].data_file

    modality_flag_split = modality_flag.rsplit('_', 1)
    if modality_flag_split[0] == 'rsfmri':
        rsfmri_volume = int(modality_flag_split[1])
    else:
        rsfmri_volume = None

    shift_transformation = data_parameters['shift_transformation']
    mirror_transformation = data_parameters['mirror_transformation']

    transformationFlag = data_parameters['transformation_flag']
    transformationMetric = data_parameters['transformation_metric']   
    fix_seed = data_parameters['fix_seed']

    X_train_paths, _ = load_file_paths(X_train_list_path, data_directory, data_file)
    X_validation_paths, _ = load_file_paths(X_validation_list_path, data_directory, data_file)
    y_train_ages = np.load(y_train_ages_path)
    y_validation_ages = np.load(y_validation_ages_path)

    print('****************************************************************')
    print("DATASET INFORMATION")
    print('====================')
    print("Modality Name: ", modality_flag)
    if rsfmri_volume != None:
        print("rsfMRI Volume: ", rsfmri_volume)
    print("Resolution: ", resolution)
    print("Scale Factor: ", scale_factor)
    print("Data File Path: ", data_file)
    print('****************************************************************')

    return (
        DynamicDataMapper( X_paths=X_train_paths, y_ages=y_train_ages, modality_flag=modality_flag, 
                            scale_factor=scale_factor, resolution=resolution, rsfmri_volume=rsfmri_volume,
                            shift_transformation=shift_transformation, mirror_transformation=mirror_transformation,
                            transformationFlag=transformationFlag, transformationMetric=transformationMetric, fix_seed=fix_seed
                            ),
        DynamicDataMapper( X_paths=X_validation_paths, y_ages=y_validation_ages, modality_flag=modality_flag, 
                            scale_factor=scale_factor, resolution=resolution, rsfmri_volume=rsfmri_volume,
                            shift_transformation=False, mirror_transformation=False,
                            transformationFlag=transformationFlag, transformationMetric=transformationMetric, fix_seed=fix_seed
                            ),
        resolution
    )


def get_test_datasets_dynamically(data_parameters, mapping_evaluation_parameters):

    X_test_list_path, y_test_ages_path, prediction_output_statistics_name = select_test_datasets_path(data_parameters, mapping_evaluation_parameters)

    data_directory = data_parameters['data_directory']
    modality_flag = data_parameters['modality_flag']
    scaling_values_simple = pd.read_csv(data_parameters['scaling_values'], index_col=0)

    scale_factor = scaling_values_simple.loc[modality_flag].scale_factor
    resolution = scaling_values_simple.loc[modality_flag].resolution
    data_file = scaling_values_simple.loc[modality_flag].data_file

    modality_flag_split = modality_flag.rsplit('_', 1)
    if modality_flag_split[0] == 'rsfmri':
        rsfmri_volume = int(modality_flag_split[1])
    else:
        rsfmri_volume = None

    transformationFlag = data_parameters['transformation_flag']
    transformationMetric = data_parameters['transformation_metric']
    fix_seed = data_parameters['fix_seed']

    X_test_paths, X_test_volumes_to_be_used = load_file_paths(X_test_list_path, data_directory, data_file)
    y_test_ages = np.load(y_test_ages_path)

    print('****************************************************************')
    print("DATASET INFORMATION")
    print('====================')
    print("Modality Name: ", modality_flag)
    if rsfmri_volume != None:
        print("rsfMRI Volume: ", rsfmri_volume)
    print("Resolution: ", resolution)
    print("Scale Factor: ", scale_factor)
    print("Data File Path: ", data_file)
    print('****************************************************************')

    return (
        DynamicDataMapper( X_paths=X_test_paths, y_ages=y_test_ages, modality_flag=modality_flag, 
                            scale_factor=scale_factor, resolution=resolution, rsfmri_volume=rsfmri_volume,
                            shift_transformation=False, mirror_transformation=False,
                            transformationFlag=transformationFlag, transformationMetric=transformationMetric, fix_seed=fix_seed
                            ),
        X_test_volumes_to_be_used,
        prediction_output_statistics_name,
        resolution
    )


def load_file_paths(data_list, data_directory, mapping_data_file):

    volumes_to_be_used = load_subjects_from_path(data_list)

    file_paths = [os.path.join(data_directory, volume, mapping_data_file) for volume in volumes_to_be_used]

    return file_paths, volumes_to_be_used 


def load_subjects_from_path(data_list):

    with open(data_list) as data_list_file:
        volumes_to_be_used = data_list_file.read().splitlines()

    return volumes_to_be_used


def load_and_preprocess_evaluation(file_path, modality_flag, resolution, scale_factor, rsfmri_volume=None):

    non_deterministic_modalities = ['T1_nonlinear', 'T1_linear', 'T2_nonlinear']

    if rsfmri_volume != None:
        volume = np.array(nib.load(file_path).dataobj)[:,:,:,rsfmri_volume]
    else:
        volume = np.array(nib.load(file_path).dataobj)

    if resolution == '2mm':
        crop_values = [5, 85, 6, 102, 0, 80]
    else:
        crop_values = [10, 170, 12, 204, 0, 160]

    volume = volume[crop_values[0]:crop_values[1],
                    crop_values[2]:crop_values[3], 
                    crop_values[4]:crop_values[5]]

    if modality_flag in non_deterministic_modalities:
        volume = volume / volume.mean()

    volume = volume / scale_factor

    volume = np.float32(volume)

    return volume


def num2vec(label, bin_range=[44,84], bin_step=1, std=1):

    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if bin_length % bin_step != 0:
        print("Error: Bin range should be divisible by the bin step!")
        return None
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + bin_step/2.0 + bin_step * np.arange(bin_number)
    
    if std == 0:
        # Uniform Distribution Case
        label = np.array(label)
        bin_values = np.floor((label - bin_start)/bin_step).astype(int)
    elif std < 0:
        print("Error! The standard deviation (& variance) must be positive")
        return None
    else:
        bin_values = np.zeros((bin_number))
        for i in range(bin_number):
            x1 = bin_centers[i] - bin_step/2.0
            x2 = bin_centers[i] + bin_step/2.0
            cdfs = norm.cdf([x1, x2], loc=label, scale=std)
            bin_values[i] = cdfs[1] - cdfs[0]       

    return bin_values, bin_centers