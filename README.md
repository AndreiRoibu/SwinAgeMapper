# SwinAgeMapper


This project looks at employing different neuroimaging modalities to understand patterns of brain ageging and factors influencing it. It represents the first ever use of SWIN Transformers for this task. For any issues, please contact Andrei Roibu at andrei-claudiu.roibu@dtc.ox.ac.uk. 

The code is still undergoing modifications and further additions will be made in the coming period.

This work is a continuation of the work conducted in AgeMapper: https://github.com/AndreiRoibu/AgeMapper

## Motivation (From PhD Thesis)

This repository houses the code and methodologies from the final chapter of my PhD thesis, which introduces a novel approach in predicting brain age using Shifted Window (SWIN) Transformers. The study explores the limitations of traditional Convolutional Neural Networks (CNNs) in medical imaging, particularly in modeling contextual information crucial for understanding brain ageing processes. 

In this repo, I present the Brain Age SWIN (BA-SWIN) network, a specialized SWIN-based architecture designed for brain age prediction. Unlike CNNs, BA-SWIN utilizes a unique approach of partitioning images into multiple non-overlapping patches, each independently processed with attention mechanisms. This method not only enhances performance but also provides a level of explainability previously unattainable in CNNs. The BA-SWIN networks were tested against CNNs in brain-age prediction tasks, showing comparable accuracy but excelling in identifying novel associations to non-imaging derived phenotypes (nIDPs) for certain contrasts. Additionally, the SWIN networks demonstrated superior resilience in handling corrupted data, especially when pretrained on high-quality datasets like UK Biobank.

## Network Architecture & Pre-Trained Networks


## Subject Datasets and Contrast Information

To access the datasets utilised for training these networks, see the various text and numpy files in the __datasets__ folder in this repository. The actual MRI scans are available upon application from the [UK Biobank](https://www.ukbiobank.ac.uk), such as all the other data utilised in this project. 

In the __datasets__ there is also a file named __scaling_values_simple.csv__. This CSV file contains information on the name of each of the contrasts, the scale factor utilised during data pre-processing, the resolution of the MRI files, and the internal UK Biobank file handle marking where the file can be found for each subject.

## Correlations to UK Biobank nIDPs

One of the major findings of this work has been that all contrasts correlate significantly and differently with a large number of nIDPs from the UK Biobank. The correlations and the information for accessing the data is made freely available for the research community to further investigate. All the correlations can be accessed at on ***LINK TO BE ADDED LATER*** This collection contains both the full correlation, very large files, as well as smaller files, containing only the statistically significant associations. 

## Installation & Usage
To download and run this code as python files, you can clone this repository using git:

```bash
git clone <link to repo>
```

In order to install the required packages, the user will need the presence of Python3 and the [pip3](https://pip.pypa.io/en/stable/) installer. 

For installation on Linux or OSX, use the following commands. This will create a virtual environment and automatically install all the requirements, as well as create the required metadata

```bash
./setup.sh
```

In order to run the code, activate the previously installed virtual environment, and utilise the run file. Several steps are needed prior to that:
* make a copy of the __settings.ini__ and __settings_eval.ini__ files, filling out the required settings. If running an evaluation, make sure that the pre-trained network name corresponds to the experiment names
* rename the two __ini__ files to either the pre-trained network name, or to something else

This approach has been used given the large number of hyperparameters and network-subject sex-MRI modality combinations.

After setting up, activate your environment using the following code:

```bash
~/(usr)$ source env/bin/activate
```

For running network training epochs, utilise this code, setting TASK='train' (or test), NAME='name_of_your_ini_files', CHECKPOINT='0' (or some other value if wishing to start from a later checkpoint), and EPOCHS='number_of_epochs_you_want_to_train_for'. For more details on these inputs, see the __run.py__ file.

```bash
~/(usr)$ python run.py -m ${TASK} -n ${NAME} -c ${CHECKPOINT} -e ${EPOCHS}
```


## References

The work presented in this repository is under consideration for publication. In place of a paper, please cite the below paper that the author has published at the 2023 10th Swiss Conferece on Data Science (SDS). The paper can be accessed at on [IEEE](https://ieeexplore.ieee.org/abstract/document/10196736). To reference this work, please use the following citation. This paper reference prior work the author has done on the topic of brain ageing, and relates to the AgeMapper repository.

```
@inproceedings{roibu2023brain,
  title={Brain Ages Derived from Different MRI Modalities are Associated with Distinct Biological Phenotypes},
  author={Roibu, Andrei-Claudiu and Adaszewski, Stanislaw and Schindler, Torsten and Smith, Stephen M and Namburete, Ana IL and Lange, Frederik J},
  booktitle={2023 10th IEEE Swiss Conference on Data Science (SDS)},
  pages={17--25},
  year={2023},
  organization={IEEE},
  doi={10.1109/SDS57534.2023.00010}}
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Licence
[BSD 3-Clause Licence](https://opensource.org/licenses/BSD-3-Clause) © [Andrei Roibu](https://github.com/AndreiRoibu)
