from setuptools import setup, find_packages

setup(
    name='BrainMapper',
    version='0.0.1',
    description='Mapping brain network activity from structural connectivity',
    license='BSD 3-clause license',
    maintainer='Andrei-Claudiu Roibu',
    maintainer_email='andrei-claudiu.roibu@dtc.ox.ac.uk',
    install_requires=[
        'numpy',
        'pandas',
        'torch==1.4',
        'fslpy',
        'tensorboardX',
        'sklearn',
        'nibabel',
        'h5py',
        'seaborn',
        'matplotlib',
        'scikit-learn',
        'scipy',
        ],
)

# NOTE: THIS NEEDS to be updated, as it is an old, out-of-date version. Need to update this by the end of the project!