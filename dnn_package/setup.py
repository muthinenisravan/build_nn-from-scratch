from setuptools import find_packages, setup
from typing import List

HYPEN_E='-e .'
def get_requirments(file_path:str)->List[str]:
    '''
    this function will return list of requirments
    '''
    print (find_packages())
    requirments=list()
    with open(file_path) as file_obj:
        requirments=file_obj.readlines()
        requirments=[req.replace('\n','') for req in requirments]
    if HYPEN_E in requirments:
        requirments.remove(HYPEN_E)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="libdnn",
    version="1.2.0",
    author="Sravan Kumar Muthineni",
    author_email="muthinenisravan@gmail.com",
    description="A basic custom NN with L-hidden layers & activations",
    long_description=long_description,
    url="https://github.com/muthinenisravan/build_nn-from-scratch/nn_pack",
    packages=find_packages(),
    install_requires=get_requirments('requirments.txt')
)
