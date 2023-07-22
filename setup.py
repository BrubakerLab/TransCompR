from setuptools import setup, find_packages

setup(
    name="TransCompR",
    version="0.2",
    packages=find_packages(),
    install_requires=[i.strip() for i in open("requirements.txt").readlines()],
    #package_data={
        #'TransCompR': ['model/*.h5ad', 'model/*.joblib','feature/*.xlsx'],
    #},
)