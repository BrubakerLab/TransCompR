from setuptools import setup, find_packages

setup(
    name="TransCompR",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'seaborn',
        'matplotlib',
        'adjustText',
        'gseapy',
        'bio',
        'scikit-learn',
        'bayesian-optimization',
        'ruamel-yaml']
)