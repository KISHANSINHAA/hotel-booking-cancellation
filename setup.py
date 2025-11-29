from setuptools import setup, find_packages

setup(
    name='hotel-booking-cancellation',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'joblib>=1.0.0',
        'streamlit>=1.0.0',
        'imbalanced-learn>=0.8.0'
    ],
)