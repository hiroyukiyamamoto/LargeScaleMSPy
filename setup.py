from setuptools import find_packages, setup


setup(
    name="LargeScaleMSPy",
    version="1.1.0",
    description="Large-scale MS/MS preprocessing with IDF-only PCA, UMAP, and visualization utilities",
    author="Hiroyuki Yamamoto",
    author_email="h.yama2396@gmail.com",
    url="https://github.com/hiroyukiyamamoto/LargeScaleMSPy",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "h5sparse",
        "pandas",
        "plotly",
        "dash",
        "umap-learn",
        "scikit-learn",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "umap-msms-viewer=LargeScaleMSPy.main:run_app",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
