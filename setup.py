import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="farabio",
    version="0.0.2",
    author="San Askaruly",
    author_email="s.askaruly@gmail.com",
    description="farabio - Deep learning for biomedical imaging",
    license='Apache license',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tuttelikz/farabio",
    project_urls={
        "Bug Tracker": "https://github.com/tuttelikz/farabio/issues",
    },
    packages=setuptools.find_packages(),
    install_requires=[
        'docutils==0.16',
        'jupyterlab==3.1.6',
        'matplotlib==3.4.3',
        'numpy==1.21.1',
        'numpydoc==1.1.0',
        'pandas==1.3.1',
        'recommonmark==0.7.1',
        'sphinx-rtd-theme==0.5.2',
        'scikit-image==0.18.2',
        'scikit-learn==0.24.2',
        'sphinx==4.1.2',
        'sphinx-git==11.0.0',
        'torch==1.9.0+cu111',
        'torchvision==0.10.0+cu111',
        'torchaudio==0.9.0',
        'tqdm==4.62.1'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
