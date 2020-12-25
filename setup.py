import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="farabi", # Replace with your own username
    version="0.0.1",
    author="San Askaruly",
    author_email="s.askaruly@gmail.com",
    description="Random description here",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tuttelikz/farabi",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)