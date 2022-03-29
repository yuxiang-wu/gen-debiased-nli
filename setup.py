from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [req for req in f.readlines() if not req.startswith("#")]

setup(
    name="gen_debiased_nli",
    version="0.1.0",
    author="Yuxiang Wu",
    author_email="yuxiang.cs@gmail.com",
    description="Generating Debiased NLI Datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jimmycode/gen-debiased-nli",
    packages=find_packages(),
    classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
)
