from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setup(
    name="BayesMCDM",
    version="0.1.1.52",
    description="Bayesian Multi-Criteria Decision Making Toolkit",
    author="Majid Mohammadi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pystan",  
    ],
)   