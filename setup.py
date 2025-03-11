from setuptools import setup, find_packages

def read_requirements() -> list:
    """
    Read the requirements from the requirements.txt file

    Returns:
        list: a list of requirements
    """
    with open("requirements.txt") as f:
        # filter out empty lines and lines starting with #
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return requirements

setup(
    name="ancogen",  # Name
    version="0.1",  # Version
    author="Samir Sadok",  # author
    author_email="samir.sadok@inria.fr",  # email
    description="AnCoGen: Analysis, Control and Generation of Speech with a Masked Autoencoder",  # description
    long_description=open("README.md").read(),  # README.md
    long_description_content_type="text/markdown",  # README
    url="https://github.com/samsad35/code-ancogen",  # URL repo
    packages=find_packages(),
    install_requires=read_requirements(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: AGPL-3.0",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    include_package_data=True,
)
