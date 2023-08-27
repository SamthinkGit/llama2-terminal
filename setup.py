from setuptools import setup, find_packages

setup(
    name="llama2-terminal",
    version="0.1.5",
    packages=find_packages(),
    install_requires=[
        "inquirer",
        "cmd2",
        "farm-haystack",
        "torch",
        "transformers>=4.0.0",
        "bitsandbytes>=0.39.0",
        "pyyaml"
        "yaspin"
    ],
    include_package_data=True,
    author="SamthinkGit",
    author_email="sebastianmayorquin@gmail.com",
    description="Descripción breve del proyecto",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SamthinkGit/llama2-terminal",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
