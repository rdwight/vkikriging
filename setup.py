import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vkikriging",
    version="1.0",
    author="Richard Dwight",
    author_email="richard.dwight@gmail.com",
    description="Kriging and Gradient-Enhanced Kriging for VKI Lecture Series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rdwight/vkikriging",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
