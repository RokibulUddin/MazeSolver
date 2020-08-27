import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MazeSolver-RokibulUddin",
    version="0.0.1",
    author="Rokibul Uddin",
    author_email="riki.rokibul@gmail.com",
    description="A* maze solver and maze creator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RokibulUddin/MazeSolver",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)