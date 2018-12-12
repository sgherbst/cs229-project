# Introduction

This project is a tool for real-time video analysis of fruit flies, and is capable of determining fly position, orientation, sex, and wing angle (for male flies).

# Installation

After cloning this repository, navigate to the top-level directory, create a new conda environment and use **pip** to install the project and its dependencies:

```shell
> conda create --name=cs229-project python=3.6
> source activate cs229-project
> pip install -e .
```

If you choose a name other than **cs229-project**, please update the **PROJECT_NAME** variable in the toplevel Makefile accordingly.

Then download the the [input directory](https://www.dropbox.com/sh/78inyvw2ouut74a/AACc1DYrC1G0UxujwT-6ryRKa?dl=0) (2 GB) and place it in the top-level directory.  This folder, containing two subfolders **images** and **video**, contains the raw and labeled data used for this project.

# Running the examples

In the top-level directory, you can train the all four models used as part of the image processing pipeline by simply running **make**.  This should take about a minute.

After the models are trained, you can view the video processing in realtime by running

```shell
> make demo
```

To run the video processing on other video clips, navigate to the **cs229** directory and run 

```shell
> python demo.py -i testN.mp4
```

where **N** is replaced with a value from 1 to 5 (the default is 4).

Finally, to run a performance test (which processes the video but does not display the results live), return to the top-level directory and run:

```shell
> make profile
```
