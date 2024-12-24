# Use an official Python runtime as a parent image
FROM --platform=linux/amd64 python:3.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install rasterio geopandas fiona==1.9.6 numpy matplotlib opencv-python imgaug 

RUN pip install pyyaml
