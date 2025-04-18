# Use an official Miniconda base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the environment file into the container
COPY environment.yml /app/environment.yml

# Create the Conda environment from the file
RUN conda env create -f environment.yml && \
    conda clean -afy # Clean up unused packages and caches

# Copy the rest of the project code into the container
COPY . /app

# Make Conda environment's binaries available in PATH
# Replace 'BCI' if your environment name in environment.yml is different
ENV PATH /opt/conda/envs/BCI/bin:$PATH

# Set the default command to execute when the container starts (e.g., open bash)
CMD ["/bin/bash"] 