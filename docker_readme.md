# EEG Image Decode Project - Docker Setup

This document describes how to build and run the Docker container for the `eeg_image_decode` project. Using Docker ensures a consistent and reproducible environment with all necessary dependencies and code included.

## Purpose

The Docker image packages the project code along with its specific Conda environment (`BCI`). This allows you to run the project on any machine with Docker installed, without needing to manually set up the Conda environment or install dependencies.

## Prerequisites

*   [Docker](https://docs.docker.com/get-docker/) must be installed on your system.

## Building the Docker Image

1.  Navigate to the project's root directory (where the `Dockerfile` is located) in your terminal:
    ```bash
    cd /path/to/EEG_Image_decode_old
    ```
2.  Run the following command to build the Docker image. This might take some time, especially the first time, as it needs to download the base image and install all Conda packages.
    ```bash
    docker build -t eeg_image_decode:latest .
    ```
    *   `-t eeg_image_decode:latest`: Tags the image with the name `eeg_image_decode` and tag `latest`. You can change these if you prefer.
    *   `.`: Specifies that the build context (including the `Dockerfile`) is the current directory.

## Running the Docker Container

Once the image is built, you can run a container based on it:

```bash
docker run -it --rm eeg_image_decode:latest
```

*   `docker run`: Command to start a new container.
*   `-it`: Runs the container in interactive mode with a pseudo-TTY, allowing you to interact with the shell inside the container.
*   `--rm`: Automatically removes the container when it exits.
*   `eeg_image_decode:latest`: The name and tag of the image to run.

This command will drop you into a `bash` shell inside the container, within the `/app` directory. The `BCI` Conda environment is automatically activated, and all project code is available in `/app`.

## Working Inside the Container

*   **Environment:** The `BCI` Conda environment is active by default. You can directly run Python scripts or other commands that rely on packages installed in this environment.
*   **Code:** Your project code is located in the `/app` directory.
*   **(Optional) Volume Mounting:** If you need to access data from your host machine or save results from the container back to your host, use the `-v` flag to mount directories. For example, to mount a local `data` directory to `/app/data` inside the container:
    ```bash
    docker run -it --rm -v /path/on/host/data:/app/data eeg_image_decode:latest
    ```
    Replace `/path/on/host/data` with the actual path to your data directory on your computer. 