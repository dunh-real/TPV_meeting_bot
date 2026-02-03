# Step 1: Start with an official Python base image.
# Using a 'slim' version is good practice for smaller image sizes.
FROM python:3.9-slim

# Step 2: Set the working directory inside the container.
# All subsequent commands will run from this directory.
WORKDIR /app

# Step 3: Copy the requirements file into the container.
COPY requirement.txt .

# Step 4: Install the Python dependencies.
RUN pip install --no-cache-dir -r requirement.txt

# Step 5: Copy your application code into the container.
COPY app.py .

# Step 6: Expose the port.
# This tells Docker that the container will listen on port 5000.
EXPOSE 5000

# Step 7: Define the command to run when the container starts.
CMD ["python", "app.py"]