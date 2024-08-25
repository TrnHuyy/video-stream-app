# Use the official Python image from Docker Hub
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the necessary packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the Flask app
CMD ["python", "app.py"]
