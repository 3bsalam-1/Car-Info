FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Run the application using the new structure
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
