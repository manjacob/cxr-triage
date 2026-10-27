# Dockerfile
FROM nvcr.io/nvidia/pytorch:25.08-py3
 
WORKDIR /workspace
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY . .
 
CMD ["/bin/bash"]

