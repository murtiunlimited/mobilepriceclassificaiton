#!/bin/bash

# Pull latest image from Docker Hub
docker pull murtiunlimited/mobileprices:latest

# Stop and remove any running container named "mobileprices"
docker stop mobileprices || true
docker rm mobileprices || true

# Run Streamlit container on port 8501
docker run -d \
  -p 8501:8501 \
  --name mobileprices \
  murtiunlimited/mobileprices:latest
