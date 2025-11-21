#!/bin/bash

docker pull murtiunlimited/mobileprices:latest

docker stop mobileapp || true
docker rm mobileapp || true

docker run -d --name mobileapp -p 8501:8501 murtiunlimited/mobileprices:latest
