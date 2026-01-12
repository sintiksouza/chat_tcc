#!/bin/bash
echo "######### Creating caddy network"
docker network inspect caddy --format {{.Id}} 2>/dev/null || docker network create --driver bridge caddy
echo ""
echo "######### Creating tieta network"
docker network inspect tieta --format {{.Id}} 2>/dev/null || docker network create --driver bridge tieta
echo ""
echo -e "\e[31mGPU NVIDIA: False. \e[39m. Creating machine WITHOUT GPU..."
echo "########## Erasing old image, if exists"
docker rmi tieta/sadai-chat:1.0.1 --force
echo ""
echo "######### Building new image"
docker build --tag tieta/sadai-chat:1.0.1 --file ./Dockerfile .

