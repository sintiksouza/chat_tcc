# Definição de arquitetura
ARG PLATFORM=linux/amd64
FROM --platform=$PLATFORM ubuntu:22.04

RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.9 python3-pip python3-dev
RUN pip3 -q install pip --upgrade
RUN apt-get install -y git
RUN apt-get install -y ffmpeg



RUN mkdir src
WORKDIR /src/

#COPY . .
COPY requirements-base.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8088


ENV TINI_VERSION=v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]


CMD ["bin/bash"]
#CMD ["jupyter", "notebook", "--port=8080", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
#CMD ["uvicorn", "app.main:app","--reload", "--host", "0.0.0.0", "--port", "8080"]


