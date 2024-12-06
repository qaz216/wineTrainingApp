FROM openjdk:11

LABEL maintainer="Aryeh Golob <ag645@njit.edu>"
LABEL version="0.1"

WORKDIR /app

RUN set -ex
RUN mkdir -p data 

COPY ./target/wine-training-app-1.0-SNAPSHOT.jar /app
COPY ./data/* data
