#!/bin/bash

docker rm -f $(docker ps -q -a -f 'label=com.openai.automanaged=true')
