#!/bin/bash

DIR=$(pwd)

while inotifywait -r -e modify $DIR; do
    bibtex report
    make 
done
