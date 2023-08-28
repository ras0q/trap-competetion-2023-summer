#!/bin/bash

COMPETITION=$1

mkdir -p $COMPETITION/{input,temp,working}
kaggle competitions download -c $COMPETITION
unzip $COMPETITION.zip -d $COMPETITION/input
rm $COMPETITION.zip

touch $COMPETITION/working/main.py
