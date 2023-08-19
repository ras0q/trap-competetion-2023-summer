#!/bin/bash

COMPETITION=$1

mkdir -p $COMPETITION/{input,output,working}
kaggle competitions download -c $COMPETITION
unzip $COMPETITION.zip -d $COMPETITION/input
rm $COMPETITION.zip
