#!/bin/bash

PAGE="www.cs.cornell.edu/~cristian/Echoes_of_power_files/"
FILE="wikipedia_conversations_corpus_v1.01"

if  [ ! -f "./wiki/$FILE.zip" ]
then
  echo "Downloading corpus from $PAGE"
  wget $PAGE$FILE.zip -P data/wiki/
fi

echo "Unzipping corpus."
unzip -j "./wiki/$FILE.zip" -d ./wiki/
