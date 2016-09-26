#!/bin/bash

PAGE="www.cs.cornell.edu/~cristian/Echoes_of_power_files/"
FILE="wikipedia_conversations_corpus_v1.01"

if  [ ! -f "$FILE.zip" ]
then
  echo "Downloading corpus from $PAGE"
  wget $PAGE$FILE.zip
fi

echo "Unzipping corpus."
unzip -j "$FILE.zip" -d data/corpus/
