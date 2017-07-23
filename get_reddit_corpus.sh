#!/bin/bash

for YEAR in "2010" "2011" "2012" "2013" "2014" "2015"
do
  for MONTH in "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12"
  do
    if  [ ! -f  "reddit/RC_$YEAR-$MONTH.bz2" ]
    then
      wget http://files.pushshift.io/reddit/comments/RC_$YEAR-$MONTH.bz2 -P "reddit/"
    fi
  done
done
