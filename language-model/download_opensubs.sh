#!/bin/bash

mkdir -p data/opensubs
cd data/opensubs
wget -O en.tar.gz http://opus.lingfil.uu.se/download.php?f=OpenSubtitles/en.tar.gz
unzip en.tar.gz
rm en.tar.gz
