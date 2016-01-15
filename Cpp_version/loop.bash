#!/bin/bash

 for file in /tmp/img_NT/FrontFace/*
do
  echo "Processing $file file..."
  #echo "$file"
  python faces.py "$file"
done
