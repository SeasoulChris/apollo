#!/bin/bash

lib=$1
ver=$2

files=`find . -name "*${lib}*"`

for item in ${files[@]}
do
	echo "found ${item}"
	if [ -f "${item}.${ver}" ]; then 
		echo "${item}.${ver} already exists."
	else
		echo "create soft link to ${item}"
		ln -s ${item} ${item}.${ver}
	fi
done
