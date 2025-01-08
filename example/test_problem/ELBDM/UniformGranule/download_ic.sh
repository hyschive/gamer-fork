#!/bin/bash

LOCAL_FILENAME="uniform-granule"
FILE_ID="677cc1c3999605c485c8de80"

# 1. download
curl https://hub.yt/api/v1/item/${FILE_ID}/download -o "${LOCAL_FILENAME}.tar.gz"

# 2. unzip
tar -zxvf ${LOCAL_FILENAME}.tar.gz
rm ${LOCAL_FILENAME}.tar.gz

