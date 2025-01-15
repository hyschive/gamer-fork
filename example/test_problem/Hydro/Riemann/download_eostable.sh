#!/bin/bash

LOCAL_FILENAME="LS220.h5"
FILE_ID="678716b0999605c485c8ded3"

# 1. download
curl https://hub.yt/api/v1/item/${FILE_ID}/download -o "${LOCAL_FILENAME}"