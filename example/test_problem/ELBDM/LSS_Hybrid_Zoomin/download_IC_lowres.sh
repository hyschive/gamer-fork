API_URL="https://girder.hub.yt/api/v1"
FILE_ID="66ea79b4999605c485c8d623"
LOCAL_FILE="Zoomin_IC"

# download
girder-cli --api-url ${API_URL} download --parent-type item ${FILE_ID} ${LOCAL_FILE}

# unzip
tar zxvf ${LOCAL_FILE}/Zoomin_IC_lowres.tar.gz
rm -r ${LOCAL_FILE}
