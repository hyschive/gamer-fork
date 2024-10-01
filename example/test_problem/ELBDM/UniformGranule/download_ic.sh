filename=uniform-granule-ic

curl https://girder.hub.yt/api/v1/item/66e0028e32f323dee1b801dc/download -o ${filename}.tgz
tar -zxvf ${filename}.tgz
rm ${filename}.tgz
