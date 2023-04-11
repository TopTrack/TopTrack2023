mkdir ../../data/MOT20
cd ../../data/MOT20
wget https://motchallenge.net/data/MOT20.zip
unzip MOT20.zip
rm MOT20.zip
mkdir annotations
cd ../../src/tools/
python convert_mot20_to_coco.py