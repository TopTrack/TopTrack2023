mkdir ../../data/mot17
cd ../../data/mot17
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip
rm MOT17.zip
mkdir annotations
cd ../../src/tools/
python convert_mot_to_coco.py