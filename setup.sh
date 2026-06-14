# dataset
wget 1sGtnHX3UYkg47lktNmz8dyQ8iu1hx_Ow
unzip lfw_preprocess.zip
rm lfw_preprocess.zip

# pretrained model
wget -O src/pretrained_model/vggface2.pt https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt
wget -O src/pretrained_model/webface.pt https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt




