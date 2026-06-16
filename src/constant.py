IMG_DIR="../lfw_preprocess/lfw_preprocess/lfw_crop_margin_5"
PAIR_PATH="../lfw_preprocess/lfw_preprocess/pairs.txt"

RESNET_VGGFACE="pretrained_model/vggface2.pt"
RESNET_WEBFACE="pretrained_model/webface.pt"
ARCFACE_MS1MV3="pretrained_model/arcface.pth"
COSFACE_GLINT360K="pretrained_model/cosface.pth"

MODEL_RESIZE = {
    'restnet_vggface': 160,
    'restnet_webface': 160,
    'arcface_ms1mv3': 112,
    'cosface_glint360k': 112,
}


OUTPUT_DIR="output"

