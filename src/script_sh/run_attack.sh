python main.py \
    --model_name resnet_vggface \
    --baseline GA \
    --fitness_type adaptive \
    --label 0


python main.py \
    --model_name resnet_vggface \
    --baseline GA \
    --fitness_type adaptive \
    --label 1


python main.py \
    --model_name resnet_vggface \
    --baseline NSGAII \
    --fitness_type normal \
    --label 0

python main.py \
    --model_name resnet_vggface \
    --baseline NSGAII \
    --fitness_type normal \
    --label 1

