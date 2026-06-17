python main.py \
    --model_name restnet_vggface \
    --baseline GA \
    --fitness_type normal \
    --label 0 \
    --recons_w 0.0 \
    --attack_w 1.0 \
    --mutate_mode multiple_rectangles


python main.py \
    --model_name restnet_vggface \
    --baseline GA \
    --fitness_type normal \
    --label 1 \
    --recons_w 0.0 \
    --attack_w 1.0 \
    --mutate_mode multiple_rectangles


python main.py \
    --model_name restnet_vggface \
    --baseline GA \
    --fitness_type normal \
    --label 0 \
    --mutate_mode multiple_rectangles


python main.py \
    --model_name restnet_vggface \
    --baseline GA \
    --fitness_type normal \
    --label 1 \
    --mutate_mode multiple_rectangles



python main.py \
    --model_name restnet_vggface \
    --baseline GA \
    --fitness_type adaptive \
    --label 0 \
    --mutate_mode multiple_rectangles


python main.py \
    --model_name restnet_vggface \
    --baseline GA \
    --fitness_type adaptive \
    --label 1 \
    --mutate_mode multiple_rectangles



python main.py \
    --model_name restnet_vggface \
    --baseline NSGAII \
    --fitness_type normal \
    --label 0 \
    --mutate_mode multiple_rectangles

python main.py \
    --model_name restnet_vggface \
    --baseline NSGAII \
    --fitness_type normal \
    --label 1 \
    --mutate_mode multiple_rectangles

