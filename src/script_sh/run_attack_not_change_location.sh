# ================================================ arestnet_vggface
python main.py \
    --model_name restnet_vggface \
    --baseline GA \
    --fitness_type adaptive \
    --label 0 \
    --prob_mutate_location 0.0


python main.py \
    --model_name restnet_vggface \
    --baseline GA \
    --fitness_type adaptive \
    --label 1 \
    --prob_mutate_location 0.0



python main.py \
    --model_name restnet_vggface \
    --baseline NSGAII \
    --fitness_type normal \
    --label 0 \
    --prob_mutate_location 0.0

python main.py \
    --model_name restnet_vggface \
    --baseline NSGAII \
    --fitness_type normal \
    --label 1 \
    --prob_mutate_location 0.0

python main.py \
    --model_name restnet_vggface \
    --baseline GA \
    --fitness_type normal \
    --label 0 \
    --recons_w 0.0 \
    --attack_w 1.0 \
    --prob_mutate_location 0.0


python main.py \
    --model_name restnet_vggface \
    --baseline GA \
    --fitness_type normal \
    --label 1 \
    --recons_w 0.0 \
    --attack_w 1.0 \
    --prob_mutate_location 0.0


python main.py \
    --model_name restnet_vggface \
    --baseline GA \
    --fitness_type normal \
    --label 0 \
    --prob_mutate_location 0.0


python main.py \
    --model_name restnet_vggface \
    --baseline GA \
    --fitness_type normal \
    --label 1 \
    --prob_mutate_location 0.0






# ================================================ arcface_ms1mv3

# python main.py \
#     --model_name arcface_ms1mv3 \
#     --baseline GA \
#     --fitness_type normal \
#     --label 0 \
#     --recons_w 0.0 \
#     --attack_w 1.0


# python main.py \
#     --model_name arcface_ms1mv3 \
#     --baseline GA \
#     --fitness_type normal \
#     --label 1 \
#     --recons_w 0.0 \
#     --attack_w 1.0


# python main.py \
#     --model_name arcface_ms1mv3 \
#     --baseline GA \
#     --fitness_type normal \
#     --label 0


# python main.py \
#     --model_name arcface_ms1mv3 \
#     --baseline GA \
#     --fitness_type normal \
#     --label 1



# python main.py \
#     --model_name arcface_ms1mv3 \
#     --baseline GA \
#     --fitness_type adaptive \
#     --label 0


# python main.py \
#     --model_name arcface_ms1mv3 \
#     --baseline GA \
#     --fitness_type adaptive \
#     --label 1



# python main.py \
#     --model_name arcface_ms1mv3 \
#     --baseline NSGAII \
#     --fitness_type normal \
#     --label 0

# python main.py \
#     --model_name arcface_ms1mv3 \
#     --baseline NSGAII \
#     --fitness_type normal \
#     --label 1

