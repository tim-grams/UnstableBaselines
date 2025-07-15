export OPENROUTER_API_KEY=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 
export WANDB_API_KEY=f9f5c478759ffbe7bf4e9ade56fe85e2c5a2f7b2

apt-get update
apt install -y python3-pip
git clone https://github.com/tim-grams/UnstableBaselines.git
cd UnstableBaselines
pip install -r requirements.txt

python3 scripts/exploration.py --checkpoints_dir "/work/tgrams/selfplay/UnstableBaselines/outputs/2025-07-15/01-11-42/exploration-Qwen3-1.7B-Base-['Wordle-v0-train']-1752534684/checkpoints"