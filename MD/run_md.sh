#!/bin/bash
#SBATCH --job-name=CDF_MD
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=CDF_MD_%j.out
#SBATCH --error=CDF_MD_%j.err

set -euo pipefail

source /mnt/PublicSoftware/amber22/amber.sh

# =========================
# 文件路?
# =========================
BASE_DIR="/home/liyanrong/last/pdb1_pred_0_len334_0001"
MDIN_DIR="/home/liyanrong/new/mdin"

PRMTOP="$BASE_DIR/system.prmtop"
EQUILIBRATED_RST="$BASE_DIR/10eq.rst"
REF_FILE="$BASE_DIR/10eq.rst"
MDIN="$MDIN_DIR/11md.mdin"

RUN_DIR="$BASE_DIR/CDF_MD_${SLURM_JOB_ID}"

echo "========================================="
echo "CDF MD 三??立重复??"
echo "作??：$SLURM_JOB_ID"
echo "拓扑文件：$PRMTOP"
echo "平衡?构：$EQUILIBRATED_RST"
echo "?入文件：$MDIN"
echo "?出目?：$RUN_DIR"
echo "========================================="

# =========================
# 文件?查
# =========================
for file in "$PRMTOP" "$EQUILIBRATED_RST" "$REF_FILE" "$MDIN"; do
    if [[ ! -s "$file" ]]; then
        echo "??：文件不存在或?空：$file" >&2
        exit 1
    fi
done

command -v pmemd.cuda >/dev/null 2>&1 || {
    echo "??：找不到 pmemd.cuda" >&2
    exit 1
}

mkdir -p "$RUN_DIR"

# =========================
# 三??立生???
# =========================
for TRAJ in 1 2 3; do

    TRAJ_DIR="$RUN_DIR/trajectory_${TRAJ}"
    mkdir -p "$TRAJ_DIR"

    cp "$EQUILIBRATED_RST" "$TRAJ_DIR/initial.rst"

    # ?每???生成不同?机种子的?入文件
    TRAJ_MDIN="$TRAJ_DIR/11md_traj${TRAJ}.mdin"
    SEED=$((10000 + TRAJ))

    sed -E \
        -e "s/ig[[:space:]]*=[[:space:]]*[^,\/]+/ig=${SEED}/" \
        "$MDIN" > "$TRAJ_MDIN"

    echo "?始?? $TRAJ，?机种子：$SEED"

    pmemd.cuda -O \
        -i "$TRAJ_MDIN" \
        -o "$TRAJ_DIR/11md_traj${TRAJ}.out" \
        -p "$PRMTOP" \
        -c "$TRAJ_DIR/initial.rst" \
        -r "$TRAJ_DIR/11md_traj${TRAJ}.rst" \
        -x "$TRAJ_DIR/11md_traj${TRAJ}.mdcrd" \
        -ref "$REF_FILE" \
        -inf "$TRAJ_DIR/11md_traj${TRAJ}.inf"

    if [[ ! -s "$TRAJ_DIR/11md_traj${TRAJ}.rst" ]]; then
        echo "?? $TRAJ ?行失?，未生成 restart 文件" >&2
        exit 1
    fi

    echo "?? $TRAJ 完成"
done

echo "========================================="
echo "三???全部完成"
echo "?果目?：$RUN_DIR"
echo "========================================="