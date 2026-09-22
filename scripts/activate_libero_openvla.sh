# Source this file before running LIBERO/OpenVLA.
export LIBERO_ENV=/home/leju-suzhou/zjt_ws/libero-openvla-env
export OPENVLA_SITE=/home/leju-suzhou/zjt_ws/openvla-ar-h100/lib/python3.10/site-packages
export LIBERO_REPO=/home/leju-suzhou/zjt_ws/openpi/third_party/libero
export OPENVLA_REPO=/home/leju-suzhou/zjt_ws/VLA-Pruner/src/openvla
export TOKEN_CD=/home/leju-suzhou/zjt_ws/token-cd
export PCD_SOURCE=/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export LIBERO_CONFIG_PATH=/home/leju-suzhou/zjt_ws/tmp/libero_cfg
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="$OPENVLA_SITE:$OPENVLA_REPO:$LIBERO_REPO:$PCD_SOURCE:$TOKEN_CD"
export PATH="$LIBERO_ENV/bin:$PATH"
