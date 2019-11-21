#!/bin/bash

declare -a default_args=("LD_LIBRARY_PATH" "LANG" "HOSTNAME"
                         "NVIDIA_VISIBLE_DEVICES" "PWD" "HOME"
                         "TERM" "CUDA_PKG_VERSION" "CUDA_VERSION"
                         "NVIDIA_DRIVER_CAPABILITIES" "SHLVL" "NVIDIA_REQUIRE_CUDA"
                         "PATH" "_" "HOST_OS")

unset IFS
args=""
for var in $(compgen -e); do
    if [[ ! " ${default_args[@]} " =~ " ${var} " ]]; then
        var_lower="${var,,}"
        args+="--$var_lower \"${!var}\" "
    fi
done

eval "python run_rnnt.py $args"