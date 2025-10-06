# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/myhome/mambaforge/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/myhome/mambaforge/etc/profile.d/conda.sh" ]; then
        . "/myhome/mambaforge/etc/profile.d/conda.sh"
    else
        export PATH="/myhome/mambaforge/bin:$PATH"
    fi
fi
unset __conda_setup

# Activate the deepFlux conda environment
conda activate deepFlux
# <<< conda initialize <<<

