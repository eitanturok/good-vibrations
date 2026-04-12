#!/bin/bash

srun --partition=normal.q --ntasks=1 --cpus-per-task=2 --mem=20G --gres=gpu:quadro_rtx_8000:1 --time=1:00:00 --pty bash
