#!/bin/bash
clear
echo "================================================================================"
echo "          MARK-SLM-3M: LOCAL TRAINING MONITOR (RTX 2060 SUPER)                  "
echo "================================================================================"
tail -n 30 -f ./scripts/train_output.log
