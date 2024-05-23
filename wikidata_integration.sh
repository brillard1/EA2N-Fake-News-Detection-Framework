#!/bin/bash
dataset=$1

queue=()
m=10
for n in {1..50}; do
  queue+=("python3 prepare/extract_property_sample.py --train_data ./amr_data/amr_2.0_$dataset/csqa/train.pred.txt --amr_files ./amr_data/amr_2.0_$dataset/csqa/train.pred_$n.txt --nprocessors 2 --concept_seed AMR_CN_PRUNE > logs/log_poli_train_$n.txt")
done

for n in {1..10}; do
  queue+=("python3 prepare/extract_property_sample.py --train_data ./amr_data/amr_2.0_$dataset/csqa/dev.pred.txt --amr_files ./amr_data/amr_2.0_$dataset/csqa/dev.pred_$n.txt --nprocessors 2 --concept_seed AMR_CN_PRUNE > logs/log_poli_dev_$n.txt")
done

for n in {1..10}; do
  queue+=("python3 prepare/extract_property_sample.py --train_data ./amr_data/amr_2.0_$dataset/csqa/test.pred.txt --amr_files ./amr_data/amr_2.0_$dataset/csqa/test.pred_$n.txt --nprocessors 2 --concept_seed AMR_CN_PRUNE > logs/log_poli_test_$n.txt")
done

echo "Queue of length ${#queue[@]} created successfully"

# Initialize an array of tag tokens from 0 to m-1
tokens=($(seq 0 $((m-1))))

# Run the first m commands with the tag tokens
for i in {0..9}; do
  command=${queue[0]}
  tagToken=${tokens[$i]}
  # Add the tag token to the command before running it
  command=$(echo $command | sed "s/--concept_seed AMR_CN_PRUNE/--tagme_tokenID $tagToken &/")
  eval "$command" &
  # Store the pid of the last command in an array
  pids+=($!)
  # Store the pid-command mapping in an associative array
  declare -A pid_command_map
  pid_command_map[$!]=$command
  queue=("${queue[@]:1}")
done

while [ ${#pids[@]} -gt 0 ]; do
  for i in ${!pids[@]}; do
    pid=${pids[$i]}
    if ! kill -0 $pid 2>/dev/null; then # check if the process is still running
      status=$?
      if [ $status -eq 0 ]; then
        echo "Process $pid completed successfully"
      else
        echo "Process $pid failed"
      fi
      # Extract the tag token from the completed command by using the pid-command mapping
      command=${pid_command_map[$pid]}
      tagToken=$(echo $command | grep -oP "(?<=--tagme_tokenID )\d+")
      echo "The finished command $command with tagToken $tagToken"
      if [ ${#queue[@]} -gt 0 ]; then
        echo "${#queue[@]} commands left"
        command=${queue[0]}
        # Add the tag token to the next command before running it
        command=$(echo $command | sed "s/--concept_seed AMR_CN_PRUNE/--tagme_tokenID $tagToken &/")
        eval "$command" &
        pids[$i]=$!
        # Update the pid-command mapping with the new pid and command
        pid_command_map[$!]=$command
        queue=("${queue[@]:1}")
      else
        unset pids[$i]
      fi
    fi
  done
  sleep 1
done