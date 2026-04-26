for i in $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | awk '{print $1}' | awk -F ',' '{print $1}');do kill -9 $i;done
