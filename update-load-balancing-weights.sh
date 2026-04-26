#!/bin/bash

# ./update-load-balancing-weights.sh 0.5 0.25 0.25

curl -X POST http://host:port/load_balancer/weights \
  -H 'Content-Type: application/json' \
  -d '{
    "expert_affinity_routing_weight": $1,
    "kv_block_prefix_routing_weight": $2,
    "load_score_routing_weight": $3
  }'


curl -X GET http://host:port/load_balancer/weights