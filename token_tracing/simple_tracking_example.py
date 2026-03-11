"""
Simple example of using the RoutingTracker with vLLM.

This demonstrates the cleanest way to track token routing without
modifying any vLLM source code.
"""

from vllm import LLM, SamplingParams
from routing_tracker import RoutingTracker

# Create your LLM as usual
llm = LLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    tensor_parallel_size=1,
    trust_remote_code=True,
)

# Create and register the routing tracker
tracker = RoutingTracker()
num_layers = tracker.register_hooks(llm)
print(f"Registered routing hooks on {num_layers} MoE layers")

# Define prompts and sampling params
prompts = [
    "Explain quantum computing in simple terms.",
    "What are the main differences between Python and JavaScript?",
    "Describe the water cycle.",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# Generate as usual - routing data is captured automatically
outputs = llm.generate(prompts, sampling_params)

# Print generated text
for output in outputs:
    print(f"Prompt: {output.prompt[:50]}...")
    print(f"Generated: {output.outputs[0].text}")
    print("-" * 80)

# Analyze routing data
print("\n" + "=" * 80)
print("ROUTING ANALYSIS")
print("=" * 80)

# Get statistics
stats = tracker.get_statistics()
print(f"\nCaptured data from {stats['num_layers']} layers")
print(f"Total captures: {stats['total_captures']}")

for layer_idx, layer_stats in stats['per_layer_stats'].items():
    print(f"\nLayer {layer_idx}:")
    print(f"  Unique experts used: {layer_stats['unique_experts_used']}")
    print(f"  Mean expert weight: {layer_stats['mean_expert_weight']:.4f}")
    print(f"  Load balance coefficient: {layer_stats['expert_load_balance_coefficient']:.4f}")
    
    # Show top 5 most used experts
    usage = layer_stats['expert_usage_counts']
    top_experts = sorted(usage.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top 5 experts:")
    for expert_id, count in top_experts:
        print(f"    Expert {expert_id}: {count} tokens")

# Convert to DataFrame for detailed analysis
df = tracker.to_dataframe()
print(f"\nDataFrame shape: {df.shape}")
print("\nSample routing decisions:")
print(df.head(10))

# Save to CSV
tracker.save_to_csv("routing_data.csv")

# Clean up
tracker.remove_hooks()
print("\nDone!")
