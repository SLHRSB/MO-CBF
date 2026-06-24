from src.cbf import MOCBF, CBFInput

cbf = MOCBF()

example_input = CBFInput(
    throttle_rl=0.8,
    brake_rl=0.0,
    leader_distance=15.0,
    ego_speed=12.0,
    leader_speed=8.0,
    current_rpm=2500.0,
)

output = cbf.filter_action(example_input)

print("Safe action:", output.safe_action)
print("Intervention:", output.intervention_type.value)
print("Barrier value:", output.barrier_value)
