import torch

# Define the Actor network (adjust if your architecture is different)
class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(75, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 24),
        )
    def forward(self, x):
        return self.net(x)

# Load the checkpoint
ckpt = torch.load("model_6000.pt", map_location="cpu")
state_dict = ckpt["model_state_dict"]
# Map keys if needed (e.g., 'actor.0.weight' -> 'net.0.weight')
actor_state = {k.replace("actor.", "net."): v for k, v in state_dict.items() if "actor" in k}

# Instantiate and load weights
actor = Actor()
actor.load_state_dict(actor_state)
actor.eval()

# Dummy input for export
dummy_input = torch.zeros(1, 75)

# Export to ONNX
torch.onnx.export(
    actor,
    dummy_input,
    "walk_policy_new.onnx",
    input_names=["obs"],
    output_names=["actions"],
    opset_version=11,
    dynamic_axes={"obs": {0: "batch_size"}},
)

print("Exported to policy_15k.onnx")