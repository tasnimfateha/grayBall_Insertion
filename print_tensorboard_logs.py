from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

log_dir = 'runs/cut_ball'  # Your TensorBoard log folder

# Find the event file
for file in os.listdir(log_dir):
    if file.startswith("events.out.tfevents"):
        event_file = os.path.join(log_dir, file)
        break

# Load the event file
event_acc = EventAccumulator(event_file)
event_acc.Reload()  # Load the event data

# List all scalar tags
print("Available tags:", event_acc.Tags()['scalars'])

# Read and print scalar values for each tag
for tag in event_acc.Tags()['scalars']:
    print(f"\nTag: {tag}")
    for scalar_event in event_acc.Scalars(tag):
        print(f"Step {scalar_event.step}, Value {scalar_event.value}")
