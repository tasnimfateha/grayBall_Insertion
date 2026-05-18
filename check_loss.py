from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = Path("runs")
event_files = list(RUNS_DIR.rglob("events.out.tfevents.*"))

if not event_files:
    print("No TensorBoard event files found in runs/")
    exit()

summary_rows = []
all_val_losses = []  # for overall average
all_steps = []       # for plotting
all_losses = []      # for plotting
epochs_global_offset = 0  # to separate runs on x-axis

for event_file in event_files:
    ea = EventAccumulator(str(event_file))
    ea.Reload()

    scalar_tags = ea.Tags().get("scalars", [])
    loss_tags = [tag for tag in scalar_tags if "loss" in tag.lower() and ("val" in tag.lower() or "valid" in tag.lower())]

    if not loss_tags:
        continue

    for tag in loss_tags:
        values = ea.Scalars(tag)
        if not values:
            continue

        # Track per-run loss
        run_losses = [v.value for v in values]
        run_steps = [v.step + epochs_global_offset for v in values]
        all_val_losses.extend(run_losses)
        all_steps.extend(run_steps)
        all_losses.extend(run_losses)

        # Best, final, average per run
        best = min(values, key=lambda x: x.value)
        final = values[-1]
        avg_val = sum(run_losses) / len(run_losses)

        summary_rows.append({
            "Run (Event File)": event_file.name,
            "Best Validation Loss": best.value,
            "Best Epoch": best.step,
            "Final Validation Loss": final.value,
            "Final Epoch": final.step,
            "Average Validation Loss": avg_val
        })

        epochs_global_offset += values[-1].step + 1  # offset next run steps

# Overall statistics
if all_val_losses:
    overall_avg = sum(all_val_losses) / len(all_val_losses)
    best_overall_loss = min(all_val_losses)
    best_overall_step = all_steps[all_losses.index(best_overall_loss)]

    summary_rows.append({
        "Run (Event File)": "Overall (Across Runs)",
        "Best Validation Loss": best_overall_loss,
        "Best Epoch": best_overall_step,
        "Final Validation Loss": "-",
        "Final Epoch": "-",
        "Average Validation Loss": overall_avg
    })

# Create and display table
df_summary = pd.DataFrame(summary_rows)
pd.set_option('display.float_format', '{:.6f}'.format)
print(df_summary)
df_summary.to_csv("validation_loss_summary.csv", index=False)

# Plot validation loss per epoch across all runs
plt.figure(figsize=(10,6))
plt.plot(all_steps, all_losses, label='Validation Loss', color='blue', linewidth=2)
plt.axhline(overall_avg, color='green', linestyle='--', label=f'Overall Average = {overall_avg:.4f}')
plt.scatter(best_overall_step, best_overall_loss, color='red', label=f'Best Loss = {best_overall_loss:.4f}')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss per Epoch Across All Runs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("validation_loss_plot.png")  # save figure
plt.show()