import os
import gc
import sys
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from corner import corner
from bilby.gw.result import CBCResult
import pandas as pd
import paths

# Ensure JAX runs on CPU
os.environ["JAX_PLATFORMS"] = "cpu"
import jax.numpy as jnp

# Read event list and status
csv = pd.read_csv(paths.static / "event_status.csv")
events = csv["Event"].values

# Parse command-line arguments: number of processes and optional label
n_procs = int(sys.argv[1])
label = f"_{sys.argv[2]}" if len(sys.argv) >= 3 else ""

# Output directories
jim_outdir = paths.static / f"jim_runs/outdir{label}"
figure_outdir = paths.src / f"figures{label}"
os.makedirs(figure_outdir, exist_ok=True)

# Mapping for Bilby result keys
key_mapping = {
    "M_c": "chirp_mass",
    "q": "mass_ratio",
    "d_L": "luminosity_distance",
    "t_c": "geocent_time",
    "phase_c": "phase",
    "s1_mag": "a_1",
    "s1_theta": "tilt_1",
    "s1_phi": "phi_1",
    "s2_mag": "a_2",
    "s2_theta": "tilt_2",
    "s2_phi": "phi_2",
    "s1_z": "chi_1",
    "s2_z": "chi_2",
    "log_L": "log_likelihood",
}

# Number of samples to plot
n_samples = 5000


# Worker function for each event
def process_event(event):
    try:
        # Skip if final figure exists
        target_file = figure_outdir / f"{event}.jpg"
        if target_file.exists():
            print(f"Skipping {event}, figure already exists.")
            return (event, None)

        print(f"Processing {event}")

        # Load Jim samples and log-probabilities
        jim_results_file = paths.static / f"{jim_outdir}/{event}/samples.npz"
        result_jim = jnp.load(jim_results_file, allow_pickle=True)
        try:
            keys = ["M_c", "q", "s1_mag", "s1_theta", "s1_phi", "s2_mag", "s2_theta", "s2_phi", "iota", "d_L", "ra", "dec", "t_c", "phase_c", "psi", "log_L"]
            samples_jim_list = [result_jim[k] for k in keys]
        except KeyError:
            keys = ["M_c", "q", "a_1", "a_2", "tilt_1", "tilt_2", "phi_jl", "phi_12", "theta_jn", "d_L", "ra", "dec", "t_c", "phase_c", "psi", "log_L"]
            samples_jim_list = [result_jim[k] for k in keys]
        del result_jim
        gc.collect()
        samples_jim = np.array(samples_jim_list).T
        del samples_jim_list
        gc.collect()
        if samples_jim.shape[0] > n_samples:
            idx = np.random.choice(samples_jim.shape[0], n_samples, replace=False)
            samples_jim = samples_jim[idx]

        # Load Bilby samples
        bilby_dir = paths.static / f"bilby_runs/outdir/{event}/final_result"
        files = os.listdir(bilby_dir)
        if len(files) != 1:
            print(f"Error: {event} does not have a unique result file")
            return (event, "error")
        res_file = bilby_dir / files[0]
        res_bilby = CBCResult.from_hdf5(res_file)
        trigger_time = float(res_bilby.meta_data["command_line_args"]["trigger_time"])
        df = res_bilby.posterior
        del res_bilby
        gc.collect()
        df["geocent_time"] -= trigger_time
        if len(df) > n_samples:
            df = df.sample(n_samples)
        else:
            print(f"Warning: {event} has only {len(df)} samples")
        bilby_list = [df[key_mapping.get(k, k)].values for k in keys]
        del df
        gc.collect()
        samples_bilby = np.array(bilby_list).T
        del bilby_list
        gc.collect()

        # Plot comparison with logL
        try:
            fig = corner(samples_jim, labels=keys, color="blue", hist_kwargs={"density": True})
            corner(samples_bilby, labels=keys, fig=fig, color="red", hist_kwargs={"density": True},)
            fig.legend(["jim", "bilby"], loc="right", fontsize=20)
            fig.savefig(target_file)
            plt.close(fig)
        except Exception as e:
            print(f"Plot 2 error for {event}: {e}")
        del samples_jim, samples_bilby
        gc.collect()

        return (event, "good")
    
    except Exception as e:
        print(f"Error processing {event}: {e}")
        return (event, "error")


if __name__ == "__main__":
    # Parallel processing of events
    with multiprocessing.Pool(processes=n_procs) as pool:
        results = pool.map(process_event, events)

    # Update CSV statuses
    for event, status in results:
        if status == "good":
            csv.loc[csv["Event"] == event, "Comparison"] = "good"
        elif status == "error":
            csv.loc[csv["Event"] == event, "Comparison"] = "error"
        # None status means skipped; leave unchanged
    csv.to_csv(paths.static / "event_status.csv", index=False)
    print("All done.")
