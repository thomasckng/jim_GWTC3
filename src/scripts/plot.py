import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import os
import gc
import sys
import multiprocessing

# Ensure JAX runs on CPU
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax.numpy as jnp
from bilby.gw.result import CBCResult
import pandas as pd
import paths

# Read event list and status
csv = pd.read_csv(paths.static / 'event_status.csv')
events = csv['Event'].values

# Parse command-line arguments: number of processes and optional label
n_procs = int(sys.argv[1])
label = f'_{sys.argv[2]}' if len(sys.argv) >= 3 else ''

# Output directories
jim_outdir = paths.static / f'jim_runs/outdir{label}'
figure_outdir = paths.src / f'figures{label}'
os.makedirs(figure_outdir, exist_ok=True)

# Mapping for Bilby result keys
key_mapping = {
    'M_c': 'chirp_mass', 'q': 'mass_ratio', 'd_L': 'luminosity_distance',
    't_c': 'geocent_time', 'phase_c': 'phase', 's1_mag': 'a_1',
    's1_theta': 'tilt_1', 's1_phi': 'phi_1', 's2_mag': 'a_2',
    's2_theta': 'tilt_2', 's2_phi': 'phi_2', 's1_z': 'chi_1', 's2_z': 'chi_2'
}

# Number of samples to plot
n_samples = 5000

# Worker function for each event
def process_event(event):
    try:
        # Skip if final figure exists
        target_file = figure_outdir / f'{event}.jpg'
        if target_file.exists():
            print(f'Skipping {event}, figure already exists.')
            return (event, None)

        print(f'Processing {event}')
        # Determine keys based on event type
        keys = ['M_c', 'q', 'd_L', 'iota', 'ra', 'dec', 't_c', 'psi', 'phase_c']
        if event.endswith('_D'):
            keys += ['s1_z', 's2_z']
        else:
            keys += ['s1_mag', 's1_theta', 's1_phi', 's2_mag', 's2_theta', 's2_phi']

        # Load Jim samples
        jim_samples_file = paths.static / f'{jim_outdir}/{event}/samples.npz'
        result_jim = jnp.load(jim_samples_file)
        samples_jim_list = [result_jim[k] for k in keys]

        # Load Jim log-probabilities
        jim_result_file = paths.static / f'{jim_outdir}/{event}/result.npz'
        result_jim_res = jnp.load(jim_result_file)
        log_posterior = result_jim_res['log_prob']
        log_prior = result_jim_res['log_prior']
        samples_jim_list.append(log_posterior - log_prior)
        samples_jim = np.array(samples_jim_list).T
        if samples_jim.shape[0] > n_samples:
            idx = np.random.choice(samples_jim.shape[0], n_samples, replace=False)
            samples_jim = samples_jim[idx]

        # Load NF samples if available
        nf_file = paths.static / f'{jim_outdir}/{event}/nf_samples.npz'
        samples_jim_nf = None
        if os.path.exists(nf_file):
            res_nf = jnp.load(nf_file)
            lst_nf = [res_nf[k] for k in keys]
            samples_jim_nf = np.array(lst_nf).T
            if samples_jim_nf.shape[0] > n_samples:
                idx_nf = np.random.choice(samples_jim_nf.shape[0], n_samples, replace=False)
                samples_jim_nf = samples_jim_nf[idx_nf]

        # Load Bilby samples
        bilby_dir = paths.static / f'bilby_runs/outdir/{event}/final_result'
        files = os.listdir(bilby_dir)
        if len(files) != 1:
            print(f'Error: {event} does not have a unique result file')
            return (event, 'error')
        res_file = bilby_dir / files[0]
        res_bilby = CBCResult.from_hdf5(res_file)
        trigger_time = float(res_bilby.meta_data['command_line_args']['trigger_time'])
        df = res_bilby.posterior
        df['geocent_time'] -= trigger_time
        if len(df) > n_samples:
            df = df.sample(n_samples)
        else:
            print(f'Warning: {event} has only {len(df)} samples')
        bilby_list = [df[key_mapping.get(k, k)].values for k in keys]
        bilby_list.append(df['log_likelihood'].values)
        samples_bilby = np.array(bilby_list).T

        # Plot comparison without logL
        try:
            fig1 = corner(samples_jim[:, :-1], labels=keys, color='blue', hist_kwargs={'density': True})
            if samples_jim_nf is not None:
                corner(samples_jim_nf, labels=keys, fig=fig1, color='green', hist_kwargs={'density': True})
                fig1.legend(['jim', 'jim_nf', 'bilby'], loc='right', fontsize=20)
            else:
                fig1.legend(['jim', 'bilby'], loc='right', fontsize=20)
            corner(samples_bilby[:, :-1], labels=keys, fig=fig1, color='red', hist_kwargs={'density': True})
            fig1.savefig(figure_outdir / f'{event}_nf.jpg')
            plt.close(fig1)
        except Exception as e:
            print(f'Plot 1 error for {event}: {e}')

        # Plot comparison with logL
        try:
            keys_l = keys + ['logL']
            fig2 = corner(samples_jim, labels=keys_l, color='blue', hist_kwargs={'density': True})
            corner(samples_bilby, labels=keys_l, fig=fig2, color='red', hist_kwargs={'density': True})
            fig2.legend(['jim', 'bilby'], loc='right', fontsize=20)
            fig2.savefig(target_file)
            plt.close(fig2)
        except Exception as e:
            print(f'Plot 2 error for {event}: {e}')

        # Clean up
        del result_jim, result_jim_res, samples_jim, samples_bilby, df
        if samples_jim_nf is not None:
            del res_nf, samples_jim_nf
        gc.collect()

        return (event, 'good')
    except Exception as e:
        print(f'Error processing {event}: {e}')
        return (event, 'error')


if __name__ == '__main__':
    # Parallel processing of events
    with multiprocessing.Pool(processes=n_procs) as pool:
        results = pool.map(process_event, events)

    # Update CSV statuses
    for event, status in results:
        if status == 'good':
            csv.loc[csv['Event'] == event, 'Comparison'] = 'good'
        elif status == 'error':
            csv.loc[csv['Event'] == event, 'Comparison'] = 'error'
        # None status means skipped; leave unchanged
    csv.to_csv(paths.static / 'event_status.csv', index=False)
    print('All done.')