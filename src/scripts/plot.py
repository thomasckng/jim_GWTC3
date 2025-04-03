import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import os
import gc

os.environ['JAX_PLATFORMS'] = 'cpu'
import jax.numpy as jnp
from bilby.gw.result import CBCResult
import pandas as pd
import paths
import sys

csv = pd.read_csv(paths.static / 'event_status.csv')
events = csv['Event'].values

if len(sys.argv) < 2:
    label = ''
else:
    label = f'_{sys.argv[1]}'

jim_outdir = paths.static / f'jim_runs/outdir{label}'
figure_outdir = paths.src / f'figures{label}'

os.makedirs(figure_outdir, exist_ok=True)

# Define key mapping for Bilby results
key_mapping = {
    'M_c': 'chirp_mass',
    'q': 'mass_ratio',
    'd_L': 'luminosity_distance',
    't_c': 'geocent_time',
    'phase_c': 'phase',
    's1_mag': 'a_1',
    's1_theta': 'tilt_1',
    's1_phi': 'phi_1',
    's2_mag': 'a_2',
    's2_theta': 'tilt_2',
    's2_phi': 'phi_2',
    's1_z': 'chi_1',
    's2_z': 'chi_2'
}

for event in events:
    try:
        # Skip if figure already exists.
        if os.path.exists(paths.src / f'figures/{event}.jpg'):
            continue

        print(f'Plotting {event}')
        n_samples = 5000

        # Define keys based on event type.
        keys = ['M_c', 'q', 'd_L', 'iota', 'ra', 'dec', 't_c', 'psi', 'phase_c']
        if event[-2:] == '_D':
            keys += ['s1_z', 's2_z']
        else:
            keys += ['s1_mag', 's1_theta', 's1_phi', 's2_mag', 's2_theta', 's2_phi']

        # ---- Load production samples from Jim ----
        jim_samples_file = paths.static / f'{jim_outdir}/{event}/samples.npz'
        result_jim = jnp.load(jim_samples_file)
        samples_jim_list = [result_jim[key] for key in keys]

        # Load the result file only once for log-probabilities.
        jim_result_file = paths.static / f'{jim_outdir}/{event}/result.npz'
        result_jim_result = jnp.load(jim_result_file)
        log_posterior = result_jim_result['log_prob']
        log_prior = result_jim_result['log_prior']

        samples_jim_list.append(log_posterior - log_prior)
        samples_jim = np.array(samples_jim_list).T
        if samples_jim.shape[0] > n_samples:
            samples_jim = samples_jim[np.random.choice(samples_jim.shape[0], n_samples, replace=False), :]

        # ---- Load NF samples from Jim (if available) ----
        samples_jim_nf = None
        nf_file = paths.static / f'{jim_outdir}/{event}/nf_samples.npz'
        if os.path.exists(nf_file):
            result_jim_nf = jnp.load(nf_file)
            samples_jim_nf_list = [result_jim_nf[key] for key in keys]
            samples_jim_nf = np.array(samples_jim_nf_list).T
            if samples_jim_nf.shape[0] > n_samples:
                samples_jim_nf = samples_jim_nf[np.random.choice(samples_jim_nf.shape[0], n_samples, replace=False), :]

        # ---- Load samples from Bilby ----
        bilby_dir = paths.static / f'bilby_runs/outdir/{event}/final_result'
        files = os.listdir(bilby_dir)
        if len(files) != 1:
            print(f'Error: {event} does not have a unique result file')
            continue
        file = files[0]
        bilby_file = bilby_dir / file
        result_bilby = CBCResult.from_hdf5(bilby_file)
        trigger_time = float(result_bilby.meta_data['command_line_args']['trigger_time'])
        result_bilby = result_bilby.posterior
        result_bilby['geocent_time'] -= trigger_time
        if len(result_bilby) > n_samples:
            result_bilby = result_bilby.sample(n_samples)
        else:
            print(f'Warning: {event} has only {len(result_bilby)} samples')
        samples_bilby_list = []
        for key in keys:
            mapped_key = key_mapping.get(key, key)  # Map keys if a replacement is defined.
            samples_bilby_list.append(result_bilby[mapped_key].values)
        samples_bilby_list.append(result_bilby['log_likelihood'].values)
        samples_bilby = np.array(samples_bilby_list).T

        # ---- Plotting ----
        try:
            # First plot: compare jim, jim_nf (if available), and bilby (without logL).
            fig_1 = corner(samples_jim[:, :-1], labels=keys, color='blue', hist_kwargs={'density': True})
            if samples_jim_nf is not None:
                corner(samples_jim_nf, labels=keys, fig=fig_1, color='green', hist_kwargs={'density': True})
                fig_1.legend(['jim', 'jim_nf', 'bilby'], loc='right', fontsize=20)
            else:
                fig_1.legend(['jim', 'bilby'], loc='right', fontsize=20)
            corner(samples_bilby[:, :-1], labels=keys, fig=fig_1, color='red', hist_kwargs={'density': True})
            fig_1.savefig(figure_outdir / f'{event}_nf.jpg')
            plt.close(fig_1)
        except Exception as e:
            print(f'Plotting error (plot 1) for {event}: {e}')

        try:
            # Second plot: add logL comparison.
            keys_with_logL = keys + ['logL']
            fig_2 = corner(samples_jim, labels=keys_with_logL, color='blue', hist_kwargs={'density': True})
            corner(samples_bilby, labels=keys_with_logL, fig=fig_2, color='red', hist_kwargs={'density': True})
            fig_2.legend(['jim', 'bilby'], loc='right', fontsize=20)
            fig_2.savefig(figure_outdir / f'{event}.jpg')
            plt.close(fig_2)
        except Exception as e:
            print(f'Plotting error (plot 2) for {event}: {e}')

        # ---- Update CSV ----
        csv.loc[csv['Event'] == event, 'Comparison'] = 'good'
        csv.to_csv(paths.static / 'event_status.csv', index=False)

        # ---- Clean up memory ----
        del result_jim, result_jim_result, samples_jim, samples_bilby
        if 'result_jim_nf' in locals():
            del result_jim_nf, samples_jim_nf
        del result_bilby
        gc.collect()

    except Exception as e:
        print(f'Error processing {event}: {e}')
        continue
