import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax.numpy as jnp
from bilby.gw.result import CBCResult
import pandas as pd
import paths
import sys

csv = pd.read_csv(paths.static/'event_status.csv')
events = csv['Event'].values

if len(sys.argv) < 2:
    label = ''
else:
    label = f'_{sys.argv[1]}'

jim_outdir = paths.static/f'jim_runs/outdir{label}'
figure_outdir = paths.src/f'figures{label}'

os.makedirs(figure_outdir, exist_ok=True)

for event in events:
    try:
        if not os.path.exists(paths.src/f'figures/{event}.jpg'):
            print(f'Plotting {event}')
            n_samples = 5000

            keys = ['M_c', 'q', 'd_L', 'iota', 'ra', 'dec', 't_c', 'psi', 'phase_c']
            if event[-2:] == '_D':
                keys += ['s1_z', 's2_z']
            else:
                keys += ['s1_mag', 's1_theta', 's1_phi', 's2_mag', 's2_theta', 's2_phi']

            # Load production samples from jim
            result_jim = jnp.load(paths.static/f'{jim_outdir}/{event}/samples.npz')
            samples_jim = []
            for key in keys:
                samples_jim.append(result_jim[key])
            try:
                log_posterior = jnp.load(paths.static/f'{jim_outdir}/{event}/result.npz')['log_prob']
                log_prior = jnp.load(paths.static/f'{jim_outdir}/{event}/result.npz')['log_prior']
            except:
                try:
                    log_posterior = jnp.load(paths.static/f'{jim_outdir}/{event}/result.npz')['log_prob'].reshape(-1)
                    log_prior = jnp.load(paths.static/f'{jim_outdir}/{event}/log_prior.npz')['arr_0']
                except:
                    log_posterior = jnp.load(paths.static/f'{jim_outdir}/{event}/result.npz')['log_prob'].reshape(-1)
                    log_prior = jnp.load(paths.static/f'{jim_outdir}/{event}/log_prior.npz')['log_prior']
            samples_jim.append(log_posterior-log_prior)
            samples_jim = np.array(samples_jim).T
            if len(samples_jim) > n_samples:
                samples_jim = samples_jim[np.random.choice(samples_jim.shape[0], n_samples), :]

            # Load NF samples from jim
            try:
                result_jim_nf = jnp.load(paths.static/f'{jim_outdir}/{event}/nf_samples.npz')
                samples_jim_nf = []
                for key in keys:
                    samples_jim_nf.append(result_jim_nf[key])
                samples_jim_nf = np.array(samples_jim_nf).T
                if len(samples_jim_nf) > n_samples:
                    samples_jim_nf = samples_jim_nf[np.random.choice(samples_jim_nf.shape[0], n_samples), :]
            except:
                pass

            # Load samples from bilby
            files = os.listdir(paths.static/f'bilby_runs/outdir/{event}/final_result')
            if len(files) != 1:
                print(f'Error: {event} does not have a unique result file')
                continue
            else:
                file = files[0]
            result_bilby = CBCResult.from_hdf5(paths.static/f'bilby_runs/outdir/{event}/final_result/{file}')
            trigger_time = float(result_bilby.meta_data['command_line_args']['trigger_time'])
            result_bilby = result_bilby.posterior
            result_bilby['geocent_time'] -= trigger_time
            if len(result_bilby) > n_samples:
                result_bilby = result_bilby.sample(n_samples)
            else:
                print(f'Warning: {event} has only {len(result_bilby)} samples')
            samples_bilby = []
            for key in keys:
                key = key.replace('M_c', 'chirp_mass')
                key = key.replace('q', 'mass_ratio')
                key = key.replace('d_L', 'luminosity_distance')
                key = key.replace('t_c', 'geocent_time')
                key = key.replace('phase_c', 'phase')
                key = key.replace('s1_mag', 'a_1')
                key = key.replace('s1_theta', 'tilt_1')
                key = key.replace('s1_phi', 'phi_1')
                key = key.replace('s2_mag', 'a_2')
                key = key.replace('s2_theta', 'tilt_2')
                key = key.replace('s2_phi', 'phi_2')
                key = key.replace('s1_z', 'chi_1')
                key = key.replace('s2_z', 'chi_2')
                samples_bilby.append(result_bilby[key].values)
            samples_bilby.append(result_bilby['log_likelihood'].values)
            samples_bilby = np.array(samples_bilby).T

            # Plot
            try:
                fig_1 = corner(samples_jim[:,:-1], labels=keys, color='blue', hist_kwargs={'density': True})
                corner(samples_jim_nf, labels=keys, fig=fig_1, color='green', hist_kwargs={'density': True})
                corner(samples_bilby[:,:-1], labels=keys, fig=fig_1, color='red', hist_kwargs={'density': True})
                fig_1.legend(['jim', 'jim_nf', 'bilby'], loc='right', fontsize=20)
                fig_1.savefig(figure_outdir/f'{event}_nf.jpg')
                plt.close(fig_1)
            except:
                pass

            keys.append('logL')
            fig_2 = corner(samples_jim, labels=keys, color='blue', hist_kwargs={'density': True})
            corner(samples_bilby, labels=keys, fig=fig_2, color='red', hist_kwargs={'density': True})
            fig_2.legend(['jim', 'bilby'], loc='right', fontsize=20)
            fig_2.savefig(figure_outdir/f'{event}.jpg')
            plt.close(fig_2)
            
            # Update csv
            csv.loc[csv['Event'] == event, 'Comparison'] = 'good'
            csv.to_csv(paths.static/'event_status.csv', index=False)
    except Exception as e:
        print(f'Error: {e}')
        continue
