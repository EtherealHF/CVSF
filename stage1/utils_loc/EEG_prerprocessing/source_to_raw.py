import os
from scipy import io
import numpy as np
import mne
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

def eeg_to_npy(sub, session, partition, project_dir, save_beta_gamma=True):
    if partition == 'train':
        n_parts = 5
    elif partition == 'test':
        n_parts = 1

    data_dir = os.path.join(project_dir, 'source_eeg_data',
                            f'sub-{sub:02}', f'ses-{session:02}')
    eeg_data_matrix = None

    for p in range(n_parts):
        if partition == 'train':
            beh_dir = os.path.join(data_dir, 'beh', f'sub-{sub:02}_ses-{session:02}_task-{partition}_part-{p+1:02}_beh.mat')
            eeg_dir = os.path.join(data_dir, 'eeg', f'sub-{sub:02}_ses-{session:02}_task-{partition}_part-{p+1:02}_eeg.vhdr')
        else:
            beh_dir = os.path.join(data_dir, 'beh', f'sub-{sub:02}_ses-{session:02}_task-{partition}_beh.mat')
            eeg_dir = os.path.join(data_dir, 'eeg', f'sub-{sub:02}_ses-{session:02}_task-{partition}_eeg.vhdr')

        beh_data = io.loadmat(beh_dir)['data']
        eeg_data = mne.io.read_raw_brainvision(eeg_dir, preload=True)

        provv_eeg_mat = eeg_data.get_data()
        info = eeg_data.info

        # Process event labels.
        events_beh = np.asarray(beh_data[0][0][2]['tot_img_number'][0], dtype=int)
        idx_targ = np.where(events_beh == 0)[0]
        events_beh[idx_targ] = 99999
        del beh_data

        events_samples, _ = mne.events_from_annotations(eeg_data)
        events_samples = events_samples[1:,0]
        del eeg_data

        # Build event channel.
        events_channel = np.zeros((1, provv_eeg_mat.shape[1]))
        idx = 0
        for s in range(events_channel.shape[1]):
            if idx < len(events_beh) and events_samples[idx] == s:
                events_channel[0, s] = events_beh[idx]
                idx += 1
        provv_eeg_mat = np.append(provv_eeg_mat, events_channel, 0)
        del events_channel, events_samples

        # Concatenate EEG parts.
        if p == 0:
            eeg_data_matrix = provv_eeg_mat
        else:
            eeg_data_matrix = np.append(eeg_data_matrix, provv_eeg_mat, 1)
        del provv_eeg_mat

    # 閫氶亾鍚嶃€佺被鍨?    ch_names = info.ch_names + ['stim']
    ch_types = ['eeg'] * (len(ch_names) - 1) + ['stim']

    # 杈撳嚭鐩綍
    output_dir = os.path.join(project_dir,'raw_eeg_data', f'sub-{sub:02}', f'ses-{session:02}')
    os.makedirs(output_dir, exist_ok=True)

    base_name = f'sub-{sub:02}_ses-{session:02}_task-{partition}'
    data_raw = {
        'raw_eeg_data': eeg_data_matrix,
        'ch_names': ch_names,
        'ch_types': ch_types,
        'sfreq': 1000,
        'highpass': 0.01,
        'lowpass': 100
    }
    np.save(os.path.join(output_dir, base_name + '_raw.npy'), data_raw)
    print(f"Saved {base_name}_raw.npy")

    if save_beta_gamma:
        eeg = eeg_data_matrix[:-1]
        stim = eeg_data_matrix[-1:]
        fs = 1000


        beta = np.array([bandpass_filter(ch, 13, 30, fs) for ch in eeg])
        beta_full = np.concatenate([beta, stim], axis=0)
        data_beta = data_raw.copy()
        data_beta.update({
            'raw_eeg_data': beta_full,
            'highpass': 13,
            'lowpass': 30
        })
        np.save(os.path.join(output_dir, base_name + '_beta.npy'), data_beta)
        print(f"Saved {base_name}_beta.npy")


        gamma = np.array([bandpass_filter(ch, 30, 80, fs) for ch in eeg])
        gamma_full = np.concatenate([gamma, stim], axis=0)
        data_gamma = data_raw.copy()
        data_gamma.update({
            'raw_eeg_data': gamma_full,
            'highpass': 30,
            'lowpass': 80
        })
        np.save(os.path.join(output_dir, base_name + '_gamma.npy'), data_gamma)
        print(f"Saved {base_name}_gamma.npy")

if __name__ == "__main__":
    subjects = [8]
    sessions = [1, 2, 3, 4]
    partitions = ['train', 'test']
    project_dir = '/path/to/THINGS-EEG2/'
    project_dir = '/path/to/THINGS-EEG2/'  # 鏀规垚浣犵殑璺緞


    for sub in subjects:
        for session in sessions:
            for partition in partitions:
                print(f"\n=== Processing sub-{sub:02}, ses-{session:02}, {partition} ===")
                try:
                    eeg_to_npy(sub, session, partition, project_dir, save_beta_gamma=True)
                except Exception as e:
                    print(f"Failed sub-{sub:02}, ses-{session:02}, {partition}: {e}")
