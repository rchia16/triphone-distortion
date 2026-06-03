import os
import pickle
import numpy as np
from scipy.signal import stft
from eeg_datasets import EEGRealtimeDataset

# BASE_ROOT = r"C:/Users/IDBA/Downloads/UTS_EEG/Dataloader/dataset_cache/Daniel"
BASE_ROOT = r"/data/raqchia/datasets/raw/imagined-speech/Daniel"


def path_builder(day: int, sess: int) -> str:
    return os.path.join(BASE_ROOT, f"Day{day}", f"sess{sess}.pkl")


def patch_pickle_classes():
    try:
        import __main__
        __main__.EEGRealtimeDataset = EEGRealtimeDataset
        print("[Info] Patched __main__.EEGRealtimeDataset")
    except Exception as e:
        print("[Warn] Could not patch EEGRealtimeDataset automatically:", e)
        print("       If unpickling fails, confirm eeg_datasets.py is importable.")


def read_pkl(pkl_path: str):
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)
    return dataset


# =========================================================
# 1. Single-trial STFT -> (times, bands)
# =========================================================
def tf_to_time_bands_fingerprint(tf, freqs):
    """
    tf: (channels, freqs, times)
    return:
        seq: (times, bands)
        band_names: list[str]   
    """
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 50),
    }

    band_names = list(bands.keys())
    time_band_list = []

    for _, (f_low, f_high) in bands.items():
        mask = (freqs >= f_low) & (freqs < f_high)

        if np.any(mask):
            curve = tf[:, mask, :].mean(axis=(0, 1))   # (times,)
        else:
            curve = np.zeros(tf.shape[-1], dtype=float)

        time_band_list.append(curve.astype(float))

    seq = np.stack(time_band_list, axis=0).T  # (times, bands)
    return seq, band_names


def compute_single_trial_time_band_sequence(
    trial,
    sr=250,
    nperseg=128,
    noverlap=96,
    eps=1e-8,
    fmax=50
):
    """
    input:
        trial: (channels, time)
    return:
        seq: (times, bands)
        band_names: list[str]
        freqs_used
        times
    """
    if hasattr(trial, "cpu"):
        trial = trial.cpu().numpy()
    else:
        trial = np.asarray(trial)

    mean = trial.mean(axis=-1, keepdims=True)
    std = trial.std(axis=-1, keepdims=True)
    trial_z = (trial - mean) / (std + eps)

    f, t, Zxx = stft(trial_z, fs=sr, nperseg=nperseg, noverlap=noverlap, axis=-1)
    tf = np.abs(Zxx)  # (channels, freqs, times)

    freq_mask = (f >= 0) & (f <= fmax)
    f_used = f[freq_mask]
    tf = tf[:, freq_mask, :]

    seq, band_names = tf_to_time_bands_fingerprint(tf, f_used)
    return seq, band_names, f_used, t


# =========================================================
# 2. Trial-level collection
# =========================================================
def collect_session_trial_sequences(
    days,
    sessions,
    nperseg=128,
    noverlap=96,
    eps=1e-8,
    fmax=50
):
    """
    將每一筆 EEG trial 都獨立保留，不做 session 內 label 平均。

    return:
        all_trial_seq: list of dict
            [
                {
                    "day": 1,
                    "sess": 1,
                    "trial_global_index": int,
                    "label_id": int,
                    "label_name": str,
                    "seq": (times, bands),
                    "band_names": [...],
                    "raw_trial": (channels, time),   # optional for debugging
                },
                ...
            ]
    """
    all_trial_seq = []

    for day in days:
        for sess in sessions:
            pkl_path = path_builder(day, sess)
            print(f"[Collect] Loading: {pkl_path}")
            dataset = read_pkl(pkl_path)

            sr = getattr(dataset, "final_sampling_rate", 250)
            all_label = np.asarray(dataset.all_label)

            for trial_idx, (trial, label_id) in enumerate(zip(dataset.all_eeg, all_label)):
                if hasattr(trial, "cpu"):
                    trial_np = trial.cpu().numpy()
                else:
                    trial_np = np.asarray(trial)

                seq, band_names, freqs_used, times = compute_single_trial_time_band_sequence(
                    trial_np,
                    sr=sr,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    eps=eps,
                    fmax=fmax
                )

                label_name = dataset.final_id2word[int(label_id)]

                all_trial_seq.append({
                    "day": day,
                    "sess": sess,
                    "trial_global_index": len(all_trial_seq),
                    "trial_index_within_session": trial_idx,
                    "label_id": int(label_id),
                    "label_name": label_name,
                    "seq": seq,                    # (times, bands)
                    "band_names": band_names,
                    "freqs": freqs_used,
                    "times": times,
                    "raw_trial": trial_np
                })

    return all_trial_seq


# =========================================================
# 4. DTW utilities
# =========================================================
def dtw_distance_multivariate(X, Y):
    """
    X: (T1, D)
    Y: (T2, D)
    return normalized dtw cost
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    T1, D1 = X.shape
    T2, D2 = Y.shape
    if D1 != D2:
        raise ValueError(f"Feature dim mismatch: {D1} vs {D2}")

    cost = np.full((T1 + 1, T2 + 1), np.inf, dtype=float)
    cost[0, 0] = 0.0

    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            dist = np.linalg.norm(X[i - 1] - Y[j - 1])
            cost[i, j] = dist + min(
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1]
            )

    return float(cost[T1, T2] / (T1 + T2))


def dtw_path_multivariate(X, Y):
    """
    return:
        path: list[(i, j)]
        normalized_cost
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    T1, D1 = X.shape
    T2, D2 = Y.shape
    if D1 != D2:
        raise ValueError(f"Feature dim mismatch: {D1} vs {D2}")

    cost = np.full((T1 + 1, T2 + 1), np.inf, dtype=float)
    cost[0, 0] = 0.0

    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            dist = np.linalg.norm(X[i - 1] - Y[j - 1])
            cost[i, j] = dist + min(
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1]
            )

    i, j = T1, T2
    path = []

    while i > 0 and j > 0:
        path.append((i - 1, j - 1))

        prev_candidates = [
            (cost[i - 1, j - 1], i - 1, j - 1),
            (cost[i - 1, j], i - 1, j),
            (cost[i, j - 1], i, j - 1),
        ]
        _, i, j = min(prev_candidates, key=lambda x: x[0])

    while i > 0:
        path.append((i - 1, 0))
        i -= 1

    while j > 0:
        path.append((0, j - 1))
        j -= 1

    path.reverse()
    normalized_cost = float(cost[T1, T2] / (T1 + T2))
    return path, normalized_cost


def warp_sequence_to_reference(seq, ref_seq):
    """
    將 seq warp 到 ref_seq 的時間軸。
    input:
        seq:     (T, D)
        ref_seq: (T_ref, D)
    return:
        warped_seq: (T_ref, D)
        path
        cost
    """
    path, cost = dtw_path_multivariate(seq, ref_seq)

    T_ref = ref_seq.shape[0]
    D = seq.shape[1]
    aligned_buckets = [[] for _ in range(T_ref)]

    for i, j in path:
        aligned_buckets[j].append(seq[i])

    warped_seq = np.zeros((T_ref, D), dtype=float)
    for j in range(T_ref):
        if len(aligned_buckets[j]) > 0:
            warped_seq[j] = np.mean(np.stack(aligned_buckets[j], axis=0), axis=0)
        else:
            warped_seq[j] = ref_seq[j]

    return warped_seq, path, cost


def select_medoid_sequence(seq_list):
    """
    從多條 sequence 中選 DTW medoid。
    return:
        medoid_idx
        distance_matrix
    """
    n = len(seq_list)
    if n == 0:
        raise ValueError("Empty seq_list")
    if n == 1:
        return 0, np.zeros((1, 1), dtype=float)

    dist_mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance_multivariate(seq_list[i], seq_list[j])
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    row_sums = dist_mat.sum(axis=1)
    medoid_idx = int(np.argmin(row_sums))
    return medoid_idx, dist_mat


# =========================================================
# 5. DTW-medoid template from all trials of same label
# =========================================================
def build_template_dtw_aligned_band_sequences_from_trials(all_trial_seq):
    """
    同一 label 的所有 trial-level sequences：
    1. 選 DTW medoid
    2. 其他 trials warp 到 medoid 時間軸
    3. 再平均
    """
    label_seq_list = {}
    label_name = {}
    band_names_ref = None

    for item in all_trial_seq:
        label = item["label_id"]

        if label not in label_seq_list:
            label_seq_list[label] = []
            label_name[label] = item["label_name"]

        if band_names_ref is None:
            band_names_ref = item["band_names"]

        label_seq_list[label].append(item["seq"])

    template_mean_seq = {}

    for label, seq_list in label_seq_list.items():
        medoid_idx, dist_mat = select_medoid_sequence(seq_list)
        medoid_seq = seq_list[medoid_idx]

        warped_list = []
        warp_costs = []

        for seq in seq_list:
            warped_seq, _, cost = warp_sequence_to_reference(seq, medoid_seq)
            warped_list.append(warped_seq)
            warp_costs.append(cost)

        warped_stack = np.stack(warped_list, axis=0)   # (n_trials, T_ref, bands)
        seq_mean = warped_stack.mean(axis=0)
        seq_std = warped_stack.std(axis=0)

        template_mean_seq[label] = {
            "name": label_name[label],
            "band_names": band_names_ref,
            "seq_mean": seq_mean,
            "seq_std": seq_std,
            "seq_all": warped_stack,
            "medoid_idx": medoid_idx,
            "medoid_seq": medoid_seq,
            "dtw_distance_matrix": dist_mat,
            "warp_costs_to_medoid": np.array(warp_costs, dtype=float),
            "n_trials_used": len(seq_list)
        }

    return template_mean_seq


# =========================================================
# 6. Read one trial from dataset
# =========================================================
def get_one_trial_from_dataset(day, sess, label, trial_index_within_label=0):
    """
    直接從某 dataset 中取出指定 label 的其中一筆 trial
    """
    pkl_path = path_builder(day, sess)
    print(f"[Ref] Loading one trial from: {pkl_path}")
    dataset = read_pkl(pkl_path)

    all_label = np.asarray(dataset.all_label)
    idx = np.where(all_label == label)[0]

    if len(idx) == 0:
        raise ValueError(f"No trial found for label={label} in Day{day} Session{sess}")

    if trial_index_within_label < 0 or trial_index_within_label >= len(idx):
        raise IndexError(
            f"trial_index_within_label={trial_index_within_label} out of range. "
            f"Available trials for label {label}: 0 ~ {len(idx)-1}"
        )

    trial = dataset.all_eeg[idx[trial_index_within_label]]
    label_name = dataset.final_id2word[label]

    if hasattr(trial, "cpu"):
        trial = trial.cpu().numpy()
    else:
        trial = np.asarray(trial)

    return trial, label_name, dataset


# =========================================================
# 7. L2 compare
# =========================================================
def compute_l2_to_template(input_seq, template_seq):
    if input_seq.shape != template_seq.shape:
        raise ValueError(f"Shape mismatch: input_seq {input_seq.shape}, template_seq {template_seq.shape}")

    per_time_l2 = np.linalg.norm(input_seq - template_seq, axis=1)
    mean_l2 = float(np.mean(per_time_l2))
    return per_time_l2, mean_l2


def compute_dtw_aligned_l2_to_template(input_seq, template_seq):
    warped_input_seq, path, dtw_cost = warp_sequence_to_reference(input_seq, template_seq)
    per_time_l2, mean_l2 = compute_l2_to_template(warped_input_seq, template_seq)

    return {
        "warped_input_seq": warped_input_seq,
        "path": path,
        "dtw_cost": dtw_cost,
        "per_time_l2": per_time_l2,
        "mean_l2": mean_l2
    }


# =========================================================
# 8. Build template object
# =========================================================
def build_template_from_dataset(
    template_days,
    template_sessions,
    nperseg=128,
    noverlap=96,
    eps=1e-8,
    fmax=50,
):
    patch_pickle_classes()

    all_trial_seq = collect_session_trial_sequences(
        days=template_days,
        sessions=template_sessions,
        nperseg=nperseg,
        noverlap=noverlap,
        eps=eps,
        fmax=fmax
    )

    template_mean_seq = build_template_dtw_aligned_band_sequences_from_trials(all_trial_seq)

    template_object = {
        "template_days": list(template_days),
        "template_sessions": list(template_sessions),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "eps": eps,
        "fmax": fmax,
        "template_mean_seq": template_mean_seq,
        "all_trial_seq": all_trial_seq
    }
    return template_object


# =========================================================
# 9. Compare one input signal to prebuilt template
# =========================================================
def compare_signal_to_prebuilt_template(
    template_object,
    input_signal,
    input_label,
    output_mode="both",
    sr=250
):
    
    template_mean_seq = template_object["template_mean_seq"]

    if input_label not in template_mean_seq:
        raise ValueError(f"input_label={input_label} not found in prebuilt template")

    input_seq, band_names, freqs_used, times = compute_single_trial_time_band_sequence(
        input_signal,
        sr=sr,
        nperseg=template_object["nperseg"],
        noverlap=template_object["noverlap"],
        eps=template_object["eps"],
        fmax=template_object["fmax"]
    )

    template_seq = template_mean_seq[input_label]["seq_mean"]

    aligned_result = compute_dtw_aligned_l2_to_template(input_seq, template_seq)
    result = {
        "label_id": input_label,
        "label_name": template_mean_seq[input_label]["name"],
        "band_names": band_names,
        "times": times,
        "per_time_l2": aligned_result["per_time_l2"],
        "mean_l2": aligned_result["mean_l2"],
        "input_seq": input_seq,
        "template_seq": template_seq,
        "warped_input_seq": aligned_result["warped_input_seq"],
        "dtw_cost": aligned_result["dtw_cost"],
        "dtw_path": aligned_result["path"]
        }

    if output_mode == "per_time":
        return {
            "label_id": result["label_id"],
            "label_name": result["label_name"],
            "times": result["times"],
            "per_time_l2": result["per_time_l2"]
        }
    elif output_mode == "mean":
        return {
            "label_id": result["label_id"],
            "label_name": result["label_name"],
            "mean_l2": result["mean_l2"]
        }
    elif output_mode == "both":
        return result
    else:
        raise ValueError("output_mode must be one of: 'per_time', 'mean', 'both'")


# =========================================================
# 10. Optional save/load
# =========================================================
def save_template_object(template_object, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(template_object, f)
    print(f"[Saved template] {save_path}")


def load_template_object(save_path):
    with open(save_path, "rb") as f:
        template_object = pickle.load(f)
    print(f"[Loaded template] {save_path}")
    return template_object


# =========================================================
# 11. Example
# =========================================================
def main():
    # 建 DTW trial-level template
    template_object = build_template_from_dataset(
        template_days=[1],
        template_sessions=range(1, 8),
        nperseg=128,
        noverlap=96,
        eps=1e-8,
        fmax=50
    )

    # 取一筆 test/ref trial
    trial, label_name, dataset = get_one_trial_from_dataset(
        day=1,
        sess=8,
        label=0,
        trial_index_within_label=0
    )
    print("Loaded trial label:", label_name)

    # 比較
    result = compare_signal_to_prebuilt_template(
        template_object=template_object,
        input_signal=trial,
        input_label=0,
        output_mode="both",
        sr=250
    )

    print("Mean L2:", result["mean_l2"])
    print("Per-time L2:", result["per_time_l2"])
    if "dtw_cost" in result:
        print("DTW cost:", result["dtw_cost"])


if __name__ == "__main__":
    main()
