import os
from os.path import join, isdir
from social_diffusion.datasets.haggling import get_haggling_test_sequences
from social_diffusion.evaluation.db import Database
import numpy as np
from tqdm import tqdm
from social_diffusion import get_output_dir
from social_diffusion.eval_utils import load_trained_model, predict


def evaluation():
    INPUT_FRAME_10PERC = 178
    AVG_TEST_FRAMES = 1784

    ds_train, ema_diffusion = load_trained_model()

    test_seq = get_haggling_test_sequences(to_sequence=False)
    db = Database(test_seq, motion_word_size=10)

    eval_out_dir = join(get_output_dir(), "eval")
    if not isdir(eval_out_dir):
        os.makedirs(eval_out_dir)

    avg_ndms_per_seq = []
    for seqid, Seq in enumerate(get_haggling_test_sequences()):
        pred = predict(ema_diffusion=ema_diffusion, ds_train=ds_train, Seq=Seq)
        pred = pred[:, INPUT_FRAME_10PERC - 10 : AVG_TEST_FRAMES]  # noqa E203
        # pred --> n_samples x n_frames x n_person x Jd
        n_person = pred.shape[2]
        fname_out = join(eval_out_dir, f"ndms_seq{seqid}.npy")
        n_batch = pred.shape[0]
        Values = []
        for b in tqdm(range(n_batch), leave=True):
            for pid in range(n_person):
                seq = pred[b, :, pid]
                d, _ = db.rolling_query(seq)
                Values.append(d)
        Values = np.array(Values, dtype=np.float32)
        np.save(fname_out, Values)  # save per-frame ndms to file..

        avg_ndms_per_seq.append(np.mean(Values))
    print("Avg ndms: ", avg_ndms_per_seq)


if __name__ == "__main__":
    evaluation()
