from social_diffusion.datasets.haggling import (
    get_haggling_test_sequences,
    get_haggling_train_sequences,
)
import numpy as np
from einops import reduce
from tqdm import tqdm
import torch
from social_diffusion.evaluation.decider import WhoSpeaksDecider
from social_diffusion.evaluation.buyer_decider import WhoIsBuyerDecider
import social_diffusion.evaluation.jsd as jsd
from social_diffusion.eval_utils import load_trained_model, predict
from scipy.spatial.distance import jensenshannon
from typing import List

device = torch.device("cuda")
whospeaks = WhoSpeaksDecider(device=device)
whoisbuyer = WhoIsBuyerDecider(device=device)


def get_p_from_sequence(sequence: List[np.ndarray]):
    HM = []
    for Seq in tqdm(sequence):
        buyer_id = whoisbuyer.who_is_the_buyer(Seq)
        attn, seller_ids = jsd.calculate_attn(Seq, buyer_id=buyer_id)
        speech_all = whospeaks.eval(Seq)
        speech = speech_all[:, seller_ids]
        speech_buyer = speech_all[:, buyer_id]
        words = jsd.attn_to_word5(
            attn=attn, speech=speech, buyer_speech=speech_buyer
        )  # noqa E501
        hm = jsd.get_hm(words)
        HM.append(hm)

    HM = reduce(np.array(HM, dtype=np.float32), "c w h -> w h", "sum")
    return jsd.to_prob(HM)


def evaluation():
    INPUT_FRAME_10PERC = 178
    AVG_TEST_FRAMES = 1784

    ds_train, ema_diffusion = load_trained_model()

    TEST = [
        seq.Seq[INPUT_FRAME_10PERC:] for seq in get_haggling_test_sequences()
    ]  # noqa E501
    TRAIN = [
        seq.Seq[INPUT_FRAME_10PERC:] for seq in get_haggling_train_sequences()
    ]  # noqa E501

    p_test = get_p_from_sequence(TEST)
    p_train = get_p_from_sequence(TRAIN)

    print("train: JSD: ", jensenshannon(p_test, p_train))

    Pred = []
    for Seq in get_haggling_test_sequences():
        pred = predict(ema_diffusion=ema_diffusion, ds_train=ds_train, Seq=Seq)
        pred = pred[:, INPUT_FRAME_10PERC:AVG_TEST_FRAMES]  # noqa E203
        Pred.append(pred)
    Pred = np.concatenate(Pred, axis=0)  # (b s) t p Jd
    p_method = get_p_from_sequence(Pred)
    print("JSD: ", jensenshannon(p_test, p_method))


if __name__ == "__main__":
    evaluation()
