from torch import from_numpy
from numpy import pad
from soundfile import read as load_audio
from huggingface_hub import hf_hub_download
from torchaudio.functional import resample
from onnxruntime import InferenceSession
from tqdm.auto import tqdm
from pandas import DataFrame
from pathlib import Path
from argparse import ArgumentParser

def read16k(path):
    wav, sr = load_audio(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = resample(from_numpy(wav), sr, 16000).numpy()
    return wav


def predict_logit(path):
    wav = read16k(path)
    L = inp.shape[1]
    n = len(wav)

    if n == 0:
        raise ValueError(f"Пустая запись: {path}")

    if n < L:
        missing = L - n
        left_pads = sorted({0,  missing // 4, missing // 2, 3 * missing // 4, missing})
        windows = [pad(wav, (left, missing - left)) for left in left_pads]
    elif n == L:
        windows = [wav]
    else:
        extra = n - L
        starts = sorted({0, extra // 4, extra // 2, 3 * extra // 4, extra})
        windows = [wav[s:s + L] for s in starts]

    scores = []
    for window in windows:
        lg = sess.run([out.name], {inp.name: window[None, :]})[0]
        scores.append(float(lg[0, 0]) - float(lg[0, 1]))

    return sum(scores) / len(scores)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test", required=True, type=Path)
    args = parser.parse_args()

    onnx_path = hf_hub_download(
        repo_id="lab260/Spectra-AASIST3",
        filename="spectra-aasist3.onnx"
    )
    sess = InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]

    test_files = sorted(args.test.iterdir(), key=lambda path: int(path.stem))

    test_probs = [predict_logit(p) for p in tqdm(test_files)]

    submit = DataFrame({
        "idx": list(map(lambda x: Path(x).name, test_files)),
        "score": test_probs
    })

    Path("subs").mkdir(exist_ok=True)
    submit.to_csv("subs/submission.csv", index=False)
