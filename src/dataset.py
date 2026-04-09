import torch
import os
import json
import numpy as np
try:
    import kaldiio
except ImportError:
    kaldiio = None
import copy
import soundfile


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        prompt_path=None,
        wav_type="wav",
        inference_mode=False,
        max_duration=15.0,
        scp_filename="daic.scp",
        task_filename="daic_multitask.jsonl",
    ):
        """
        data_path: Data directory, must contain the configured scp/task metadata files
        prompt_path: Path to the jsonl file containing task prompts
        wav_type: 'wav' or 'ark'
        inference_mode: When True, target is not returned and is used for inference
        max_duration: Maximum audio duration (seconds); longer audio will be truncated
        scp_filename: Name of the wav scp metadata file inside data_path
        task_filename: Name of the multitask metadata file inside data_path
        """
        self.wav_scp = {}
        self.tasks = []
        self.prompt = {}
        self.wav_type = wav_type
        self.inference_mode = inference_mode
        self.max_duration = max_duration  # Maximum allowed duration (seconds)
        self.scp_filename = scp_filename
        self.task_filename = task_filename
        self.datasets_root = os.environ.get("AUDIOLLM_DATASETS_ROOT")
        if self.datasets_root is not None:
            self.datasets_root = os.path.abspath(self.datasets_root)
        self.data_path = os.path.abspath(data_path)

        # Read wav.scp
        with open(os.path.join(data_path, self.scp_filename)) as f:
            for line in f:
                utt_id, wav_path = line.strip().split(" ", 1)
                self.wav_scp[utt_id] = self._resolve_wav_path(wav_path)

        # Read the task file
        with open(os.path.join(data_path, self.task_filename)) as f:
            for line in f:
                self.tasks.append(json.loads(line))

        # Read the prompt file
        with open(os.path.join(prompt_path)) as f:
            for line in f:
                item = json.loads(line)
                self.prompt[item["task"]] = item["prompt"]

    def _resolve_wav_path(self, wav_path):
        if os.path.isabs(wav_path):
            return os.path.abspath(wav_path)

        if self.datasets_root is not None:
            candidate = os.path.abspath(os.path.join(self.datasets_root, wav_path))
            if os.path.exists(candidate):
                return candidate

        candidate = os.path.abspath(os.path.join(self.data_path, wav_path))
        if os.path.exists(candidate):
            return candidate

        return wav_path

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        # Get task information
        key = self.tasks[idx]["key"]
        target = self.tasks[idx]["target"]
        prompt = self.prompt[self.tasks[idx]["task"]]

        # Read audio
        if self.wav_type == "ark":
            audio = kaldiio.load_mat(self.wav_scp[key])[1].astype(np.float32) / 32768
            sr = 16000  # Data sampling rate is 16 kHz
        elif self.wav_type == "wav":
            try:
                audio, sr = soundfile.read(self.wav_scp[key])
            except Exception as exc:
                raise RuntimeError(
                    "Failed to read audio file "
                    f"for key={key} path={self.wav_scp[key]} "
                    f"(datasets_root={self.datasets_root}, data_path={self.data_path})"
                ) from exc
            if len(audio.shape) > 1:
                # CHECKPOINT
                # Instead of choosing one channel,
                # we can also average the channels to create a mono signal,!!!
                # If multi-channel, take the first channel
                audio = audio[:, 0]
        else:
            raise ValueError(f"Unsupported wav_type: {self.wav_type}")

        # CHECKPOINT == commented truncation logic f
        # # ===== Truncation logic =====
        # max_samples = int(self.max_duration * sr)
        # if len(audio) > max_samples:
        #     if not self.inference_mode:
        #         # Training mode: randomly crop a segment of audio
        #         start = np.random.randint(0, len(audio) - max_samples)
        #         audio = audio[start:start + max_samples]
        #     else:
        #         # Inference mode: take the first 15 seconds
        #         audio = audio[:max_samples]

        # ===== Return data =====
        if not self.inference_mode:
            return {
                "prompt": prompt,
                "audio": audio,
                "target": target
            }
        else:
            return {
                "prompt": prompt,
                "audio": audio,
                "target": "",
                "key": key
            }

# OLD VERSION (BAD USAGE OF MASKING OF THE PADDING)
# def collate_fn_qwen2audio(samples, processor):
#     prompt = [_["prompt"] for _ in samples] # prompt = transcript+emotion description
#     audio = [_["audio"] for _ in samples]
#     target = [_["target"] for _ in samples]

#     # Concatenate prompt + target to form the full input text
#     processed_data = processor(
#         text=[i + j for i, j in zip(prompt, target)],
#         audios=audio, #this is the wrong but working better (no audio will being processed because of the error)
#         # audio=audio, #this is the correct but working worse (audio will be processed)
#         sampling_rate=processor.feature_extractor.sampling_rate,
#         return_tensors="pt",
#         padding=True
#     )

#     # ===== Process labels (mask the prompt portion) =====
#     labels = copy.deepcopy(processed_data["input_ids"])
#     text_ids = processor(prompt, return_tensors="pt", padding=True)

#     for i, attention_mask in enumerate(text_ids["attention_mask"]):
#         labels[i, :sum(attention_mask) +
#                (processed_data["input_ids"][i] == processor.tokenizer.pad_token_id).sum().item()] = -100

#     processed_data["labels"] = labels

#     if "key" in samples[0]:
#         keys = [_["key"] for _ in samples]
#         processed_data["keys"] = keys

#     return processed_data

def collate_fn_qwen2audio(samples, processor):
    prompt = [s["prompt"] for s in samples]
    audio = [s["audio"] for s in samples]
    target = [s["target"] for s in samples]

    processed_data = processor(
        text=[p + t for p, t in zip(prompt, target)],
        audio=audio,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True
    )

    labels = processed_data["input_ids"].clone()

    prompt_ids = processor(
        text=prompt,
        return_tensors="pt",
        padding=True
    )

    prompt_lens = prompt_ids["attention_mask"].sum(dim=1)

    for i, prompt_len in enumerate(prompt_lens):
        labels[i, :prompt_len] = -100

    # padding tokens loss = -100
    labels[processed_data["attention_mask"] == 0] = -100

    processed_data["labels"] = labels

    if "key" in samples[0]:
        processed_data["keys"] = [s["key"] for s in samples]

    return processed_data