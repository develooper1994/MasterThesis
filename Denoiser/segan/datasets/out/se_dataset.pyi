from ahoproc_tools.interpolate import *
from ahoproc_tools.io import *
from torch.utils.data.dataset import Dataset
from typing import Any, Optional

def collate_fn(batch: Any): ...
def slice_signal(signal: Any, window_sizes: Any, stride: float = ...): ...
def slice_index_helper(args: Any): ...
def slice_signal_index(path: Any, window_size: Any, stride: Any): ...
def abs_normalize_wave_minmax(x: Any): ...
def abs_short_normalize_wave_minmax(x: Any): ...
def dynamic_normalize_wave_minmax(x: Any): ...
def normalize_wave_minmax(x: Any): ...
def pre_emphasize(x: Any, coef: float = ...): ...
def de_emphasize(y: Any, coef: float = ...): ...

class SEDataset(Dataset):
    clean_names: Any = ...
    noisy_names: Any = ...
    slice_workers: Any = ...
    cache_dir: Any = ...
    slice_size: Any = ...
    stride: Any = ...
    split: Any = ...
    verbose: Any = ...
    preemph: Any = ...
    preemph_norm: Any = ...
    random_scale: Any = ...
    idx2slice: Any = ...
    num_samples: Any = ...
    slicings: Any = ...
    def __init__(self, clean_dir: Any, noisy_dir: Any, preemph: Any, cache_dir: str = ..., split: str = ..., slice_size: Any = ..., stride: float = ..., max_samples: Optional[Any] = ..., do_cache: bool = ..., verbose: bool = ..., slice_workers: int = ..., preemph_norm: bool = ..., random_scale: Any = ...) -> None: ...
    def read_wav_file(self, wavfilename: Any): ...
    clean_paths: Any = ...
    noisy_paths: Any = ...
    def read_wavs(self) -> None: ...
    clean_wavs: Any = ...
    noisy_wavs: Any = ...
    def read_wavs_and_cache(self) -> None: ...
    def prepare_slicing(self) -> None: ...
    def extract_slice(self, index: Any): ...
    def __getitem__(self, index: Any): ...
    def __len__(self): ...

class RandomChunkSEDataset(Dataset):
    preemph: Any = ...
    utt2spk: Any = ...
    spk2idx: Any = ...
    samples: Any = ...
    slice_size: Any = ...
    clean_names: Any = ...
    def __init__(self, clean_dir: Any, noisy_dir: Any, preemph: Any, split: str = ..., slice_size: Any = ..., max_samples: Optional[Any] = ..., utt2spk: Optional[Any] = ..., spk2idx: Optional[Any] = ...) -> None: ...
    def read_utt2spk(self) -> None: ...
    def read_wav_file(self, wavfilename: Any): ...
    def __getitem__(self, index: Any): ...
    def __len__(self): ...

class RandomChunkSEF0Dataset(Dataset):
    preemph: Any = ...
    samples: Any = ...
    slice_size: Any = ...
    clean_names: Any = ...
    def __init__(self, clean_dir: Any, noisy_dir: Any, preemph: int = ..., split: str = ..., slice_size: Any = ..., max_samples: Optional[Any] = ...) -> None: ...
    def read_wav_file(self, wavfilename: Any): ...
    def __getitem__(self, index: Any): ...
    def __len__(self): ...

class SEH5Dataset(Dataset):
    data_root: Any = ...
    split: Any = ...
    preemph: Any = ...
    verbose: Any = ...
    random_scale: Any = ...
    f: Any = ...
    def __init__(self, data_root: Any, split: Any, preemph: Any, verbose: bool = ..., preemph_norm: bool = ..., random_scale: Any = ...) -> None: ...
    def __getitem__(self, index: Any): ...
    def __len__(self): ...