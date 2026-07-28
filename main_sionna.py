import torch
from sionna.phy import config
from sionna.phy.channel import AWGN
from sionna.phy.mapping import BinarySource
from sionna.phy.utils import complex_normal, count_errors

config.seed = 0
device = config.device

num_bits = 1000000

source = BinarySource(precision="single", device=device)

bits_per_frame = 8
num_frames = num_bits // bits_per_frame

bits = source([num_frames, bits_per_frame]).to(torch.long)

symbols = (bits * 2 - 1).to(torch.float32)

snr_dbs = [0, 2, 4, 6, 8, 10, 12, 14]

h = complex_normal(
    (num_frames, bits_per_frame),
    precision="single",
    device=device,
)

received_signal_power = torch.mean(torch.abs(symbols * h) ** 2)

awgn = AWGN(precision="single", device=device)

for snr_db_target in snr_dbs:

    snr_linear_target = 10 ** (snr_db_target / 10)
    noise_power_target = 1 / snr_linear_target

    noise_power_target_tensor = torch.tensor(
        noise_power_target,
        dtype=torch.float32,
        device=device
    )

    clean_received = symbols * h
    received = awgn(clean_received, noise_power_target_tensor)

    equalized = received / h

    decoded_bits = (torch.real(equalized) > 0).long()

    errors = decoded_bits != bits

    frame_errors = torch.any(errors, dim=1)
    num_frame_errors = frame_errors.sum()
    fer = num_frame_errors / num_frames

    abs_h = torch.abs(h)

    deep_fade_mask = abs_h < 0.2
    good_channel_mask = abs_h >= 0.2

    deep_fade_errors = errors[deep_fade_mask].sum()
    deep_fade_count = deep_fade_mask.sum()

    good_channel_errors = errors[good_channel_mask].sum()
    good_channel_count = good_channel_mask.sum()

    deep_fade_ber = deep_fade_errors / deep_fade_count
    good_channel_ber = good_channel_errors / good_channel_count

    bit_errors = count_errors(bits, decoded_bits)

    ber = bit_errors / num_bits

    print("target SNR dB:", snr_db_target)
    print("BER:", ber.item())
    print("deep fade BER:", deep_fade_ber.item())
    print("good channel BER:", good_channel_ber.item())
    print("FER: ", fer.item())
    print()