import torch

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

num_bits = 100000

bits = torch.randint(
    low=0,
    high=2,
    size=(num_bits,),
    device=device
)

symbols = (bits * 2 - 1).to(torch.float32)

snr_dbs = [0, 2, 4, 6, 8, 10, 12, 14]

h_real = torch.randn(size=(num_bits,), device=device)
h_imag = torch.randn(size=(num_bits,), device=device)

h = (h_real + 1j * h_imag) / torch.sqrt(torch.tensor(2.0, device=device))

print("first 20 h values:", h[:20].cpu().tolist())
print("smallest |h|:", torch.min(torch.abs(h)).item())
print("average |h|^2:", torch.mean(torch.abs(h) ** 2).item())
print()

received_signal_power = torch.mean(torch.abs(symbols * h) ** 2)

for snr_db_target in snr_dbs:

    snr_linear_target = 10 ** (snr_db_target / 10)
    noise_power_target = 1 / snr_linear_target
    noise_std = noise_power_target ** 0.5

    noise_real = torch.randn(size=(num_bits,), device=device)
    noise_imag = torch.randn(size=(num_bits,), device=device)

    noise = noise_std * (noise_real + 1j * noise_imag) / torch.sqrt(torch.tensor(2.0, device=device))

    noise_power = torch.mean(torch.abs(noise) ** 2)

    received = symbols * h + noise

    equalized = received / h

    decoded_bits = (torch.real(equalized) > 0).long()

    errors = decoded_bits != bits

    abs_h = torch.abs(h)

    deep_fade_mask = abs_h < 0.2
    good_channel_mask = abs_h >= 0.2

    deep_fade_errors = errors[deep_fade_mask].sum()
    deep_fade_count = deep_fade_mask.sum()

    good_channel_errors = errors[good_channel_mask].sum()
    good_channel_count = good_channel_mask.sum()

    deep_fade_ber = deep_fade_errors / deep_fade_count
    good_channel_ber = good_channel_errors / good_channel_count

    bit_errors = errors.sum()

    ber = bit_errors / num_bits

    SNR = received_signal_power / noise_power

    snr_db = 10 * torch.log10(SNR)

    print("target SNR dB:", snr_db_target)
    print("noise_std:", noise_std)
    print("bit errors:", bit_errors.item())
    print("BER:", ber.item())
    print("signal power:", received_signal_power.item())
    print("noise power:", noise_power.item())
    print("SNR: ", SNR.item())
    print("SNR dB: ", snr_db.item())
    print("deep fade count:", deep_fade_count.item())
    print("deep fade BER:", deep_fade_ber.item())
    print("good channel count:", good_channel_count.item())
    print("good channel BER:", good_channel_ber.item())
    print()