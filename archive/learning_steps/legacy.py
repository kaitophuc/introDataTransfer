def qam16_modulate(bits, device):
    b0 = bits[..., 0]
    b1 = bits[..., 1]
    b2 = bits[..., 2]
    b3 = bits[..., 3]

    real_part = torch.empty_like(b0, dtype=torch.float32)
    imag_part = torch.empty_like(b2, dtype=torch.float32)

    real_part[(b0 == 0) & (b1 == 0)] = -3
    real_part[(b0 == 0) & (b1 == 1)] = -1
    real_part[(b0 == 1) & (b1 == 1)] = 1
    real_part[(b0 == 1) & (b1 == 0)] = 3

    imag_part[(b2 == 0) & (b3 == 0)] = -3
    imag_part[(b2 == 0) & (b3 == 1)] = -1
    imag_part[(b2 == 1) & (b3 == 1)] = 1
    imag_part[(b2 == 1) & (b3 == 0)] = 3

    sqrt_10 = torch.sqrt(torch.tensor(10.0, device=device))
    symbols = (real_part + 1j * imag_part).to(torch.complex64)

    return symbols / sqrt_10

def qam16_demodulate(symbols, device):
    sqrt_10 = torch.sqrt(torch.tensor(10.0, device=device))
    scaled_symbols = symbols * sqrt_10

    decoded_b0 = (torch.real(scaled_symbols) > 0).long()
    decoded_b1 = (torch.abs(torch.real(scaled_symbols)) < 2).long()

    decoded_b2 = (torch.imag(scaled_symbols) > 0).long()
    decoded_b3 = (torch.abs(torch.imag(scaled_symbols)) < 2).long()

    return torch.stack(
        [decoded_b0, decoded_b1, decoded_b2, decoded_b3],
        dim=-1,
    )

def ofdm_ifft(x_freq):
    x_freq = torch.fft.ifftshift(x_freq, dim=-1)
    return torch.fft.ifft(x_freq.to(torch.complex64), dim=-1, norm="ortho")

def add_cyclic_prefix(x_time, cp_len):
    cyclic_prefix = x_time[..., -cp_len:]
    return torch.cat([cyclic_prefix, x_time], dim=-1)

def remove_cyclic_prefix(y_time_cp, cp_len, num_subcarriers):
    return y_time_cp[..., cp_len:cp_len + num_subcarriers]

def ofdm_fft(y_time):
    y_freq = torch.fft.fft(y_time, dim=-1, norm="ortho")
    return torch.fft.fftshift(y_freq, dim=-1)

def apply_time_domain_channel(x_time, h_time):
    y_time_clean = torch.zeros_like(x_time)

    for tap_index, tap_value in enumerate(h_time):
        y_time_clean[..., tap_index:] += tap_value * x_time[..., :x_time.shape[-1] - tap_index]

    return y_time_clean

def equalize_frequency_domain(y_freq, h_freq):
    return y_freq / h_freq