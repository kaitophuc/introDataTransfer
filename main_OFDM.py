import torch
import numpy as np
from sionna.phy import config
from sionna.phy.channel import AWGN, RayleighBlockFading, TimeChannel
from sionna.phy.mapping import BinarySource, Mapper, Demapper
from sionna.phy.utils import count_errors
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, OFDMModulator, OFDMDemodulator, LSChannelEstimator
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import LMMSEEqualizer
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.channel.tr38901 import TDL

config.seed = 0
device = config.device

num_frames = 1000

num_subcarriers = 64
num_ofdm_symbols = 14
num_pilot_symbols = 1
num_data_ofdm_symbols = num_ofdm_symbols - num_pilot_symbols

bits_per_qam_symbol = 4

code_rate = 1/2

num_coded_bits_per_frame = num_data_ofdm_symbols * num_subcarriers * bits_per_qam_symbol

num_info_bits_per_frame = int(num_coded_bits_per_frame * code_rate)

num_info_bits = num_frames * num_info_bits_per_frame
num_coded_bits = num_frames * num_coded_bits_per_frame

cp_len = 8

snr_dbs = [0, 4, 8, 12, 16, 20, 24, 28]

ber_results = []

source = BinarySource(precision="single", device=device)

mapper = Mapper(
    constellation_type="qam",
    num_bits_per_symbol=bits_per_qam_symbol,
    precision="single",
    device=device,
)

demapper = Demapper(
    demapping_method="app",
    constellation_type="qam",
    num_bits_per_symbol=bits_per_qam_symbol,
    hard_out=False,
    precision="single",
    device=device,
)

awgn = AWGN(precision="single", device=device)

info_bits = source([num_frames, num_info_bits_per_frame]).to(torch.long)

rg = ResourceGrid(
    num_ofdm_symbols=num_ofdm_symbols,
    fft_size=num_subcarriers,
    subcarrier_spacing=15e3,
    num_tx=1,
    num_streams_per_tx=1,
    cyclic_prefix_length=cp_len,
    dc_null=False,
    pilot_pattern="kronecker",
    pilot_ofdm_symbol_indices=[0],
    precision="single",
    device=device,
)

rg_mapper = ResourceGridMapper(
    rg,
    precision="single",
    device=device,
)

ofdm_modulator = OFDMModulator(
    cyclic_prefix_length=cp_len,
    precision="single",
    device=device,
)

ofdm_demodulator = OFDMDemodulator(
    fft_size=num_subcarriers,
    l_min=0,
    cyclic_prefix_length=cp_len,
    precision="single",
    device=device,
)

ls_estimator = LSChannelEstimator(
    rg,
    interpolation_type="nn",
    precision="single",
    device=device,
)

rx_tx_association = np.array([[1]])

stream_management = StreamManagement(
    rx_tx_association,
    num_streams_per_tx=1,
)

lmmse_equalizer = LMMSEEqualizer(
    rg,
    stream_management,
    precision="single",
    device=device,
)

channel_model = RayleighBlockFading(
    num_rx=1,
    num_rx_ant=1,
    num_tx=1,
    num_tx_ant=1,
    precision="single",
    device=device,
)

sionna_time_channel = TimeChannel(
    channel_model=channel_model,
    bandwidth=15e3 * num_subcarriers,
    num_time_samples=num_ofdm_symbols * (num_subcarriers + cp_len),
    l_min=0,
    l_max=0,
    normalize_channel=True,
    return_channel=False,
    precision="single",
    device=device,
)

ldpc_encoder = LDPC5GEncoder(
    k=num_info_bits_per_frame,
    n=num_coded_bits_per_frame,
    num_bits_per_symbol=bits_per_qam_symbol,
    precision="single",
    device=device,
)

ldpc_decoder = LDPC5GDecoder(
    ldpc_encoder,
    hard_out=True,
    return_infobits=True,
    num_iter=20,
    precision="single",
    device=device,
)

coded_bits_flat = ldpc_encoder(info_bits)

coded_bits = coded_bits_flat.reshape(
    num_frames,
    num_data_ofdm_symbols,
    num_subcarriers,
    bits_per_qam_symbol,
)

x_freq = mapper(coded_bits).squeeze(-1)

x_freq_sionna_input = x_freq.reshape(num_frames, 1, 1, rg.num_data_symbols)

x_grid_sionna = rg_mapper(x_freq_sionna_input)

x_time_sionna = ofdm_modulator(x_grid_sionna)

y_time_clean = sionna_time_channel(x_time_sionna)

for snr_db_target in snr_dbs:

    snr_linear_target = 10 ** (snr_db_target / 10)
    noise_power_target = 1 / snr_linear_target

    noise_power_target_tensor = torch.tensor(
        noise_power_target,
        dtype=torch.float32,
        device=device
    )

    y_time = awgn(y_time_clean, noise_power_target_tensor)

    y_grid_sionna = ofdm_demodulator(y_time)

    h_hat_sionna, err_var = ls_estimator(y_grid_sionna, noise_power_target_tensor)

    x_hat_sionna, no_eff = lmmse_equalizer(
        y_grid_sionna,
        h_hat_sionna,
        err_var,
        noise_power_target_tensor,
    )

    equalized_data_freq = x_hat_sionna.reshape(
        num_frames,
        num_data_ofdm_symbols,
        num_subcarriers,
    )

    llr = demapper(
        equalized_data_freq.unsqueeze(-1),
        no_eff.reshape(num_frames, num_data_ofdm_symbols, num_subcarriers).unsqueeze(-1),
    )

    llr_flat = llr.reshape(
        num_frames,
        num_coded_bits_per_frame,
    )

    decoded_info_bits = ldpc_decoder(llr_flat).to(torch.long)

    errors = decoded_info_bits != info_bits

    bit_errors = count_errors(info_bits, decoded_info_bits)
    ber = bit_errors / num_info_bits

    ber_results.append(ber.item())

    frame_errors = torch.any(errors, dim=1)
    fer = frame_errors.float().mean()

    print("target SNR dB:", snr_db_target)
    print("BER estimated channel:", ber.item())
    print("FER:", fer.item())
    print()