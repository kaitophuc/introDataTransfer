import math
import torch
import numpy as np
from sionna.phy import config
from sionna.phy.channel import AWGN, TimeChannel
from sionna.phy.mapping import BinarySource, Mapper, Demapper
from sionna.phy.utils import count_errors
from sionna.phy.mimo import StreamManagement
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.ofdm import (
    LMMSEEqualizer,
    LSChannelEstimator,
    OFDMDemodulator,
    OFDMModulator,
    PilotPattern,
    ResourceGrid,
    ResourceGridMapper,
)

config.seed = 0

def build_scattered_pilot_pattern(num_ofdm_symbols, num_subcarriers, device):
    pilot_mask = torch.zeros(
        (1, 1, num_ofdm_symbols, num_subcarriers),
        dtype=torch.bool,
        device=device,
    )

    pilot_mask[:, :, 0::4, 0::4] = True
    pilot_mask[:, :, 2::4, 2::4] = True

    num_pilot_res = int(torch.sum(pilot_mask).item())

    pilot_symbols = torch.ones(
        (1, 1, num_pilot_res),
        dtype=torch.complex64,
        device=device,
    )

    return PilotPattern(
        pilot_mask,
        pilot_symbols,
        precision="single",
        device=device,
    )

def run_one_batch(
    batch_frames,
    noise_power_target_tensor,
    source,
    ldpc_encoder,
    mapper,
    rg,
    rg_mapper,
    ofdm_modulator,
    sionna_time_channel,
    awgn,
    ofdm_demodulator,
    ls_estimator,
    lmmse_equalizer,
    demapper,
    ldpc_decoder,
    num_info_bits_per_frame,
    num_coded_bits_per_frame,
    bits_per_qam_symbol,
):
    info_bits = source([
        batch_frames,
        num_info_bits_per_frame,
    ]).to(torch.long)

    coded_bits_flat = ldpc_encoder(info_bits)

    coded_bits = coded_bits_flat.reshape(
        batch_frames,
        rg.num_data_symbols,
        bits_per_qam_symbol,
    )

    x_freq = mapper(coded_bits).squeeze(-1)

    x_freq_sionna_input = x_freq.reshape(
        batch_frames,
        1,
        1,
        rg.num_data_symbols,
    )

    x_grid_sionna = rg_mapper(x_freq_sionna_input)

    x_time_sionna = ofdm_modulator(x_grid_sionna)

    y_time_clean = sionna_time_channel(x_time_sionna)

    y_time = awgn(y_time_clean, noise_power_target_tensor)

    y_grid_sionna = ofdm_demodulator(y_time)

    h_hat_sionna, err_var = ls_estimator(
        y_grid_sionna,
        noise_power_target_tensor,
    )

    x_hat_sionna, no_eff = lmmse_equalizer(
        y_grid_sionna,
        h_hat_sionna,
        err_var,
        noise_power_target_tensor,
    )

    equalized_data_freq = x_hat_sionna.reshape(
        batch_frames,
        rg.num_data_symbols,
    )

    llr = demapper(
        equalized_data_freq.unsqueeze(-1),
        no_eff.reshape(batch_frames, rg.num_data_symbols).unsqueeze(-1)
    )

    llr_flat = llr.reshape(
        batch_frames,
        num_coded_bits_per_frame,
    )

    decoded_info_bits = ldpc_decoder(llr_flat).to(torch.long)

    errors = decoded_info_bits != info_bits

    bit_errors = count_errors(info_bits, decoded_info_bits)
    frame_errors = torch.any(errors, dim=1)

    return (
        int(bit_errors.item()),
        int(frame_errors.sum().item()),
        batch_frames * num_info_bits_per_frame,
    )

def run_one_snr(
    snr_db_target,
    total_frames,
    batch_size,
    device,
    source,
    ldpc_encoder,
    mapper,
    rg,
    rg_mapper,
    ofdm_modulator,
    sionna_time_channel,
    awgn,
    ofdm_demodulator,
    ls_estimator,
    lmmse_equalizer,
    demapper,
    ldpc_decoder,
    num_info_bits_per_frame,
    num_coded_bits_per_frame,
    bits_per_qam_symbol,
):
    snr_linear_target = 10 ** (snr_db_target / 10)
    noise_power_target = 1 / snr_linear_target

    noise_power_target_tensor = torch.tensor(
        noise_power_target,
        dtype=torch.float32,
        device=device,
    )

    total_bit_errors = 0
    total_frame_errors = 0
    total_info_bits_done = 0
    total_frames_done = 0

    while total_frames_done < total_frames:
        batch_frames = min(batch_size, total_frames - total_frames_done)

        bit_errors, frame_errors, info_bits_done = run_one_batch(
            batch_frames,
            noise_power_target_tensor,
            source,
            ldpc_encoder,
            mapper,
            rg,
            rg_mapper,
            ofdm_modulator,
            sionna_time_channel,
            awgn,
            ofdm_demodulator,
            ls_estimator,
            lmmse_equalizer,
            demapper,
            ldpc_decoder,
            num_info_bits_per_frame,
            num_coded_bits_per_frame,
            bits_per_qam_symbol,
        )

        total_bit_errors += bit_errors
        total_frame_errors += frame_errors
        total_info_bits_done += info_bits_done
        total_frames_done += batch_frames

    ber = total_bit_errors / total_info_bits_done
    fer = total_frame_errors / total_frames_done

    return ber, fer

def build_system(
    num_subcarriers,
    num_ofdm_symbols,
    bits_per_qam_symbol,
    code_rate,
    cp_len,
    device,
):
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

    pilot_pattern = build_scattered_pilot_pattern(
        num_ofdm_symbols,
        num_subcarriers,
        device,
    )

    rg = ResourceGrid(
        num_ofdm_symbols=num_ofdm_symbols,
        fft_size=num_subcarriers,
        subcarrier_spacing=15e3,
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=cp_len,
        dc_null=False,
        pilot_pattern=pilot_pattern,
        pilot_ofdm_symbol_indices=[0, 7, 13],
        precision="single",
        device=device,
    )

    num_data_qam_symbols_per_frame = rg.num_data_symbols

    num_coded_bits_per_frame = (
        num_data_qam_symbols_per_frame * bits_per_qam_symbol
    )

    num_info_bits_per_frame = int(num_coded_bits_per_frame * code_rate)

    rg_mapper = ResourceGridMapper(
        rg,
        precision="single",
        device=device,
    )

    ls_estimator = LSChannelEstimator(
        rg,
        interpolation_type="lin",
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

    channel_model = TDL(
        model="A",
        delay_spread=300e-9,
        carrier_frequency=3.5e9,
        min_speed=10.0,
        max_speed=10.0,
        num_rx_ant=1,
        num_tx_ant=1,
        precision="single",
        device=device,
    )

    sionna_time_channel = TimeChannel(
        channel_model=channel_model,
        bandwidth=15e3 * num_subcarriers,
        num_time_samples=num_ofdm_symbols * (num_subcarriers + cp_len),
        normalize_channel=True,
        return_channel=False,
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
        l_min=sionna_time_channel.l_min,
        cyclic_prefix_length=cp_len,
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

    return {
        "source": source,
        "mapper": mapper,
        "demapper": demapper,
        "awgn": awgn,
        "rg": rg,
        "rg_mapper": rg_mapper,
        "ofdm_modulator": ofdm_modulator,
        "ofdm_demodulator": ofdm_demodulator,
        "ls_estimator": ls_estimator,
        "lmmse_equalizer": lmmse_equalizer,
        "sionna_time_channel": sionna_time_channel,
        "ldpc_encoder": ldpc_encoder,
        "ldpc_decoder": ldpc_decoder,
        "num_coded_bits_per_frame": num_coded_bits_per_frame,
        "num_info_bits_per_frame": num_info_bits_per_frame,
    }

def run_snr_sweep(
    snr_dbs,
    total_frames,
    batch_size,
    device,
    source,
    ldpc_encoder,
    mapper,
    rg,
    rg_mapper,
    ofdm_modulator,
    sionna_time_channel,
    awgn,
    ofdm_demodulator,
    ls_estimator,
    lmmse_equalizer,
    demapper,
    ldpc_decoder,
    num_info_bits_per_frame,
    num_coded_bits_per_frame,
    bits_per_qam_symbol,
):
    ber_results = []

    with torch.no_grad():
        for snr_db_target in snr_dbs:
            ber, fer = run_one_snr(
                snr_db_target,
                total_frames,
                batch_size,
                device,
                source,
                ldpc_encoder,
                mapper,
                rg,
                rg_mapper,
                ofdm_modulator,
                sionna_time_channel,
                awgn,
                ofdm_demodulator,
                ls_estimator,
                lmmse_equalizer,
                demapper,
                ldpc_decoder,
                num_info_bits_per_frame,
                num_coded_bits_per_frame,
                bits_per_qam_symbol,
            )

            ber_results.append(ber)

            print("target SNR dB:", snr_db_target)
            print("BER estimated channel:", ber)
            print("FER:", fer)
            print()

    return ber_results

def main():

    device = config.device

    sim_config = {
        "target_coded_bits": 10_000_000,
        "batch_size": 1000,
        "num_subcarriers": 64,
        "num_ofdm_symbols": 14,
        "bits_per_qam_symbol": 4,
        "code_rate": 1 / 2,
        "cp_len": 16,
        "snr_dbs": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    }

    target_coded_bits = sim_config["target_coded_bits"]
    batch_size = sim_config["batch_size"]
    num_subcarriers = sim_config["num_subcarriers"]
    num_ofdm_symbols = sim_config["num_ofdm_symbols"]
    bits_per_qam_symbol = sim_config["bits_per_qam_symbol"]
    code_rate = sim_config["code_rate"]
    cp_len = sim_config["cp_len"]
    snr_dbs = sim_config["snr_dbs"]

    system = build_system(
        num_subcarriers,
        num_ofdm_symbols,
        bits_per_qam_symbol,
        code_rate,
        cp_len,
        device,
    )

    source = system["source"]
    mapper = system["mapper"]
    demapper = system["demapper"]
    awgn = system["awgn"]
    rg = system["rg"]
    rg_mapper = system["rg_mapper"]
    ofdm_modulator = system["ofdm_modulator"]
    ofdm_demodulator = system["ofdm_demodulator"]
    ls_estimator = system["ls_estimator"]
    lmmse_equalizer = system["lmmse_equalizer"]
    sionna_time_channel = system["sionna_time_channel"]
    ldpc_encoder = system["ldpc_encoder"]
    ldpc_decoder = system["ldpc_decoder"]
    num_coded_bits_per_frame = system["num_coded_bits_per_frame"]
    num_info_bits_per_frame = system["num_info_bits_per_frame"]
    
    total_frames = math.ceil(target_coded_bits / num_coded_bits_per_frame)

    ber_results = run_snr_sweep(
        snr_dbs,
        total_frames,
        batch_size,
        device,
        source,
        ldpc_encoder,
        mapper,
        rg,
        rg_mapper,
        ofdm_modulator,
        sionna_time_channel,
        awgn,
        ofdm_demodulator,
        ls_estimator,
        lmmse_equalizer,
        demapper,
        ldpc_decoder,
        num_info_bits_per_frame,
        num_coded_bits_per_frame,
        bits_per_qam_symbol,
    )

if __name__ == "__main__":
    main()