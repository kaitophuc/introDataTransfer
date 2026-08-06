import math
import torch
import numpy as np
from sionna.phy import config
from sionna.phy.channel import AWGN, TimeChannel
from sionna.phy.mapping import BinarySource, Mapper, Demapper
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
from chart_maker import plot_receiver_comparison
from neural_receiver import OFDMNeuralReceiverTrainer
import csv

training_seed = 0
evaluation_seeds = [1000, 1001, 1002]

config.seed = training_seed
should_train_model = False
checkpoint_path = "checkpoints/full_grid_receiver.pt"

def build_comb_scattered_pilot_pattern(num_ofdm_symbols, num_subcarriers, device):
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

def build_ofdm_system(
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

    pilot_pattern = build_comb_scattered_pilot_pattern(
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

def main():

    device = config.device

    sim_config = {
        "target_coded_bits": 1_000_000_000,
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

    ofdm_system = build_ofdm_system(
        num_subcarriers,
        num_ofdm_symbols,
        bits_per_qam_symbol,
        code_rate,
        cp_len,
        device,
    )

    num_coded_bits_per_frame = ofdm_system["num_coded_bits_per_frame"]
    
    total_frames = math.ceil(target_coded_bits / num_coded_bits_per_frame)

    trainer = OFDMNeuralReceiverTrainer(
        system=ofdm_system,
        bits_per_qam_symbol=bits_per_qam_symbol,
        device=device,
    )

    if should_train_model:

        trainer.train(
            num_training_steps=2000,
            batch_size=batch_size,
            training_snr_db_min=6.0,
            training_snr_db_max=16.0,
            print_every=100,
        )

        torch.save(
            trainer.full_grid_receiver.state_dict(),
            "checkpoints/full_grid_receiver.pt",
        )

        print("saved model:", checkpoint_path)

    else:
        trainer.full_grid_receiver.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device,
            )
        )

        print("loaded model:", checkpoint_path)

    neural_bers = []
    neural_fers = []
    classical_bers = []
    classical_fers = []

    for snr_db in snr_dbs:
        neural_ber_sum = 0.0
        neural_fer_sum = 0.0
        classical_ber_sum = 0.0
        classical_fer_sum = 0.0

        for evaluation_seed in evaluation_seeds:
            config.seed = evaluation_seed

            neural_ber, neural_fer = trainer.evaluate_neural_snr(
                snr_db=snr_db,
                total_frames=total_frames,
                batch_size=batch_size,
            )

            config.seed = evaluation_seed

            classical_ber, classical_fer = trainer.evaluate_classical_snr(
                snr_db=snr_db,
                total_frames=total_frames,
                batch_size=batch_size,
            )

            neural_ber_sum += neural_ber
            neural_fer_sum += neural_fer
            classical_ber_sum += classical_ber
            classical_fer_sum += classical_fer

        num_evaluation_seeds = len(evaluation_seeds)

        neural_ber = neural_ber_sum / num_evaluation_seeds
        neural_fer = neural_fer_sum / num_evaluation_seeds
        classical_ber = classical_ber_sum / num_evaluation_seeds
        classical_fer = classical_fer_sum / num_evaluation_seeds

        neural_bers.append(neural_ber)
        neural_fers.append(neural_fer)
        classical_bers.append(classical_ber)
        classical_fers.append(classical_fer)

        print("target SNR dB:", snr_db)
        print("BER neural receiver:", neural_ber)
        print("FER neural receiver:", neural_fer)
        print("BER classical receiver:", classical_ber)
        print("FER classical receiver:", classical_fer)
        print()

    chart_path = plot_receiver_comparison(
        snr_dbs,
        neural_bers,
        classical_bers,
        neural_fers,
        classical_fers,
    )
    print("saved chart:", chart_path)

    csv_path = "results/neural_vs_classical_receiver.csv"

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "training_seed",
            "evaluation_seeds",
            "target_coded_bits",
            "batch_size",
            "num_subcarriers",
            "num_ofdm_symbols",
            "bits_per_qam_symbol",
            "code_rate",
            "cp_len",
            "snr_db",
            "neural_ber",
            "neural_fer",
            "classical_ber",
            "classical_fer",
        ])

        for row in zip(
            snr_dbs,
            neural_bers,
            neural_fers,
            classical_bers,
            classical_fers,
        ):
            writer.writerow([
                training_seed,
                ";".join(str(seed) for seed in evaluation_seeds),
                target_coded_bits,
                batch_size,
                num_subcarriers,
                num_ofdm_symbols,
                bits_per_qam_symbol,
                code_rate,
                cp_len,
                *row,
            ])

    print("saved results:", csv_path)

if __name__ == "__main__":
    main()