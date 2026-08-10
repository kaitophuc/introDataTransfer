import torch
import torch.nn as nn

class OFDMNeuralReceiverTrainer:
    def __init__(
        self,
        system,
        bits_per_qam_symbol,
        device,
    ):
        self.system = system
        self.bits_per_qam_symbol = bits_per_qam_symbol
        self.device = device

        self.source = system["source"]
        self.mapper = system["mapper"]
        self.demapper = system["demapper"]
        self.awgn = system["awgn"]
        self.rg = system["rg"]
        self.rg_mapper = system["rg_mapper"]
        self.ofdm_modulator = system["ofdm_modulator"]
        self.ofdm_demodulator = system["ofdm_demodulator"]
        self.ls_estimator = system["ls_estimator"]
        self.lmmse_equalizer = system["lmmse_equalizer"]
        self.sionna_time_channel = system["sionna_time_channel"]
        self.ldpc_encoder = system["ldpc_encoder"]
        self.ldpc_decoder = system["ldpc_decoder"]

        self.num_coded_bits_per_frame = system["num_coded_bits_per_frame"]
        self.num_info_bits_per_frame = system["num_info_bits_per_frame"]

        self.num_spatial_streams = self.rg.num_tx * self.rg.num_streams_per_tx
        self.num_data_symbols_per_stream = self.rg.num_data_symbols
        self.num_total_data_symbols = self.num_spatial_streams * self.num_data_symbols_per_stream

        self.full_grid_receiver = FullGridNeuralReceiver(
            input_channels=23,
            bits_per_symbol=self.num_spatial_streams * bits_per_qam_symbol,
        ).to(device)

        self.full_grid_loss_function = torch.nn.BCEWithLogitsLoss()

        self.full_grid_optimizer = torch.optim.Adam(
            self.full_grid_receiver.parameters(),
            lr=1e-3,
        )

    def generate_received_grid_batch(self, batch_frames, noise_power):
        noise_power_tensor = torch.tensor(
            noise_power,
            dtype=torch.float32,
            device=self.device,
        )

        info_bits = self.source([
            batch_frames,
            self.num_info_bits_per_frame,
        ]).to(torch.long)

        coded_bits_flat = self.ldpc_encoder(info_bits)

        coded_bits = coded_bits_flat.reshape(
            batch_frames, self.rg.num_tx, self.rg.num_streams_per_tx,
            self.rg.num_data_symbols,
            self.bits_per_qam_symbol,
        )

        x_freq = self.mapper(coded_bits).squeeze(-1)

        x_grid_sionna = self.rg_mapper(x_freq)

        x_time_sionna = self.ofdm_modulator(x_grid_sionna)

        y_time_clean = self.sionna_time_channel(x_time_sionna)

        y_time = self.awgn(y_time_clean, noise_power_tensor)

        y_grid_sionna = self.ofdm_demodulator(y_time)

        return info_bits, coded_bits, y_grid_sionna, noise_power_tensor

    def run_classical_frontend_batch(self, batch_frames, noise_power):
        info_bits, coded_bits, y_grid_sionna, noise_power_tensor = (
            self.generate_received_grid_batch(batch_frames, noise_power)
        )

        h_hat_sionna, err_var = self.ls_estimator(
            y_grid_sionna,
            noise_power_tensor,
        )

        x_hat_sionna, no_eff = self.lmmse_equalizer(
            y_grid_sionna,
            h_hat_sionna,
            err_var,
            noise_power_tensor,
        )

        equalized_data_freq = x_hat_sionna.reshape(
            batch_frames,
            self.num_total_data_symbols,
        )
        return info_bits, coded_bits, equalized_data_freq, no_eff

    def generate_neural_training_batch(self, batch_frames, noise_power):
        info_bits, coded_bits, y_grid_sionna, noise_power_tensor = (
            self.generate_received_grid_batch(batch_frames, noise_power)
        )

        pilot_mask = self.rg.pilot_pattern.mask

        h_hat_sionna, err_var = self.ls_estimator(
            y_grid_sionna,
            noise_power_tensor,
        )

        grid_features, data_mask = make_neural_grid_features(
            y_grid_sionna,
            noise_power_tensor,
            pilot_mask,
            h_hat_sionna,
            err_var,
            self.rg.num_tx,
            self.rg.num_streams_per_tx,
        )

        labels = coded_bits.reshape(
            batch_frames,
            self.num_total_data_symbols,
            self.bits_per_qam_symbol,
        ).float()

        return grid_features, data_mask, labels

    def train_step(self, batch_frames, noise_power):
        grid_features, data_mask, labels = self.generate_neural_training_batch(
            batch_frames,
            noise_power,
        )

        self.full_grid_receiver.train()

        predicted_llr = self.full_grid_receiver(
            grid_features,
            data_mask,
        )

        loss = self.full_grid_loss_function(
            predicted_llr,
            labels,
        )

        self.full_grid_optimizer.zero_grad()
        loss.backward()
        self.full_grid_optimizer.step()

        with torch.no_grad():
            predicted_bits = (predicted_llr > 0).to(labels.dtype)

            bit_accuracy = torch.mean(
                (predicted_bits == labels).to(torch.float32)
            )

        return loss.item(), bit_accuracy.item()

    def train(
        self,
        num_training_steps,
        batch_size,
        training_snr_db_min,
        training_snr_db_max,
        print_every=20,
    ):
        last_loss = None
        last_accuracy = None

        for training_step in range(num_training_steps):

            random_snr_db = torch.empty(
                (),
                device=self.device,
            ).uniform_(training_snr_db_min, training_snr_db_max).item()

            snr_linear = 10 ** (random_snr_db / 10)
            noise_power = 1 / snr_linear
            
            loss, accuracy = self.train_step(
                batch_size,
                noise_power,
            )

            last_loss = loss
            last_accuracy = accuracy

            if training_step % print_every == 0:
                print(
                    "full-grid training step:",
                    training_step,
                    "train SNR dB:",
                    random_snr_db,
                    "loss:",
                    loss,
                    "accuracy:",
                    accuracy,
                )

        return last_loss, last_accuracy

    def evaluate_neural_batch(self, batch_frames, noise_power):
        info_bits, _, y_grid_sionna, noise_power_tensor = (
            self.generate_received_grid_batch(batch_frames, noise_power)
        )

        pilot_mask = self.rg.pilot_pattern.mask

        h_hat_sionna, err_var = self.ls_estimator(
            y_grid_sionna,
            noise_power_tensor,
        )

        grid_features, data_mask = make_neural_grid_features(
            y_grid_sionna,
            noise_power_tensor,
            pilot_mask,
            h_hat_sionna,
            err_var,
            self.rg.num_tx,
            self.rg.num_streams_per_tx,
        )

        self.full_grid_receiver.eval()

        with torch.no_grad():
            llr = self.full_grid_receiver(
                grid_features,
                data_mask,
            )

            llr_flat = llr.reshape(
                batch_frames,
                self.num_coded_bits_per_frame,
            )

            decoded_info_bits = self.ldpc_decoder(llr_flat).to(torch.long)

            errors = decoded_info_bits != info_bits

            bit_errors = torch.sum(errors)
            frame_errors = torch.any(errors, dim=1)

        return (
            int(bit_errors.item()),
            int(frame_errors.sum().item()),
            batch_frames * self.num_info_bits_per_frame,
        )

    def evaluate_neural_snr(self, snr_db, total_frames, batch_size):
        snr_linear = 10 ** (snr_db / 10)
        noise_power = 1 / snr_linear

        total_bit_errors = 0
        total_frame_errors = 0
        total_info_bits = 0
        total_frames_done = 0

        while total_frames_done < total_frames:
            batch_frames = min(
                batch_size,
                total_frames - total_frames_done,
            )

            bit_errors, frame_errors, info_bits = self.evaluate_neural_batch(
                batch_frames,
                noise_power,
            )

            total_bit_errors += bit_errors
            total_frame_errors += frame_errors
            total_info_bits += info_bits
            total_frames_done += batch_frames

        ber = total_bit_errors / total_info_bits
        fer = total_frame_errors / total_frames_done

        return ber, fer


    def evaluate_classical_batch(self, batch_frames, noise_power):
        info_bits, _, equalized_data_freq, no_eff = self.run_classical_frontend_batch(batch_frames, noise_power)

        llr = self.demapper(
            equalized_data_freq.unsqueeze(-1),
            no_eff.reshape(batch_frames, self.num_total_data_symbols).unsqueeze(-1)
        )

        llr_flat = llr.reshape(
            batch_frames,
            self.num_coded_bits_per_frame,
        )

        decoded_info_bits = self.ldpc_decoder(llr_flat).to(torch.long)

        errors = decoded_info_bits != info_bits

        bit_errors = torch.sum(errors)
        frame_errors = torch.any(errors, dim=1)

        return (
            int(bit_errors.item()),
            int(frame_errors.sum().item()),
            batch_frames * self.num_info_bits_per_frame,
        )

    def evaluate_classical_snr(self, snr_db, total_frames, batch_size):
        snr_linear = 10 ** (snr_db / 10)
        noise_power = 1 / snr_linear

        total_bit_errors = 0
        total_frame_errors = 0
        total_info_bits = 0
        total_frames_done = 0

        while total_frames_done < total_frames:
            batch_frames = min(
                batch_size,
                total_frames - total_frames_done,
            )

            bit_errors, frame_errors, info_bits = self.evaluate_classical_batch(
                batch_frames,
                noise_power,
            )

            total_bit_errors += bit_errors
            total_frame_errors += frame_errors
            total_info_bits += info_bits
            total_frames_done += batch_frames

        ber = total_bit_errors / total_info_bits
        fer = total_frame_errors / total_frames_done

        return ber, fer

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        correction = self.net(x)
        return self.activation(residual + correction)

class FullGridNeuralReceiver(nn.Module):
    def __init__(self, input_channels=23, hidden_channels=96, bits_per_symbol=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.Conv2d(hidden_channels, bits_per_symbol, kernel_size=1),
        )

    def forward(self, grid_features, data_mask):
        grid_llr = self.net(grid_features)

        batch_frames = grid_llr.shape[0]
        num_streams = data_mask.shape[0]
        num_ofdm_symbols = data_mask.shape[1]
        num_subcarriers = data_mask.shape[2]
        bits_per_symbol = grid_llr.shape[1] // num_streams

        grid_llr = grid_llr.reshape(
            batch_frames,
            num_streams,
            bits_per_symbol,
            num_ofdm_symbols,
            num_subcarriers,
        )

        grid_llr = grid_llr.permute(0, 1, 3, 4, 2)

        data_llr = grid_llr[:, data_mask.bool(), :]

        return data_llr

def make_neural_grid_features(y_grid_sionna, noise_power_tensor, pilot_mask, h_hat_sionna, err_var, num_tx, num_streams_per_tx):
    batch_frames = y_grid_sionna.shape[0]
    num_rx = y_grid_sionna.shape[1]
    num_rx_ant = y_grid_sionna.shape[2]
    num_ofdm_symbols = y_grid_sionna.shape[3]
    num_subcarriers = y_grid_sionna.shape[4]
    num_stream_masks = num_tx * num_streams_per_tx

    y_grid = y_grid_sionna.reshape(
        batch_frames,
        num_rx * num_rx_ant,
        num_ofdm_symbols,
        num_subcarriers,
    )

    h_hat_grid = h_hat_sionna.reshape(
        batch_frames,
        num_rx * num_rx_ant * num_stream_masks,
        num_ofdm_symbols,
        num_subcarriers,
    )

    err_var_grid = err_var.reshape(
        batch_frames,
        num_rx * num_rx_ant * num_stream_masks,
        num_ofdm_symbols,
        num_subcarriers,
    )

    real_feature = torch.real(y_grid)
    imag_feature = torch.imag(y_grid)
    received_features = torch.cat(
        [
            real_feature,
            imag_feature,
        ],
        dim=1,
    )

    device = y_grid.device 
    
    noise_feature = torch.full(
        (batch_frames, 1, num_ofdm_symbols, num_subcarriers),
        torch.log(noise_power_tensor).item(),
        dtype=torch.float32,
        device=device,
    )

    pilot_mask = pilot_mask.to(device=device, dtype=torch.bool)
    pilot_mask = pilot_mask.reshape(
        num_stream_masks,
        num_ofdm_symbols,
        num_subcarriers
    )

    data_mask = ~pilot_mask

    pilot_feature = pilot_mask.to(torch.float32).unsqueeze(0).expand(
        batch_frames,
        -1,
        -1,
        -1,
    )

    data_feature = data_mask.to(torch.float32).unsqueeze(0).expand(
        batch_frames,
        -1,
        -1,
        -1,
    )

    time_index = torch.linspace(-1.0, 1.0, num_ofdm_symbols, device=device)
    freq_index = torch.linspace(-1.0, 1.0, num_subcarriers, device=device)

    time_feature = time_index.view(1, 1, num_ofdm_symbols, 1).expand(
        batch_frames,
        -1,
        -1,
        num_subcarriers,
    )

    freq_feature = freq_index.view(1, 1, 1, num_subcarriers).expand(
        batch_frames,
        -1,
        num_ofdm_symbols,
        -1,
    )

    h_hat_real_feature = torch.real(h_hat_grid)
    h_hat_imag_feature = torch.imag(h_hat_grid)
    err_var_feature = torch.log(err_var_grid + 1e-12)

    grid_features = torch.cat(
        [
            received_features,
            noise_feature,
            pilot_feature,
            data_feature,
            time_feature,
            freq_feature,
            h_hat_real_feature,
            h_hat_imag_feature,
            err_var_feature,
        ],
        dim=1,
    )

    return grid_features, data_mask