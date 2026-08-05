import torch
import torch.nn as nn

class NeuralDemapper(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, features):
        return self.net(features)

class NeuralReceiverTrainer:
    def __init__(
        self,
        neural_demapper,
        system,
        bits_per_qam_symbol,
        device,
    ):
        self.neural_demapper = neural_demapper
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

        self.loss_function = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.neural_demapper.parameters(),
            lr=1e-3,
        )

    def run_receiver_front_end(self, batch_frames, noise_power):
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
            batch_frames,
            self.rg.num_data_symbols,
            self.bits_per_qam_symbol,
        )

        x_freq = self.mapper(coded_bits).squeeze(-1)

        x_freq_sionna_input = x_freq.reshape(
            batch_frames,
            1,
            1,
            self.rg.num_data_symbols,
        )

        x_grid_sionna = self.rg_mapper(x_freq_sionna_input)

        x_time_sionna = self.ofdm_modulator(x_grid_sionna)

        y_time_clean = self.sionna_time_channel(x_time_sionna)

        y_time = self.awgn(y_time_clean, noise_power_tensor)

        y_grid_sionna = self.ofdm_demodulator(y_time)

        h_hat_sionna, err_var = self.ls_estimator(
            y_grid_sionna,
            noise_power_tensor,
        )

        y_grid_no_sionna_dims = y_grid_sionna.squeeze(1).squeeze(1)

        grid_features = torch.stack(
            [
                torch.real(y_grid_no_sionna_dims),
                torch.imag(y_grid_no_sionna_dims),
            ],
            dim=1,
        )

        noise_feature = torch.full_like(
            torch.real(y_grid_no_sionna_dims),
            torch.log(noise_power_tensor),
        )

        pilot_mask = self.rg.pilot_pattern.mask.squeeze(0).squeeze(0)
        pilot_mask = pilot_mask.to(device=self.device, dtype=torch.float32)

        pilot_feature = pilot_mask.unsqueeze(0).expand(
            batch_frames,
            -1,
            -1,
        )

        data_mask = (~pilot_mask.bool()).to(torch.float32)

        data_feature = data_mask.unsqueeze(0).expand(
            batch_frames,
            -1,
            -1,
        )

        grid_features = torch.cat(
            [
                grid_features,
                noise_feature.unsqueeze(1),
            ],
            dim=1,
        )
        
        grid_features = torch.cat(
            [
                grid_features,
                pilot_feature.unsqueeze(1),
            ],
            dim=1,
        )

        grid_features = torch.cat(
            [
                grid_features,
                data_feature.unsqueeze(1),
            ],
            dim=1,
        )

        fake_grid_llr = torch.zeros(
            batch_frames,
            self.bits_per_qam_symbol,
            self.rg.num_ofdm_symbols,
            self.rg.fft_size,
            device=self.device,
        )

        data_positions = data_mask.bool()

        fake_data_llr = fake_grid_llr.permute(0, 2, 3, 1)[:, data_positions, :]

        print("fake_grid_llr shape:", fake_grid_llr.shape)
        print("data_positions shape:", data_positions.shape)
        print("fake_data_llr shape:", fake_data_llr.shape)
        print("coded_bits shape:", coded_bits.shape)
        raise SystemExit

        x_hat_sionna, no_eff = self.lmmse_equalizer(
            y_grid_sionna,
            h_hat_sionna,
            err_var,
            noise_power_tensor,
        )

        equalized_data_freq = x_hat_sionna.reshape(
            batch_frames,
            self.rg.num_data_symbols,
        )
        return info_bits, coded_bits, equalized_data_freq, no_eff


    def generate_training_batch(self, batch_frames, noise_power):

        info_bits, coded_bits, equalized_data_freq, no_eff = self.run_receiver_front_end(batch_frames, noise_power)
        
        neural_features = make_neural_features(
            equalized_data_freq,
            no_eff,
        )

        features_flat = flatten_neural_features(neural_features)

        labels_flat = flatten_coded_bits(coded_bits)

        return features_flat, labels_flat

    def train_step(self, batch_frames, noise_power):
        features_flat, labels_flat = self.generate_training_batch(
            batch_frames,
            noise_power,
        )

        self.neural_demapper.train()

        predicted_llr_flat = self.neural_demapper(features_flat)

        loss = self.loss_function(
            predicted_llr_flat,
            labels_flat,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            predicted_bits = (predicted_llr_flat > 0).to(labels_flat.dtype)

            bit_accuracy = torch.mean(
                (predicted_bits == labels_flat).to(torch.float32)
            )

        return loss.item(), bit_accuracy.item()

    def evaluate_receiver_batch(self, batch_frames, noise_power):
        info_bits, _, equalized_data_freq, no_eff = self.run_receiver_front_end(batch_frames, noise_power)

        neural_features = make_neural_features(
            equalized_data_freq,
            no_eff,
        )

        features_flat = flatten_neural_features(neural_features)

        self.neural_demapper.eval()

        with torch.no_grad():
            llr_flat = self.neural_demapper(features_flat).reshape(
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

    def evaluate_snr(self, snr_db, total_frames, batch_size):
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

            bit_errors, frame_errors, info_bits = self.evaluate_receiver_batch(
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

    def train(
        self,
        num_training_steps,
        batch_size,
        training_snr_db,
        print_every=20,
    ):
        snr_linear = 10 ** (training_snr_db / 10)
        noise_power = 1 / snr_linear

        last_loss = None
        last_accuracy = None

        for training_step in range(num_training_steps):
            loss, accuracy = self.train_step(
                batch_size,
                noise_power,
            )

            last_loss = loss
            last_accuracy = accuracy

            if training_step % print_every == 0:
                print(
                    "training step:",
                    training_step,
                    "loss:",
                    loss,
                    "accuracy:",
                    accuracy,
                )

        return last_loss, last_accuracy

    def evaluate_classical_batch(self, batch_frames, noise_power):
        info_bits, _, equalized_data_freq, no_eff = self.run_receiver_front_end(batch_frames, noise_power)

        llr = self.demapper(
            equalized_data_freq.unsqueeze(-1),
            no_eff.reshape(batch_frames, self.rg.num_data_symbols).unsqueeze(-1)
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

def make_neural_features(equalized_data_freq, no_eff):
    no_eff = no_eff.reshape(equalized_data_freq.shape)
    no_eff = torch.clamp(no_eff, min=1e-12)

    return torch.stack(
        [
            torch.real(equalized_data_freq),
            torch.imag(equalized_data_freq),
            torch.log(no_eff),
        ],
        dim=-1,
    )

def flatten_neural_features(neural_features):
    return neural_features.reshape(-1, neural_features.shape[-1])

def flatten_coded_bits(coded_bits):
    return coded_bits.float().reshape(-1, coded_bits.shape[-1])